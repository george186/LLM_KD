import transformers
from transformers import TrainerCallback
import os
import gc
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import math

# ==============================================================================
# 1. 核心架构组件 (Bridge & Layers)
#    这些是我们之前定义的，用于将 Llama 参数映射到 GPT2
# ==============================================================================

class GroupedAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, groups=64, mid_ratio=1.0):
        """
        多层分组卷积适配器。
        
        Args:
            in_dim: 输入维度
            out_dim: 输出维度
            groups: 分组数 (推荐 64)。必须能整除输入和输出维度。
            mid_ratio: 中间层维度的倍率。默认为1，即中间层维度等于输入维度。
        """
        super().__init__()
        
        # 1. 计算中间维度
        mid_dim = int(in_dim * mid_ratio)
        
        # 2. 动态调整 groups (确保能被整除)
        # 找到 in_dim, mid_dim, out_dim 的最大公约数，确保 groups 不超过它
        limit = math.gcd(in_dim, math.gcd(mid_dim, out_dim))
        if groups > limit:
            print(f"[Warning] Groups {groups} is too large for dims ({in_dim}->{out_dim}). Adjusted to {limit}.")
            groups = limit
            
        self.groups = groups

        # 3. 定义网络结构
        # 结构: Pointwise Conv -> GroupNorm -> GELU -> Pointwise Conv
        self.net = nn.Sequential(
            # Layer 1: 变换/混合特征
            nn.Conv1d(in_dim, mid_dim, kernel_size=1, groups=groups, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=mid_dim), # GroupNorm 配合 GroupConv 效果极佳
            nn.GELU(),
            
            # Layer 2: 映射到目标维度
            nn.Conv1d(mid_dim, out_dim, kernel_size=1, groups=groups, bias=False)
        )
        
        # 4. 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x Input: (Batch, Seq, Dim)
        
        # 1. 转置为 Conv1d 格式: (Batch, Dim, Seq)
        x = x.transpose(1, 2)
        
        # 2. 前向计算
        x = self.net(x)
        
        # 3. 还原格式: (Batch, Seq, Dim)
        x = x.transpose(1, 2)
        return x


class DimensionAdaptedLoRALayer(nn.Module):
    def __init__(self, original_layer, params_dict, rank=16, alpha=32, layer_type="attn_q", opt_dim=None, llama_dim=4096):
        """
        Args:
            original_layer: OPT 的原始 Linear 层
            params_dict: 包含 Llama LoRA 参数的字典
            layer_type: 区分是 q, k, v 还是 o
            opt_dim: OPT 模型的隐藏层维度 (例如 768 或 2048)
            llama_dim: Llama 模型的隐藏层维度 (4096)
        """
        super().__init__()
        self.base_layer = original_layer
        self.scaling = alpha / rank
        self.layer_type = layer_type
        
        # 1. 获取 OPT 维度
        # OPT 使用 nn.Linear，所以是 in_features 和 out_features
        opt_in = original_layer.in_features
        opt_out = original_layer.out_features

        # 2. 冻结原 OPT 参数
        self.base_layer.weight.requires_grad = False
        if getattr(self.base_layer, 'bias', None) is not None:
            self.base_layer.bias.requires_grad = False

        # 3. 定义数据适配器 (Data Adapters)
        # Up Projector: 将 OPT 的输入维度 (e.g. 2048) 升维到 Llama (4096)
        # 注意: 只需要适配 Input 维度，因为 LoRA 是旁路计算
        groups = 64  
        self.adapter_up = GroupedAdapter(opt_in, llama_dim, groups=groups, mid_ratio=1.0)
        
        # Down Projector: 将 Llama 的输出 (4096) 降维回 OPT 的输出维度
        # 注意: LoRA B 矩阵输出是 4096，我们需要将其映射回 opt_out 以便与 base_layer 相加
        self.adapter_down = GroupedAdapter(llama_dim, opt_out, groups=groups, mid_ratio=1.0)

        # nn.init.zeros_(self.adapter_down.net[-1].weight)
        
        # 5. 直接加载 Llama LoRA 参数 (保持原始维度)
        dtype = torch.float32
        
        # 根据 layer_type 选择参数
        if layer_type == "attn_q":
            self.lora_A_q = nn.Parameter(params_dict['q'][0].clone().detach().to(dtype))
            self.lora_B_q = nn.Parameter(params_dict['q'][1].clone().detach().to(dtype))
        elif layer_type == "attn_k":
            self.lora_A_k = nn.Parameter(params_dict['k'][0].clone().detach().to(dtype))
            self.lora_B_k = nn.Parameter(params_dict['k'][1].clone().detach().to(dtype))
        elif layer_type == "attn_v":
            self.lora_A_v = nn.Parameter(params_dict['v'][0].clone().detach().to(dtype))
            self.lora_B_v = nn.Parameter(params_dict['v'][1].clone().detach().to(dtype))
        else: # proj
            self.lora_A = nn.Parameter(params_dict['o'][0].clone().detach().to(dtype))
            self.lora_B = nn.Parameter(params_dict['o'][1].clone().detach().to(dtype))


    def forward(self, x):
        # 1. 计算原模型输出
        base_out = self.base_layer(x)

        # 2. 维度适配: OPT -> Llama
        # x_llama: (Batch, Seq, 4096)
        x_llama = self.adapter_up(x)

        if self.layer_type == "attn_q":
            lora_inner = (x_llama @ self.lora_A_q.T @ self.lora_B_q.T) * self.scaling

        elif self.layer_type == "attn_k":
            lora_inner = (x_llama @ self.lora_A_k.T @ self.lora_B_k.T) * self.scaling
            
        elif self.layer_type == "attn_v":
            lora_inner = (x_llama @ self.lora_A_v.T @ self.lora_B_v.T) * self.scaling

        else:
            lora_inner = (x_llama @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # 3. 维度还原: Llama -> OPT
        # lora_out: (Batch, Seq, opt_out)
        lora_out = self.adapter_down(lora_inner)
        
        # 4. 残差连接
        return base_out + lora_out

class LlamaToGPT2Bridge(nn.Module):
    def __init__(self, rank, target_dim, llama_dim=4096, projector_rank=128):
        """
        使用低秩分解 (Low-Rank Decomposition) 将 Llama 维度映射到 GPT2 维度。
        
        Args:
            rank: LoRA 的秩 (这里主要用于兼容接口，线性层自动处理 Batch 维度)
            target_dim: 目标维度 (GPT2 Dim, e.g., 1600)
            llama_dim: 输入维度 (Llama Dim, e.g., 4096)
            projector_rank: 中间瓶颈层的秩。用户要求设为 16。
                            (注意: 如果觉得信息损失太大，可以适当调大这个值，如 64 或 128)
        """
        super().__init__()
        
        # 1. 定义两个可训练矩阵 (通过 Linear 层实现)
        # 矩阵 B: (16, 4096) -> 对应 Linear(4096, 16)
        # 作用: 将 Llama 维度 (4096) 压缩到 中间维度 (16)
        self.down_proj = nn.Linear(llama_dim, projector_rank, bias=False)
        
        # 矩阵 A: (1600, 16) -> 对应 Linear(16, 1600)
        # 作用: 将 中间维度 (16) 映射到 GPT2 维度 (1600)
        self.up_proj = nn.Linear(projector_rank, target_dim, bias=False)
        
        # 可选: 中间是否加激活函数？
        # 如果是纯粹的"矩阵分解"数学定义，不需要激活函数 (Linear -> Linear)。
        # 如果为了增加非线性表达能力，可以加一个 GELU。
        # 这里为了严格贴合你"矩阵分解"的描述，我们不加激活函数，仅做线性变换。
        
        self._init_weights()

    def _init_weights(self):
        # 使用正交初始化或 Kaiming 初始化有助于保持梯度的稳定性
        # nn.init.kaiming_normal_(self.down_proj.weight, mode='fan_out', nonlinearity='linear')
        # nn.init.zeros_(self.up_proj.weight) # 类似于 LoRA 的初始化策略，B矩阵为0，使得初始状态接近 0 (可选)
        # 或者两个都用 Kaiming/Xavier
        nn.init.xavier_normal_(self.down_proj.weight) 
        nn.init.xavier_normal_(self.up_proj.weight) 

    def forward(self, x):
        """
        x: (Rank, Llama_Dim) 或者是 (Llama_Dim, Rank)
        """
        # 1. 形状检查与转置
        # 我们期望输入 x 的形状是 (Batch/Rank, Input_Features)
        # 即 (16, 4096)
        is_transposed = False
        if x.shape[0] > x.shape[1]: 
            # 如果输入是 (4096, 16)，说明是 B 矩阵，转置为 (16, 4096)
            x = x.T 
            is_transposed = True
            
        # 2. 线性投影 (Matrix Multiplication)
        # Step 1: (16, 4096) @ (4096, 16) -> (16, 16)
        mid = self.down_proj(x)
        
        # Step 2: (16, 16) @ (16, 1600) -> (16, 1600)
        out = self.up_proj(mid)
        
        # 3. 还原形状
        if is_transposed:
            out = out.T
            
        return out
    
# ==============================================================================
# 2. 升级版 LoRA 层: 支持分离的 Q/K/V 管理
# ==============================================================================
class SplitBridgedLoRALayer(nn.Module):
    def __init__(self, original_layer, params_dict, rank=16, alpha=32, layer_type="attn"):
        """
        Args:
            params_dict: 包含 'q', 'k', 'v' (针对 attn) 或 'o' (针对 proj) 的 A/B 参数
            layer_type: 'attn' 或 'proj'
        """
        super().__init__()
        self.base_layer = original_layer
        self.scaling = alpha / rank
        self.layer_type = layer_type
        self.rank = rank
        
        gpt2_in = original_layer.in_features   # 1600
        gpt2_out = original_layer.out_features  # 4800 (attn) or 1600 (proj)

        # 冻结原 GPT2 参数
        self.base_layer.weight.requires_grad = False
        if getattr(self.base_layer, 'bias', None) is not None:
            self.base_layer.bias.requires_grad = False

        # 强制使用 FP32 存储 Llama 参数以保证计算稳定
        dtype = torch.float32

        if layer_type == "attn_q":
            # 分离存储 Q, K, V
            self.lora_A_q = nn.Parameter(params_dict['q'][0].clone().detach().to(dtype))
            self.lora_B_q = nn.Parameter(params_dict['q'][1].clone().detach().to(dtype))

            self.bridge_A_q = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_B_q = LlamaToGPT2Bridge(rank, gpt2_out)

        elif layer_type == "attn_k":
            self.lora_A_k = nn.Parameter(params_dict['k'][0].clone().detach().to(dtype))
            self.lora_B_k = nn.Parameter(params_dict['k'][1].clone().detach().to(dtype))
            self.bridge_A_k = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_B_k = LlamaToGPT2Bridge(rank, gpt2_out)

        elif layer_type == "attn_v":
            self.lora_A_v = nn.Parameter(params_dict['v'][0].clone().detach().to(dtype))
            self.lora_B_v = nn.Parameter(params_dict['v'][1].clone().detach().to(dtype))
            self.bridge_A_v = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_B_v = LlamaToGPT2Bridge(rank, gpt2_out)

        else: # proj
            self.lora_A = nn.Parameter(params_dict['o'][0].clone().detach().to(dtype))
            self.lora_B = nn.Parameter(params_dict['o'][1].clone().detach().to(dtype))
            
            self.bridge_A = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_B = LlamaToGPT2Bridge(rank, gpt2_out)

    def forward(self, x):
        if self.layer_type == "attn_q":
            # 1. 获取动态投影后的矩阵
            # bridge_A_q: (16, 4096) -> (16, 1600)
            A = self.bridge_A_q(self.lora_A_q)
            # bridge_B_q: (4096, 16) -> (1600, 16)
            B = self.bridge_B_q(self.lora_B_q)
            
        elif self.layer_type == "attn_k":
            A = self.bridge_A_k(self.lora_A_k)
            B = self.bridge_B_k(self.lora_B_k)
            
        elif self.layer_type == "attn_v":
            A = self.bridge_A_v(self.lora_A_v)
            B = self.bridge_B_v(self.lora_B_v)

        else:
            # 处理 Output Proj
            A = self.bridge_A(self.lora_A)
            B = self.bridge_B(self.lora_B)

        lora_term = (x @ A.T @ B.T) * self.scaling
            
        return self.base_layer(x) + lora_term



# ==============================================================================
# 3. 封装类：HybridGeneralClient,不回滚
#    结合了原来的流程控制和新的模型架构
# ==============================================================================

class HybridGeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, llama_inject_idx):
        self.client_id = client_id
        self.model = model 
        self.llama_inject_idx = llama_inject_idx
        
        self.local_data_path = os.path.join(data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size):
        if local_val_set_size > 0:
            local_train_val = self.local_data["train"].train_test_split(
                test_size=local_val_set_size, shuffle=True, seed=42
            )
            self.local_train_dataset = (
                local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            )
            self.local_eval_dataset = (
                local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
            )
        else:
            self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
            self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size, #在每个GPU/CPU上用于训练的实际批次大小
            gradient_accumulation_steps=gradient_accumulation_steps, #梯度累积的步数。
            dataloader_pin_memory=False,
            warmup_steps=0, #不使用学习率预热
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            # lr_scheduler_type="constant", #使用恒定学习率
            lr_scheduler_type="cosine",
            fp16=True, #混合精度训练
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if self.local_val_set_size > 0 else "no", #按训练步数间隔在验证集上评估模型
            save_strategy="no", #按训练步数间隔来保存模型检查点
            eval_steps=200 if self.local_val_set_size > 0 else None,
            output_dir=self.local_output_dir,
            save_total_limit=0,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False, #在训练结束时，是否加载在验证集上性能最好的那个模型作为最终模型
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length, #是否在构建批次时，将长度相近的样本分组在一起。
            dataloader_drop_last=False,
            gradient_checkpointing=False, #开启梯度检查点减少显存消耗
            # report_to="none", #不生成runs日志
        )
        
        self.local_trainer = transformers.Trainer(
            model=self.model,
            train_dataset=self.local_train_dataset,
            eval_dataset=self.local_eval_dataset,
            args=self.train_args,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            ),
        )



    def train(self):
        self.local_trainer.train()

    def save_client_checkpoint(self, output_dir, filename="client_state.pt"):
        """
        保存客户端的所有可训练参数（Llama LoRA + Bridge 参数）。
        这用于在下一轮训练时恢复客户端的个性化状态。
        """
        os.makedirs(output_dir, exist_ok=True)
        # 只保存 requires_grad=True 的参数，节省磁盘空间
        trainable_state = {k: v.cpu() for k, v in self.model.state_dict().items() if "lora_" in k or "bridge_" in k}
        torch.save(trainable_state, os.path.join(output_dir, filename))

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        """
        [修改点]
        1. 仅提取并保存训练后的 Llama 格式参数。
        2. 删除回滚逻辑，保持 self.model 为最新状态。
        """
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        
        # 1. 提取 Llama 格式参数 (基于新的映射逻辑：后32层)
        new_adapter_weight_llama_fmt = extract_llama_state_dict_from_gpt2(
            self.model, self.llama_inject_idx
        )
        
        single_output_dir = os.path.join(self.output_dir, "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight_llama_fmt, single_output_dir + "/pytorch_model.bin")
        
        # [修改点] 移除了 load_state_dict(params_dict_old) 的回滚操作
        
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id
        
        del new_adapter_weight_llama_fmt
        
        self._clean_training_memory()
        gc.collect()
        torch.cuda.empty_cache()
        
        # 返回更新后的 GPT2 模型
        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id
    
    def _clean_training_memory(self):
        """清理 Trainer 相关的显存，保留模型参数"""
        if hasattr(self, 'local_trainer'):
            del self.local_trainer
        
        if hasattr(self, 'local_train_dataset'):
            del self.local_train_dataset
        if hasattr(self, 'local_eval_dataset'):
            del self.local_eval_dataset

        gc.collect()
        torch.cuda.empty_cache()

# ==============================================================================
# 确保 GPT2 的前 16 层是冻结的，只有后 32 层注入了 SplitBridgedLoRALayer。
# ==============================================================================

def inject_and_freeze_last_32(gpt2_model, llama_params, rank, device, llama_inject_idx):
    """
    (补充函数) 适配后32层映射的注入逻辑
    """
    llama_layer_count = len(llama_params) # 32
    total_gpt2_layers = 24 
    start_layer_idx = total_gpt2_layers - llama_layer_count 
    
    print(f"\n[Injection] Injecting into last {llama_layer_count} layers (Idx {start_layer_idx}-{total_gpt2_layers-1})...")

    # 创建注入映射：从Llama2的32层中选择24层
    # 规则：每3层注入，然后跳过1层
    llama_to_gpt2_map = []
    gpt2_layer_idx = 0
    gpt2_idx = []

    for llama_idx in llama_inject_idx:   
        if gpt2_layer_idx >= total_gpt2_layers:
            break
            
        llama_to_gpt2_map.append((gpt2_layer_idx, llama_idx))
        gpt2_idx.append(gpt2_layer_idx)
        gpt2_layer_idx += 1
    
    print(f"[Injection] Mapping: {llama_to_gpt2_map}")

    inject_idx = 0
    for i, block in enumerate(gpt2_model.model.decoder.layers):
        # Case 1: 前 16 层 -> 彻底冻结
        if i not in gpt2_idx:
            for param in block.parameters():
                param.requires_grad = False
                
        # Case 2: 后 32 层 -> 注入并微调
        else:
            # Llama list index 从 0 开始
            
            p = llama_params[inject_idx]
            inject_idx = inject_idx + 1
            
            block.self_attn.q_proj = DimensionAdaptedLoRALayer(
                block.self_attn.q_proj, p['attn'], rank=rank, layer_type="attn_q"
            ).to(device)

            block.self_attn.k_proj = DimensionAdaptedLoRALayer(
                block.self_attn.k_proj, p['attn'], rank=rank, layer_type="attn_k"
            ).to(device)

            block.self_attn.v_proj = DimensionAdaptedLoRALayer(
                block.self_attn.v_proj, p['attn'], rank=rank, layer_type="attn_v"
            ).to(device)
            
            block.self_attn.out_proj = DimensionAdaptedLoRALayer(
                block.self_attn.out_proj, p['proj'], rank=rank, layer_type="proj"
            ).to(device)
            
            # 冻结 Block 内的其他组件
            for param in block.self_attn_layer_norm.parameters(): param.requires_grad = False
            for param in block.final_layer_norm.parameters(): param.requires_grad = False
            for param in block.fc1.parameters(): param.requires_grad = False
            for param in block.fc2.parameters():  param.requires_grad = False

    # 冻结全局组件
    for param in gpt2_model.model.decoder.embed_tokens.parameters(): param.requires_grad = False
    for param in gpt2_model.model.decoder.embed_positions.parameters(): param.requires_grad = False
    for param in gpt2_model.model.decoder.final_layer_norm.parameters(): param.requires_grad = False
    if hasattr(gpt2_model, 'lm_head'):
         for param in gpt2_model.lm_head.parameters(): param.requires_grad = False
         
    print(f"[Injection] Layers {gpt2_idx}: Trainable (LoRA).")


def extract_llama_state_dict_from_gpt2(gpt2_model, llama_inject_idx):
    """
    遍历 GPT2 模型，提取后32层的参数并重组为 Llama PEFT 格式。
    映射逻辑: GPT2 Layer [Total-32, Total-1] -> Llama Layer [0, 31]
    """
    llama_state_dict = OrderedDict()
    
    total_gpt2_layers = len(gpt2_model.model.decoder.layers) 

    
    # 我们只遍历 GPT2 的后 32 层
    for i in range(0, total_gpt2_layers):
        block = gpt2_model.model.decoder.layers[i]
        
        # 计算对应的 Llama 层索引 (0 到 31)
        llama_idx = llama_inject_idx[i]
        prefix = f"base_model.model.model.layers.{llama_idx}.self_attn"
        
        # 提取参数逻辑不变
        if isinstance(block.self_attn.q_proj, DimensionAdaptedLoRALayer):
            layer = block.self_attn.q_proj
            llama_state_dict[f"{prefix}.q_proj.lora_A.weight"] = layer.lora_A_q.detach().cpu()
            llama_state_dict[f"{prefix}.q_proj.lora_B.weight"] = layer.lora_B_q.detach().cpu()

        if isinstance(block.self_attn.k_proj, DimensionAdaptedLoRALayer):
            layer = block.self_attn.k_proj
            llama_state_dict[f"{prefix}.k_proj.lora_A.weight"] = layer.lora_A_k.detach().cpu()
            llama_state_dict[f"{prefix}.k_proj.lora_B.weight"] = layer.lora_B_k.detach().cpu()

        if isinstance(block.self_attn.v_proj, DimensionAdaptedLoRALayer):
            layer = block.self_attn.v_proj
            llama_state_dict[f"{prefix}.v_proj.lora_A.weight"] = layer.lora_A_v.detach().cpu()
            llama_state_dict[f"{prefix}.v_proj.lora_B.weight"] = layer.lora_B_v.detach().cpu()

        if isinstance(block.self_attn.out_proj, DimensionAdaptedLoRALayer):
            layer = block.self_attn.out_proj
            llama_state_dict[f"{prefix}.o_proj.lora_A.weight"] = layer.lora_A.detach().cpu()
            llama_state_dict[f"{prefix}.o_proj.lora_B.weight"] = layer.lora_B.detach().cpu()
            
    return llama_state_dict

def toggle_lora_alternating(model, round_idx):
    """
    根据轮次交替冻结 LoRA 的 A 矩阵和 B 矩阵。
    策略:
    - 偶数轮 (0, 2, ...): 训练 A，冻结 B。
    - 奇数轮 (1, 3, ...): 冻结 A，训练 B。
    
    注意: Bridge (卷积层) 始终保持训练状态，以适配冻结一侧带来的变化。
    """
    # 逻辑判断
    train_B = (round_idx % 2 == 0)
    train_A = not train_B

    # train_A = (round_idx % 2 == 0)
    # train_B = not train_A
    
    status_msg = f"Round {round_idx}: "
    status_msg += "Training [A] / Freezing [B]" if train_A else "Freezing [A] / Training [B]"
    print(f"\n[Alternating Strategy] {status_msg}")

    # 计数器
    frozen_count = 0
    trainable_count = 0

    for name, param in model.named_parameters():
        # 我们只操作 LoRA 参数，忽略 GPT2 原生参数（它们本来就是 frozen 的）
        
        # 1. 处理 A 矩阵 (lora_A, lora_A_q, lora_A_k, lora_A_v)
        if "lora_A" in name:
            param.requires_grad = train_A
        
        # 2. 处理 B 矩阵 (lora_B, lora_B_q, lora_B_k, lora_B_v)
        elif "lora_B" in name:
            param.requires_grad = train_B
            
        # 3. Bridge 参数 (bridge_A_q, bridge_B_k 等)
        # 建议: 始终保持 Bridge 可训练。
        # 原因: 即使 A 矩阵冻结了，A 的投影层 (Bridge) 依然可以微调以适配 GPT2。
        # elif "bridge_A" in name:
        #     param.requires_grad = True

        # elif "bridge_B" in name:
        #     param.requires_grad = True

        # elif "adapter_up" in name:
        #     param.requires_grad = train_B

        # elif "adapter_down" in name:
        #     param.requires_grad = train_B
            
        # 统计
        if param.requires_grad:
            trainable_count += param.numel()
        else:
            frozen_count += param.numel()

    print(f"[Alternating Strategy] Active Params: {trainable_count}, Frozen Params: {frozen_count}")


def toggle_lora_bridge(model):

    for name, param in model.named_parameters():
        # 我们只操作 LoRA 参数，忽略 GPT2 原生参数（它们本来就是 frozen 的）
        
        # 1. 处理 A 矩阵 (lora_A, lora_A_q, lora_A_k, lora_A_v)
        if "fc.lora" in name:
            param.requires_grad = True
        
        # 2. 处理 B 矩阵 (lora_B, lora_B_q, lora_B_k, lora_B_v)
        elif "self_attn" in name:
            param.requires_grad = False
            
