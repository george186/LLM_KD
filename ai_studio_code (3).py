import transformers
from transformers import TrainerCallback
import os
import gc
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
import torch.nn as nn

# ==============================================================================
# 1. 核心架构组件 (保持不变，确保已定义)
# ==============================================================================

class LlamaToGPT2Bridge(nn.Module):
    def __init__(self, rank, target_dim, kernel_size=31, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels=rank, out_channels=rank,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=rank, bias=True
        )
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool1d(target_dim)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        is_transposed = False
        if x.shape[0] > x.shape[1]: 
            x = x.T 
            is_transposed = True
        feat = self.pool(self.act(self.conv(x.unsqueeze(0))))
        out = feat.squeeze(0)
        if is_transposed: out = out.T
        return out

class SplitBridgedLoRALayer(nn.Module):
    def __init__(self, original_layer, params_dict, rank=16, alpha=32, layer_type="attn"):
        super().__init__()
        self.base_layer = original_layer
        self.scaling = alpha / rank
        self.layer_type = layer_type
        
        gpt2_in = original_layer.weight.shape[0]
        gpt2_out = original_layer.weight.shape[1]

        self.base_layer.weight.requires_grad = False
        if getattr(self.base_layer, 'bias', None) is not None:
            self.base_layer.bias.requires_grad = False

        dtype = torch.float32 # 强制 FP32

        if layer_type == "attn":
            self.lora_A_q = nn.Parameter(params_dict['q'][0].clone().detach().to(dtype))
            self.lora_B_q = nn.Parameter(params_dict['q'][1].clone().detach().to(dtype))
            self.lora_A_k = nn.Parameter(params_dict['k'][0].clone().detach().to(dtype))
            self.lora_B_k = nn.Parameter(params_dict['k'][1].clone().detach().to(dtype))
            self.lora_A_v = nn.Parameter(params_dict['v'][0].clone().detach().to(dtype))
            self.lora_B_v = nn.Parameter(params_dict['v'][1].clone().detach().to(dtype))
            
            self.bridge_A_q = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_A_k = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_A_v = LlamaToGPT2Bridge(rank, gpt2_in)
            
            target_out_per_head = gpt2_out // 3
            self.bridge_B_q = LlamaToGPT2Bridge(rank, target_out_per_head)
            self.bridge_B_k = LlamaToGPT2Bridge(rank, target_out_per_head)
            self.bridge_B_v = LlamaToGPT2Bridge(rank, target_out_per_head)

        else: # proj
            self.lora_A = nn.Parameter(params_dict['o'][0].clone().detach().to(dtype))
            self.lora_B = nn.Parameter(params_dict['o'][1].clone().detach().to(dtype))
            
            self.bridge_A = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_B = LlamaToGPT2Bridge(rank, gpt2_out)

    def forward(self, x):
        if self.layer_type == "attn":
            d_Q = (x @ self.bridge_A_q(self.lora_A_q).T @ self.bridge_B_q(self.lora_B_q).T) * self.scaling
            d_K = (x @ self.bridge_A_k(self.lora_A_k).T @ self.bridge_B_k(self.lora_B_k).T) * self.scaling
            d_V = (x @ self.bridge_A_v(self.lora_A_v).T @ self.bridge_B_v(self.lora_B_v).T) * self.scaling
            lora_term = torch.cat([d_Q, d_K, d_V], dim=-1)
        else:
            A = self.bridge_A(self.lora_A)
            B = self.bridge_B(self.lora_B)
            lora_term = (x @ A.T @ B.T) * self.scaling
        return self.base_layer(x) + lora_term

# ==============================================================================
# 2. 修改后的提取函数 (Proposal 1: 后32层映射)
# ==============================================================================

def extract_llama_state_dict_from_gpt2(gpt2_model, llama_num_layers=32):
    """
    遍历 GPT2 模型，提取后32层的参数并重组为 Llama PEFT 格式。
    映射逻辑: GPT2 Layer [Total-32, Total-1] -> Llama Layer [0, 31]
    """
    llama_state_dict = OrderedDict()
    
    total_gpt2_layers = len(gpt2_model.transformer.h) # 48
    start_layer_idx = total_gpt2_layers - llama_num_layers # 48 - 32 = 16
    
    # 我们只遍历 GPT2 的后 32 层
    for i in range(start_layer_idx, total_gpt2_layers):
        block = gpt2_model.transformer.h[i]
        
        # 计算对应的 Llama 层索引 (0 到 31)
        llama_idx = i - start_layer_idx
        prefix = f"base_model.model.model.layers.{llama_idx}.self_attn"
        
        # 提取参数逻辑不变
        if isinstance(block.attn.c_attn, SplitBridgedLoRALayer):
            layer = block.attn.c_attn
            llama_state_dict[f"{prefix}.q_proj.lora_A.default.weight"] = layer.lora_A_q.detach().cpu()
            llama_state_dict[f"{prefix}.q_proj.lora_B.default.weight"] = layer.lora_B_q.detach().cpu()
            
            llama_state_dict[f"{prefix}.k_proj.lora_A.default.weight"] = layer.lora_A_k.detach().cpu()
            llama_state_dict[f"{prefix}.k_proj.lora_B.default.weight"] = layer.lora_B_k.detach().cpu()
            
            llama_state_dict[f"{prefix}.v_proj.lora_A.default.weight"] = layer.lora_A_v.detach().cpu()
            llama_state_dict[f"{prefix}.v_proj.lora_B.default.weight"] = layer.lora_B_v.detach().cpu()

        if isinstance(block.attn.c_proj, SplitBridgedLoRALayer):
            layer = block.attn.c_proj
            llama_state_dict[f"{prefix}.o_proj.lora_A.default.weight"] = layer.lora_A.detach().cpu()
            llama_state_dict[f"{prefix}.o_proj.lora_B.default.weight"] = layer.lora_B.detach().cpu()
            
    return llama_state_dict

# ==============================================================================
# 3. 修改后的封装类 (Proposal 2: 不回滚)
# ==============================================================================

class HybridGeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, llama_num_layers=32):
        self.client_id = client_id
        self.model = model 
        self.llama_num_layers = llama_num_layers
        
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
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_pin_memory=False,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True, 
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False,
            gradient_checkpointing=True,
            report_to="none",
            remove_unused_columns=False
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

    def initiate_local_training(self):
        """
        [修改点] 
        因为不需要回滚，所以这里不再备份参数副本 params_dict_old。
        这可以显著减少显存占用。
        """
        self.model.config.use_cache = False
        # 清理之前的状态（如果有）
        if hasattr(self, 'params_dict_old'): del self.params_dict_old
        if hasattr(self, 'params_dict_new'): del self.params_dict_new
        
        # 这里仅做简单的标记或日志，不需要实际的数据复制
        pass

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        """
        [修改点]
        1. 仅提取并保存训练后的 Llama 格式参数。
        2. 删除回滚逻辑，保持 self.model 为最新状态。
        """
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        
        # 1. 提取 Llama 格式参数 (基于新的映射逻辑：后32层)
        new_adapter_weight_llama_fmt = extract_llama_state_dict_from_gpt2(
            self.model, self.llama_num_layers
        )
        
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight_llama_fmt, single_output_dir + "/pytorch_model.bin")
        
        # [修改点] 移除了 load_state_dict(params_dict_old) 的回滚操作
        
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id
        
        self._clean_training_memory()
        
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
# 注意: 为了配合上述改动，你的注入函数 inject_and_freeze 也需要相应调整。
# 必须确保 GPT2 的前 16 层是冻结的，只有后 32 层注入了 SplitBridgedLoRALayer。
# ==============================================================================

def inject_and_freeze_last_32(gpt2_model, llama_params, rank, device):
    """
    (补充函数) 适配后32层映射的注入逻辑
    """
    llama_layer_count = len(llama_params) # 32
    total_gpt2_layers = len(gpt2_model.transformer.h) # 48
    start_layer_idx = total_gpt2_layers - llama_layer_count # 16
    
    print(f"\n[Injection] Injecting into last {llama_layer_count} layers (Idx {start_layer_idx}-{total_gpt2_layers-1})...")

    for i, block in enumerate(gpt2_model.transformer.h):
        # Case 1: 前 16 层 -> 彻底冻结
        if i < start_layer_idx:
            for param in block.parameters():
                param.requires_grad = False
                
        # Case 2: 后 32 层 -> 注入并微调
        else:
            # Llama list index 从 0 开始
            llama_idx = i - start_layer_idx
            p = llama_params[llama_idx]
            
            block.attn.c_attn = SplitBridgedLoRALayer(
                block.attn.c_attn, p['attn'], rank=rank, layer_type="attn"
            ).to(device)
            
            block.attn.c_proj = SplitBridgedLoRALayer(
                block.attn.c_proj, p['proj'], rank=rank, layer_type="proj"
            ).to(device)
            
            # 冻结 Block 内的其他组件
            for param in block.ln_1.parameters(): param.requires_grad = False
            for param in block.ln_2.parameters(): param.requires_grad = False
            for param in block.mlp.parameters():  param.requires_grad = False

    # 冻结全局组件
    for param in gpt2_model.transformer.wte.parameters(): param.requires_grad = False
    for param in gpt2_model.transformer.wpe.parameters(): param.requires_grad = False
    for param in gpt2_model.ln_f.parameters(): param.requires_grad = False
    if hasattr(gpt2_model, 'lm_head'):
         for param in gpt2_model.lm_head.parameters(): param.requires_grad = False
         
    print(f"[Injection] Layers 0-{start_layer_idx-1}: Frozen.")
    print(f"[Injection] Layers {start_layer_idx}-{total_gpt2_layers-1}: Trainable (LoRA).")