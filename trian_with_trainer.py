import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel, 
    LlamaForCausalLM, 
    Trainer, 
    TrainingArguments, 
    TrainerCallback
)
from peft import get_peft_model, LoraConfig

# ==============================================================================
# 1. 引入之前的核心组件 (Bridge, Layer, Helpers)
#    为了代码简洁，这里假设你已经定义了以下类和函数。
#    在实际文件中，请把之前的类定义完整粘贴在这里。
# ==============================================================================
# - LlamaToGPT2Bridge
# - SplitBridgedLoRALayer
# - get_peft_llama
# - extract_params
# - inject_and_freeze
# - sync_back
# ------------------------------------------------------------------------------
# (此处省略类定义，直接复用上一轮回答中的代码实现)
# 为确保代码可运行，我会在下面提供简化的占位符，请替换为上一段代码的完整实现
# ------------------------------------------------------------------------------

# --- [请将上一轮完整代码中的类定义粘贴到这里] ---
# 必须包含: LlamaToGPT2Bridge, SplitBridgedLoRALayer
# 必须包含: get_peft_llama, extract_params, inject_and_freeze, sync_back

# 为了演示，我这里快速重写最核心的部分，实际使用请用上一轮的完整版
# ==============================================================================
# 1. 通用组件: 桥接器 (保持不变)
# ==============================================================================
class LlamaToGPT2Bridge(nn.Module):
    def __init__(self, rank, target_dim, kernel_size=31, stride=2):
        super().__init__()
        padding = (kernel_size - 1) // 2
        # 注意: 这里的 in_channels 会根据 rank 动态变化 (r 或 3r)
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
        # x: (Rank, Length) -> Output (Rank, Target_Dim)
        is_transposed = False
        if x.shape[0] > x.shape[1]: 
            x = x.T 
            is_transposed = True
        
        feat = self.pool(self.act(self.conv(x.unsqueeze(0))))
        out = feat.squeeze(0)
        
        if is_transposed: out = out.T
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
        
        gpt2_in = original_layer.weight.shape[0]   # 1600
        gpt2_out = original_layer.weight.shape[1]  # 4800 (attn) or 1600 (proj)

        # 冻结原 GPT2 参数
        self.base_layer.weight.requires_grad = False
        if getattr(self.base_layer, 'bias', None) is not None:
            self.base_layer.bias.requires_grad = False

        # --- 核心逻辑: 分别存储 Llama 参数以实现完美回填 ---
        if layer_type == "attn":
            # 这里的 rank 将变为 3*r (物理上)，但逻辑上是 3 个 r
            # 我们分别注册参数，这样回填时非常简单
            self.lora_A_q = nn.Parameter(params_dict['q'][0].clone().detach())
            self.lora_B_q = nn.Parameter(params_dict['q'][1].clone().detach())
            
            self.lora_A_k = nn.Parameter(params_dict['k'][0].clone().detach())
            self.lora_B_k = nn.Parameter(params_dict['k'][1].clone().detach())
            
            self.lora_A_v = nn.Parameter(params_dict['v'][0].clone().detach())
            self.lora_B_v = nn.Parameter(params_dict['v'][1].clone().detach())
            
            # 定义 Bridge
            # 我们需要 3 对 Bridge，或者 1 对 rank=3r 的 Bridge
            # 为了简单和独立性，我们使用 rank=16 的 Bridge 分别处理
            # A 矩阵: Llama(4096) -> GPT2(1600)
            self.bridge_A_q = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_A_k = LlamaToGPT2Bridge(rank, gpt2_in)
            self.bridge_A_v = LlamaToGPT2Bridge(rank, gpt2_in)
            
            # B 矩阵: Llama(4096) -> GPT2(Attn输出是拼接的, 所以每份占 1/3 即 1600)
            # GPT2 c_attn out is 4800, so each head type gets 1600
            target_out_per_head = gpt2_out // 3
            self.bridge_B_q = LlamaToGPT2Bridge(rank, target_out_per_head)
            self.bridge_B_k = LlamaToGPT2Bridge(rank, target_out_per_head)
            self.bridge_B_v = LlamaToGPT2Bridge(rank, target_out_per_head)

        else: # proj (Output)
            self.lora_A = nn.Parameter(params_dict['o'][0].clone().detach())
            self.lora_B = nn.Parameter(params_dict['o'][1].clone().detach())
            
            self.bridge_A = LlamaToGPT2Bridge(rank, gpt2_in) # 4096 -> 1600
            self.bridge_B = LlamaToGPT2Bridge(rank, gpt2_out) # 4096 -> 1600

    def forward(self, x):
        if self.layer_type == "attn":
            # 1. 分别计算 Q, K, V 的 LoRA 增量
            # x: (B, S, 1600)
            
            # Q path
            Aq = self.bridge_A_q(self.lora_A_q) # (16, 1600)
            Bq = self.bridge_B_q(self.lora_B_q) # (1600, 16)
            delta_Q = (x @ Aq.T @ Bq.T) * self.scaling
            
            # K path
            Ak = self.bridge_A_k(self.lora_A_k)
            Bk = self.bridge_B_k(self.lora_B_k)
            delta_K = (x @ Ak.T @ Bk.T) * self.scaling
            
            # V path
            Av = self.bridge_A_v(self.lora_A_v)
            Bv = self.bridge_B_v(self.lora_B_v)
            delta_V = (x @ Av.T @ Bv.T) * self.scaling
            
            # 2. 拼接结果 (GPT2 格式: [Q, K, V] on last dim)
            lora_term = torch.cat([delta_Q, delta_K, delta_V], dim=-1)
            
        else:
            # 标准投影
            A = self.bridge_A(self.lora_A)
            B = self.bridge_B(self.lora_B)
            lora_term = (x @ A.T @ B.T) * self.scaling
            
        return self.base_layer(x) + lora_term

# ==============================================================================
# 3. 功能函数: 提取与回填
# ==============================================================================
def get_peft_llama(rank=16):
    """初始化一个带 LoRA 的 Llama 模型"""
    try:
        base = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/huggingface_cache/hub/models--NousResearch--Llama-2-7b-chat-hf/snapshots/351844e75ed0bcbbe3f10671b3c808d2b83894ee", device_map="cpu", torch_dtype=torch.float16)
    except:
        from transformers import LlamaConfig
        base = LlamaForCausalLM(LlamaConfig(hidden_size=4096, num_hidden_layers=32))
    
    print(base)
    config = LoraConfig(
        r=rank, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    return get_peft_model(base, config)

def extract_params(peft_model):
    """提取参数并按层分组，保持 Q/K/V 独立"""
    extracted = []
    num_layers = peft_model.config.num_hidden_layers
    sd = peft_model.state_dict()
    
    for i in range(num_layers):
        prefix = f"base_model.model.model.layers.{i}.self_attn"
        
        layer_params = {'attn': {}, 'proj': {}}
        
        # 提取 Q, K, V (独立提取，不合并!)
        for module in ['q', 'k', 'v']:
            key_A = f"{prefix}.{module}_proj.lora_A.default.weight"
            key_B = f"{prefix}.{module}_proj.lora_B.default.weight"
            layer_params['attn'][module] = (sd[key_A], sd[key_B])
            
        # 提取 O
        key_A = f"{prefix}.o_proj.lora_A.default.weight"
        key_B = f"{prefix}.o_proj.lora_B.default.weight"
        layer_params['proj']['o'] = (sd[key_A], sd[key_B])
        
        extracted.append(layer_params)
    return extracted

def inject_and_freeze(gpt2_model, llama_params, rank, device):
    """
    策略:
    - 前 len(llama_params) 层 (32层): 注入 LoRA 参数，设置为 Trainable。
    - 剩余层 (48-32=16层): 不注入，全部 Freeze。
    """
    print(f"\n[Injection] Configuring GPT2 layers (Partial Alignment Strategy)...")
    
    llama_layer_count = len(llama_params)
    gpt2_layer_count = len(gpt2_model.transformer.h)
    
    for i, block in enumerate(gpt2_model.transformer.h):
        # Case 1: 对齐层 (0 - 31)
        if i < llama_layer_count:
            p = llama_params[i]
            
            # 替换 Attn
            block.attn.c_attn = SplitBridgedLoRALayer(
                block.attn.c_attn, p['attn'], rank=rank, layer_type="attn"
            ).to(device)
            
            # 替换 Proj
            block.attn.c_proj = SplitBridgedLoRALayer(
                block.attn.c_proj, p['proj'], rank=rank, layer_type="proj"
            ).to(device)
            
            # 注意: SplitBridgedLoRALayer 内部会自动冻结 base_layer，但 LoRA 参数是 trainable 的
            # 我们还需要冻结 MLP 和 LayerNorm，因为我们只微调 Llama 参数
            for param in block.ln_1.parameters(): param.requires_grad = False
            for param in block.ln_2.parameters(): param.requires_grad = False
            for param in block.mlp.parameters():  param.requires_grad = False
            
        # Case 2: 剩余层 (32 - 47) -> 彻底冻结
        else:
            # 遍历该块的所有参数，设为不可训练
            for param in block.parameters():
                param.requires_grad = False
    
    # 冻结 GPT2 的 Embedding 和 Head (通常我们只关心中间层变换)
    for param in gpt2_model.transformer.wte.parameters(): param.requires_grad = False
    for param in gpt2_model.transformer.wpe.parameters(): param.requires_grad = False
    for param in gpt2_model.transformer.ln_f.parameters(): param.requires_grad = False
    # 如果有 lm_head
    if hasattr(gpt2_model, 'lm_head'):
         for param in gpt2_model.lm_head.parameters(): param.requires_grad = False
            
    print(f"[Injection] Layers 0-{llama_layer_count-1}: Injected & Trainable (LoRA only).")
    print(f"[Injection] Layers {llama_layer_count}-{gpt2_layer_count-1}: Fully Frozen.")

def sync_back(gpt2_model, llama_peft_model):
    """只同步前 32 层"""
    print("\n[Sync] Syncing parameters back to Llama2...")
    llama_sd = llama_peft_model.state_dict()
    num_layers = llama_peft_model.config.num_hidden_layers
    
    with torch.no_grad():
        for i in range(num_layers):
            # 这里的 i 既是 GPT2 的 layer index，也是 Llama 的 layer index
            gpt2_block = gpt2_model.transformer.h[i]
            prefix = f"base_model.model.model.layers.{i}.self_attn"
            
            # 1. Attn (Q, K, V)
            custom_layer = gpt2_block.attn.c_attn
            
            # Q
            llama_sd[f"{prefix}.q_proj.lora_A.default.weight"].copy_(custom_layer.lora_A_q)
            llama_sd[f"{prefix}.q_proj.lora_B.default.weight"].copy_(custom_layer.lora_B_q)
            # K
            llama_sd[f"{prefix}.k_proj.lora_A.default.weight"].copy_(custom_layer.lora_A_k)
            llama_sd[f"{prefix}.k_proj.lora_B.default.weight"].copy_(custom_layer.lora_B_k)
            # V
            llama_sd[f"{prefix}.v_proj.lora_A.default.weight"].copy_(custom_layer.lora_A_v)
            llama_sd[f"{prefix}.v_proj.lora_B.default.weight"].copy_(custom_layer.lora_B_v)
            
            # 2. Proj (O)
            custom_layer_o = gpt2_block.attn.c_proj
            llama_sd[f"{prefix}.o_proj.lora_A.default.weight"].copy_(custom_layer_o.lora_A)
            llama_sd[f"{prefix}.o_proj.lora_B.default.weight"].copy_(custom_layer_o.lora_B)
            
    llama_peft_model.load_state_dict(llama_sd)
    print("[Sync] Sync complete.")

# ==============================================================================
# 2. 自定义数据集 (Dummy Data)
# ==============================================================================
class TextDataset(Dataset):
    def __init__(self, length=100, seq_len=64):
        self.input_ids = torch.randint(0, 50257, (length, seq_len))
        self.labels = self.input_ids.clone()
        
    def __len__(self):
        return len(self.input_ids)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx]
        }

# ==============================================================================
# 3. 自定义 Callback: 实现自动回填 (Sync)
# ==============================================================================
class LlamaSyncCallback(TrainerCallback):
    """
    在训练结束或保存 checkpoint 时，将参数从 GPT2 同步回 Llama
    """
    def __init__(self, llama_peft_model, gpt2_model):
        self.llama_peft_model = llama_peft_model
        self.gpt2_model = gpt2_model
        
    def on_save(self, args, state, control, **kwargs):
        """当 Trainer 保存 Checkpoint 时触发"""
        print(f"\n[Callback] Trainer is saving. Syncing params back to Llama...")
        # 注意: 这里需要引用之前的 sync_back 函数
        # sync_back(self.gpt2_model, self.llama_peft_model) 
        # 为了避免依赖外部函数，我们可以把逻辑写在这里，或者传入函数
        # 这里假设 sync_back 在全局作用域可用
        sync_back(self.gpt2_model, self.llama_peft_model)
        
        # 可选: 保存 Llama 的 LoRA 权重
        save_path = f"{args.output_dir}/llama_lora_checkpoint-{state.global_step}"
        self.llama_peft_model.save_pretrained(save_path)
        print(f"[Callback] Llama LoRA adapters saved to {save_path}")

    def on_train_end(self, args, state, control, **kwargs):
        """训练结束时触发"""
        print(f"\n[Callback] Training finished. Final sync...")
        sync_back(self.gpt2_model, self.llama_peft_model)

# ==============================================================================
# 4. 主程序
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = 16
    gpt_model_path =  "/root/autodl-tmp/huggingface_cache/hub/models--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2"
    
    # ------------------------------------------------------------------
    # Step 1: 准备模型 (和之前一样)
    # ------------------------------------------------------------------
    print("Initializing Llama...")
    # 注意: 如果这里显存不足，可以让 Llama 保持在 CPU，sync_back 时会慢一点但省显存
    llama_model = get_peft_llama(rank) 

    # 记录初始值用于验证 (取 Layer 0 的 Q-Proj A 矩阵)
    initial_val = llama_model.state_dict()["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"][0,0].item()
    print(f"Initial Llama Param Value: {initial_val:.9f}")

    print("Extracting Params...")
    llama_params = extract_params(llama_model)
    
    print("Loading GPT2-XL...")
    try:
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt_model_path)
    except:
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # 注入并冻结
    inject_and_freeze(gpt2_model, llama_params, rank, device)
    
    # 确保模型在 GPU
    gpt2_model.to(device)

    # ------------------------------------------------------------------
    # Step 2: 准备数据
    # ------------------------------------------------------------------
    # 这里使用伪造数据，实际请加载你的 dataset
    train_dataset = TextDataset(length=500, seq_len=128)
    
    # ------------------------------------------------------------------
    # Step 3: 配置 Trainer
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir="./gpt2_llama_bridge_output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        num_train_epochs=1,
        save_steps=50000,         # 每 100 步保存一次
        logging_steps=10,
        remove_unused_columns=False, # 关键: 防止 Trainer 自动删除不认识的列
        report_to="none"
    )

    # ------------------------------------------------------------------
    # Step 4: 初始化 Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=gpt2_model,
        args=training_args,
        train_dataset=train_dataset,
        # 注册我们的自定义 Callback
        callbacks=[LlamaSyncCallback(llama_model, gpt2_model)]
    )
    
    # 打印可训练参数量确认
    trainable_params = sum(p.numel() for p in gpt2_model.parameters() if p.requires_grad)
    print(f"\nStarting Trainer... Trainable Params: {trainable_params}")

    # ------------------------------------------------------------------
    # Step 5: 开始训练
    # ------------------------------------------------------------------
    # Trainer 会自动处理 forward, backward, optimizer, scheduler
    trainer.train()

    # 记录训练后用于验证 (取 Layer 0 的 Q-Proj A 矩阵)
    after_val = llama_model.state_dict()["base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight"][0,0].item()
    print(f"After Llama Param Value: {after_val:.9f}")
    # ------------------------------------------------------------------
    # Step 6: 最终保存
    # ------------------------------------------------------------------
    # 训练结束后，Callback 已经执行了 sync_back。
    # 我们现在只需要保存 Llama 的最终结果。

    # 合并LoRA挂载
    final_model = llama_model.merge_and_unload()
    print("[Merge] Merge complete. Model type is now:", type(final_model))
    
    final_save_path = "./final_llama_lora_output"
    llama_model.save_pretrained(final_save_path)
    print(f"\nDone! Final Llama LoRA parameters saved to {final_save_path}")

if __name__ == "__main__":
    # 请确保你在运行前已经补全了 SplitBridgedLoRALayer 等类的完整实现
    main()