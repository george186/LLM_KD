import os
import gc
from torch.utils.data import DataLoader, Dataset
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import get_peft_model
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM

class CrossAttentionAdapter(nn.Module):
    """
    标准化 Cross-Attention Adapter
    功能: 将 Client (OPT) 特征对齐到 Server (Llama) 的语义空间和序列长度
    """
    def __init__(self, client_dim, server_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.server_dim = server_dim
        
        # 1. 维度对齐 (Client -> Server)
        self.input_proj = nn.Linear(client_dim, server_dim)
        
        # 2. Cross Attention (Query=Server, Key/Value=Client)
        self.attn_norm = nn.LayerNorm(server_dim)
        self.attn = nn.MultiheadAttention(embed_dim=server_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 3. FFN
        self.ffn_norm = nn.LayerNorm(server_dim)
        self.ffn = nn.Sequential(
            nn.Linear(server_dim, server_dim * 4),
            nn.GELU(),
            nn.Linear(server_dim * 4, server_dim),
            nn.Dropout(dropout)
        )
        self.out_norm = nn.LayerNorm(server_dim)

    def forward(self, client_feat, server_query):
        if server_query.dtype == torch.float16:
            server_query = server_query.to(torch.float32)
        # A. 维度投影
        kv = self.input_proj(client_feat) 
        
        # B. Cross Attention
        q = self.attn_norm(server_query)
        k = v = self.attn_norm(kv)
        
        attn_out, _ = self.attn(query=q, key=k, value=v, need_weights=False)
        x = attn_out + server_query # 引入 Server Query 残差

        # x = self.ffn_norm(x) + q # 引入 Server Query 残差
        
        # C. FFN
        x = x + self.ffn(self.ffn_norm(x))
        # x = x + self.ffn(x)
        return self.out_norm(x)

def contrastive_loss(student_feat, teacher_feat, attention_mask, temperature=0.07):
    """InfoNCE Loss: 确保 Student 特征在空间上靠近 Teacher 特征"""
    batch_size, seq_len, dim = student_feat.shape

    # 1. 展平特征 [B*L, D]
    s_flat = student_feat.view(-1, dim)
    t_flat = teacher_feat.view(-1, dim)

    # 2. 展平 Mask [B*L]
    mask_flat = attention_mask.view(-1)

    # 3. 过滤 Padding (只取有效 Token)
    # active_indices 是所有非 Pad 的索引
    active_indices = torch.nonzero(mask_flat).squeeze()

    if active_indices.numel() == 0:
        return torch.tensor(0.0, device=student_feat.device, requires_grad=True)
    
    s_active = s_flat[active_indices]
    t_active = t_flat[active_indices]

    # 4. 计算 NCE Loss
    # 每一个 s_active[i] 应该与 t_active[i] 相似，与 t_active[j] 不相似
    s_emb = F.normalize(s_active, dim=-1)
    t_emb = F.normalize(t_active, dim=-1)
    logits = torch.matmul(s_emb, t_emb.T) / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)

class GeneralClient:
    def __init__(self, client_id, model_path, output_dir, peft_config, server_dim ):
        self.client_id = client_id
        self.base_model =  AutoModelForCausalLM.from_pretrained(
                                                                    model_path,
                                                                    load_in_8bit=False,
                                                                    torch_dtype=torch.float32,
                                                                    device_map="cpu",
                                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"
        self.model = get_peft_model(self.base_model, peft_config)

        self.client_dim = self.base_model.config.hidden_size
        self.adapter = CrossAttentionAdapter(self.client_dim, server_dim, num_heads=4).to("cpu")

        self.output_dir = output_dir
        self.features = {}
        self._register_hook()
        self.local_output_dir = os.path.join(self.output_dir, "local_output_{}".format(self.client_id))
        self.base_model.gradient_checkpointing_enable() # 开启梯度检查点
        self.base_model.enable_input_require_grads() 

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features['out'] = output
        try:
            layer = self.model.base_model.model.model.decoder.final_layer_norm
        except:
            layer = [m for n, m in self.model.named_modules() if "final_layer_norm" in n][-1]
        layer.register_forward_hook(hook_fn)

    def tokenize(self, prompt, add_eos_token=True):
        prompt_with_eos = [t + self.tokenizer.eos_token for t in prompt]
        result = self.tokenizer(
            prompt_with_eos,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        ).to('cuda')

        labels = result['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 将处理好的 labels 放回字典
        result['labels'] = labels
        
        return result
    
    # --- Step 1: Local Training (Private Data) ---
    def local_train(self, dataset, epochs, batch_size, learn_rate):
        print(f"[{self.client_id}] Step 1: Local Training on Private Data...")
        self.model.train()
        self.adapter.eval()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learn_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(dataloader) * epochs
        warmup_steps = int(0.1 * total_steps)  # 10%的warmup
        
        # 线性调度器（带warmup）
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        for ep in range(epochs):
            # 初始化统计变量
            total_loss = 0
            batch_count = 0
            epoch_loss = 0
            batch_pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs}", leave=False)

            for i, texts in enumerate(batch_pbar):
                enc = self.tokenize(texts)
                outputs = self.model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'], labels=enc['labels'])
                loss = outputs.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 累加loss
                total_loss += loss.item()
                batch_count += 1
                epoch_loss += loss.item()
                
                # 每10个batch输出一次平均loss
                if (i + 1) % 30 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"\n Epoch {ep+1}, Batch {i+1}: Average loss = {avg_loss:.4f}")
                    
                    # 重置统计
                    total_loss = 0
                    batch_count = 0

                # 更新进度条信息
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")

        del dataloader
        del optimizer
        gc.collect()

    # --- Step 2: Adapter Alignment (Public Data) ---
    def train_adapter(self, dataset, batch_size, epochs, learn_rate, temp_nce, temp_kl, server_node):
        print(f"[{self.client_id}] Step 2: Training Adapter on Public Data...")
        self.model.eval()   
        self.adapter.train() 
        
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=learn_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        server_head = server_node.model.base_model.lm_head
        total_steps = len(dataloader) * epochs
        warmup_steps = int(0.05 * total_steps)  # 10%的warmup
        
        # 线性调度器（带warmup）
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        for ep in range(epochs):
            # 初始化统计变量
            total_loss = 0
            batch_count = 0
            epoch_loss = 0
            batch_pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs}", leave=False)

            for i, texts in enumerate(batch_pbar):
                # Server (Teacher)
                with torch.no_grad():
                    s_enc = server_node.tokenize(texts)
                    server_mask = s_enc['attention_mask']
                    teacher_feat = server_node.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                    teacher_logits = server_head(teacher_feat)
                
                # Client (Student)
                c_enc = self.tokenize(texts)
                with torch.no_grad():
                    st_output = self.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'])
                    # self.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'])
                    raw_client_feat = self.features['out'].detach()
                
                # Adapter Forward
                aligned_feat = self.adapter(raw_client_feat, teacher_feat)
                aligned_feat = aligned_feat.to(torch.float16)
                # Loss Calculation
                student_logits = server_head(aligned_feat)

                # ========== 将 logits 转换为自然语言 ==========
        
                # 方法1: 使用 argmax 获取最可能的 token
                if i == 0:
                    teacher_token_ids = torch.argmax(teacher_logits, dim=-1)  # [batch_size, seq_len]
                    ori_student_token_ids = torch.argmax(st_output.logits, dim=-1)
                    student_token_ids = torch.argmax(student_logits, dim=-1)  # [batch_size, seq_len]
                    
                    # 解码为文本
                    teacher_texts = []
                    ori_student_texts = []
                    student_texts = []
                    
                    for j in range(len(texts)):  # 遍历批次中的每个样本
                        # 解码教师输出
                        teacher_tokens = teacher_token_ids[j]
                        # 移除 padding 和特殊 token
                        teacher_tokens = teacher_tokens[teacher_tokens != server_node.tokenizer.pad_token_id]
                        teacher_tokens = teacher_tokens[teacher_tokens != server_node.tokenizer.eos_token_id]
                        teacher_text = server_node.tokenizer.decode(teacher_tokens, skip_special_tokens=True)
                        teacher_texts.append(teacher_text)
                        
                        ori_student_tokens = ori_student_token_ids[j]
                        ori_student_tokens = ori_student_tokens[ori_student_tokens != self.tokenizer.pad_token_id]
                        ori_student_tokens = ori_student_tokens[ori_student_tokens != self.tokenizer.eos_token_id]
                        ori_student_text = self.tokenizer.decode(ori_student_tokens, skip_special_tokens=True)
                        ori_student_texts.append(ori_student_text)

                        # 解码学生输出
                        student_tokens = student_token_ids[j]
                        student_tokens = student_tokens[student_tokens != server_node.tokenizer.pad_token_id]
                        student_tokens = student_tokens[student_tokens != server_node.tokenizer.eos_token_id]
                        student_text = server_node.tokenizer.decode(student_tokens, skip_special_tokens=True)
                        student_texts.append(student_text)
                    
                    # 打印示例（可选）
                    for i in range(4):  # 只打印第一个批次
                        print(f"\n=== 批次 {i} 示例 ===")
                        print(f"原始输入: {texts[i]}")
                        print(f"教师输出: {teacher_texts[i]}")
                        print(f"学生原始输出: {ori_student_texts[i]}")
                        print(f"学生转换后输出: {student_texts[i]}")

                # =============================================

                loss_kl = F.kl_div(F.log_softmax(student_logits / temp_kl, dim=-1), F.softmax(teacher_logits / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl
                loss_kl = loss_kl / student_logits.shape[1]

                # loss_kl = F.kl_div(F.log_softmax(aligned_feat / temp_kl, dim=-1), F.softmax(teacher_feat / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl
                # loss_kl = loss_kl / aligned_feat.shape[1]
                
                loss_nce = contrastive_loss(aligned_feat, teacher_feat, server_mask, temp_nce)
                
                loss = 0.8 * loss_kl + 0.2 * loss_nce
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 累加loss
                total_loss += loss.item()
                batch_count += 1
                epoch_loss += loss.item()
                
                # 每10个batch输出一次平均loss
                if (i + 1) % 30 == 0:
                    avg_loss = total_loss / batch_count
                    print(f"\n Epoch {ep+1}, Batch {i+1}: Average loss = {avg_loss:.4f}")
                    
                    # 重置统计
                    total_loss = 0
                    batch_count = 0

                # 更新进度条信息
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'kl_loss': f'{(loss_kl.item())*0.8:.4f}',
                    'nce_loss': f'{(loss_nce.item())*0.2:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")

        del dataloader
        del optimizer
        gc.collect()


    # --- 显存管理核心方法 ---

    def to_gpu(self):
        """将模型移动到 GPU"""
        print(f"[{self.client_id}] Moving to GPU...")
        self.model.to("cuda")
        self.adapter.to("cuda")
        torch.cuda.empty_cache()

    def unload(self):
        """将模型移回 CPU 并清理显存"""
        print(f"[{self.client_id}] Unloading to CPU...")
        for param in self.model.parameters():
            param.grad = None
        
        for param in self.adapter.parameters():
            param.grad = None
        self.model.to("cpu")
        self.adapter.to("cpu")
        
        # 强制清理
        self.features = {} # 清空缓存的特征
        torch.cuda.empty_cache()
        gc.collect()

    def save_state(self):
        """保存 LoRA 和 Adapter 参数到磁盘"""
        if not os.path.exists(self.local_output_dir):
            os.makedirs(self.local_output_dir)
            
        # 1. 保存 LoRA (PEFT 自带方法)
        self.model.save_pretrained(os.path.join(self.local_output_dir, "lora"))
        
        # 2. 保存 Adapter
        torch.save(self.adapter.state_dict(), os.path.join(self.local_output_dir, "adapter.bin"))
        print(f"[{self.client_id}] State saved to {self.local_output_dir}")

    def pre_save_state(self):
        """预训练保存 Adapter 参数到磁盘"""
        for i in range(10):
            pre_dir = os.path.join(self.output_dir, "local_output_{}".format(i))
            if not os.path.exists(pre_dir):
                os.makedirs(pre_dir)
            
            # 保存 Adapter
            torch.save(self.adapter.state_dict(), os.path.join(pre_dir, "adapter.bin"))
            print(f"[{i}] Adapter saved to {pre_dir}")

    def load_state(self):
        """从磁盘加载参数"""
        lora_path = os.path.join(self.local_output_dir, "lora")
        adapter_path = os.path.join(self.local_output_dir, "adapter.bin")
        
        if os.path.exists(lora_path):
            # 加载 LoRA 权重
            self.model.load_adapter(lora_path, adapter_name="default")
            
        if os.path.exists(adapter_path):
            # 加载 Adapter 权重
            # 注意：需先确保 adapter 在 CPU 上，加载完再 to_gpu
            self.adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
            
        print(f"[{self.client_id}] State loaded.")
