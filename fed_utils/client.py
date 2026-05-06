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
from transformers import Trainer, TrainingArguments
import itertools

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

    def forward(self, client_feat, server_query, client_mask=None):
        if server_query.dtype == torch.float16:
            server_query = server_query.to(torch.float32)
        # A. 维度投影
        kv = self.input_proj(client_feat) 
        
        # B. Cross Attention
        q = self.attn_norm(server_query)
        k = v = self.attn_norm(kv)

        # ======== 核心改进：引入 Key Padding Mask ========
        if client_mask is not None:
            # PyTorch 的 key_padding_mask 要求: Padding 位置为 True, 有效位置为 False
            # 你的 mask 中 1 是有效, 0 是 pad，所以需要取反
            key_padding_mask = (client_mask == 0)
        else:
            key_padding_mask = None
        
        attn_out, _ = self.attn(query=q, key=k, value=v, need_weights=False, key_padding_mask=key_padding_mask)
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

def smooth_l1_feature_loss(student_feat, teacher_feat, attention_mask, beta=1.0):
    """
    Smooth L1 Feature Alignment Loss
    精准匹配特征的方向和绝对尺度，同时免疫异常值梯度爆炸。
    """
    # 1. 强制转为 FP32 保证数值稳定
    student_feat = student_feat.to(torch.float32)
    teacher_feat = teacher_feat.to(torch.float32)

    # 2. 展平特征 [B*L, D]
    dim = student_feat.shape[-1]
    s_flat = student_feat.view(-1, dim)
    t_flat = teacher_feat.view(-1, dim)
    mask_flat = attention_mask.view(-1)

    # 3. 过滤 Padding，仅提取真实有效的 Token
    active_indices = torch.nonzero(mask_flat).squeeze()
    
    if active_indices.numel() == 0:
        return torch.tensor(0.0, device=student_feat.device, requires_grad=True)
    
    s_active = s_flat[active_indices]
    t_active = t_flat[active_indices]

    # 4. 计算 Smooth L1 Loss (Huber Loss)
    # 当 |s - t| < beta 时，等价于 0.5 * MSE
    # 当 |s - t| >= beta 时，等价于 L1 - 0.5 * beta
    loss = F.smooth_l1_loss(s_active, t_active, beta=beta, reduction='mean')
    
    return loss

class GeneralClient:
    def __init__(self, client_id, model_path, output_dir, peft_config, server_dim ):
        self.client_id = client_id
        self.base_model =  AutoModelForCausalLM.from_pretrained(
                                                                    model_path,
                                                                    # load_in_8bit=False,
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
        # prompt_with_eos = [t + self.tokenizer.eos_token for t in prompt]

        max_len = 511 
        result = self.tokenizer(prompt, 
                                   truncation=True, 
                                   max_length=max_len, 
                                   padding=False,)
        
        for i in range(len(result["input_ids"])):
            result["input_ids"][i].append(self.tokenizer.eos_token_id)
            result["attention_mask"][i].append(1)

        result = self.tokenizer.pad(result, padding=True, return_tensors='pt').to('cuda')
        # result = self.tokenizer(
        #     prompt_with_eos,
        #     truncation=True,
        #     max_length=512,
        #     padding=True,
        #     return_tensors='pt'
        # ).to('cuda')

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

        # 1. 定义 TrainingArguments
        train_args = TrainingArguments(
            per_device_train_batch_size=batch_size, #在每个GPU/CPU上用于训练的实际批次大小
            gradient_accumulation_steps=4, #梯度累积的步数。
            dataloader_pin_memory=False,
            warmup_steps=0, #不使用学习率预热
            num_train_epochs=epochs,
            learning_rate=learn_rate,
            # lr_scheduler_type="constant", #使用恒定学习率
            lr_scheduler_type="cosine",
            fp16=True, #混合精度训练
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="no", #按训练步数间隔在验证集上评估模型
            save_strategy="no", #按训练步数间隔来保存模型检查点
            output_dir=self.local_output_dir,
            save_total_limit=0,
            load_best_model_at_end=False, #在训练结束时，是否加载在验证集上性能最好的那个模型作为最终模型
            ddp_find_unused_parameters=False,
            # group_by_length=False, #是否在构建批次时，将长度相近的样本分组在一起。
            dataloader_drop_last=False,
            gradient_checkpointing=False, #开启梯度检查点减少显存消耗
            report_to="none", #不生成runs日志
        )

        # 2. 自定义 Data Collator，直接复用你写好的 tokenize 方法
        def custom_data_collator(features):
            # features 是从 dataset 中抽取出的 batch_size 个样本的列表
            # 处理不同格式的 dataset 返回值 (支持 dataset 返回 dict 或直接返回 str)
            if isinstance(features[0], dict) and "text" in features[0]:
                texts = [f["text"] for f in features]
            elif isinstance(features[0], str):
                texts = features
            else:
                texts =[str(f) for f in features]

            # 复用你写好的 tokenize (内部包含了加 eos, padding 和设置 labels)
            # 注意: 你的 tokenize 里有 .to('cuda')，Trainer 本身也会做 device 转移，这不冲突
            batch_enc = self.tokenize(texts)
            
            return {
                "input_ids": batch_enc["input_ids"],
                "attention_mask": batch_enc["attention_mask"],
                "labels": batch_enc["labels"]
            }
        
        # 3. 初始化 Trainer
        trainer = Trainer(
            model=self.model,
            args=train_args,
            train_dataset=dataset,
            data_collator=custom_data_collator,
        )

        trainer.train()

        del trainer
        torch.cuda.empty_cache()
        gc.collect()


    def train_adapter(self, dataset, batch_size, epochs, learn_rate, temp_nce, temp_kl, server_node):
        print(f"[{self.client_id}] Step 2: Training Adapter on Public Data...")
        self.model.eval()   
        self.adapter.train()
        
        optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=learn_rate)
        
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        server_head = server_node.model.base_model.lm_head
        total_steps = len(dataloader) * epochs
        
        # 线性调度器（带warmup）
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for ep in range(epochs):
            # 初始化统计变量
            total_loss = 0
            batch_count = 0
            epoch_loss = 0
            skip_count = 0
            batch_pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs}", leave=False)

            for i, texts in enumerate(batch_pbar):
                # Server (Teacher)
                with torch.no_grad():
                    s_enc = server_node.tokenize(texts)
                    server_mask = s_enc['attention_mask']

                    s_outputs = server_node.model(
                        input_ids=s_enc['input_ids'], 
                        attention_mask=s_enc['attention_mask'], 
                        labels=s_enc['labels']
                    )
                    # server_ce_loss = s_outputs.loss.item() # 标量
                    teacher_logits = s_outputs.logits
                    teacher_feat = server_node.get_features(s_enc['input_ids'], s_enc['attention_mask'])

                    # has_nan = torch.isnan(teacher_feat).any()
                    # print(f"Has NaN: {has_nan.item()}")

                    # teacher_logits = server_head(teacher_feat)
                
                # Client (Student)
                c_enc = self.tokenize(texts)
                c_mask = c_enc['attention_mask']

                with torch.no_grad():
                    st_output = self.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'], labels=c_enc['labels'])
                    # self.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'])
                    # client_ce_loss = st_output.loss.item() # 标量
                    
                    raw_client_feat = self.features['out'].detach()

                # 判断 Client 是否已经比 Server 更好
                # if client_ce_loss < server_ce_loss:
                #     skip_count += 1
                #     # 打印信息（可选，建议每 30 个 batch 打印一次看比例）
                #     # print(f"Skip Distill: Client({client_ce_loss:.4f}) < Server({server_ce_loss:.4f})")
                    
                #     # 策略：跳过本次更新。
                #     # 因为已有预训练基础，Adapter 的映射关系在小范围内是稳定的。
                #     continue 
            
                # Adapter Forward
                aligned_feat = self.adapter(raw_client_feat, teacher_feat, client_mask=c_enc['attention_mask'])

                aligned_feat = aligned_feat.to(torch.float16)

                # Loss Calculation
                student_logits = server_head(aligned_feat)

                # ========== 将 logits 转换为自然语言 ==========
        
                # 方法1: 使用 argmax 获取最可能的 token
                if i == len(dataloader)-2:
                    teacher_token_ids = torch.argmax(teacher_logits, dim=-1)  # [batch_size, seq_len]
                    ori_student_token_ids = torch.argmax(st_output.logits, dim=-1)
                    student_token_ids = torch.argmax(student_logits, dim=-1)  # [batch_size, seq_len]
                    
                    # 解码为文本
                    teacher_texts = []
                    ori_student_texts = []
                    student_texts = []
                    
                    for j in range(len(texts)):  # 遍历批次中的每个样本
                        # 解码教师输出
                        effective_mask_server = server_mask[j]  # [seq_len]，值0表示padding
                        effective_mask_client = c_mask[j] 

                        teacher_tokens = teacher_token_ids[j]
                        # 移除 padding 和特殊 token
                        teacher_tokens = teacher_tokens[effective_mask_server.bool()]  # 只保留有效位置

                        teacher_tokens = teacher_tokens[teacher_tokens != server_node.tokenizer.pad_token_id]
                        teacher_tokens = teacher_tokens[teacher_tokens != server_node.tokenizer.eos_token_id]
                        teacher_text = server_node.tokenizer.decode(teacher_tokens, skip_special_tokens=True)
                        teacher_texts.append(teacher_text)
                        
                        ori_student_tokens = ori_student_token_ids[j]
                        ori_student_tokens = ori_student_tokens[effective_mask_client.bool()]


                        ori_student_tokens = ori_student_tokens[ori_student_tokens != self.tokenizer.pad_token_id]
                        ori_student_tokens = ori_student_tokens[ori_student_tokens != self.tokenizer.eos_token_id]
                        ori_student_text = self.tokenizer.decode(ori_student_tokens, skip_special_tokens=True)
                        ori_student_texts.append(ori_student_text)

                        # 解码学生输出
                        student_tokens = student_token_ids[j]
                        student_tokens = student_tokens[effective_mask_server.bool()]

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
                # 新增计算真实 Label 的 CE Loss
                # 自回归模型需要 Shift: 拿 t 时刻的 logits 去预测 t+1 时刻的 label
                shift_logits = student_logits[..., :-1, :].contiguous()
                shift_labels = s_enc['labels'][..., 1:].contiguous()
                
                # 计算 CE Loss，PyTorch 的 CrossEntropyLoss 会自动忽略 ignore_index=-100 的 Padding 部分
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss_ce = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                # 物理隔离 Padding
                valid_mask = (server_mask == 1)
                v_s_logits = student_logits[valid_mask]
                v_t_logits = teacher_logits[valid_mask]

                # # ================= Top-K 截断蒸馏 =================
                # top_k = 30  # 经验值：通常取 10 到 50 之间
            
                # # 1. 找到 Teacher Logits 的 Top-K 的值和索引
                # t_topk_values, t_topk_indices = torch.topk(v_t_logits, k=top_k, dim=-1)
                
                # # 2. 构造一个全是负无穷 (-inf) 的掩码矩阵
                # filtered_v_t_logits = torch.full_like(v_t_logits, float('-inf'))
                
                # # 3. 把 Top-K 的真实 Logits 填回对应的位置
                # # 其他所有非 Top-K 的位置依然是 -inf，经过 Softmax 后概率绝对为 0
                # filtered_v_t_logits.scatter_(dim=-1, index=t_topk_indices, src=t_topk_values)
                # # ================= Top-K 截断蒸馏 =================

                loss_kl = F.kl_div(F.log_softmax(v_s_logits / temp_kl, dim=-1), F.softmax(v_t_logits / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl

                # loss_kl = F.kl_div(F.log_softmax(aligned_feat / temp_kl, dim=-1), F.softmax(teacher_feat / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl
                # loss_kl = loss_kl / aligned_feat.shape[1]
                
                # loss_nce = contrastive_loss(aligned_feat, teacher_feat, server_mask, temp_nce)
                loss_nce = smooth_l1_feature_loss(aligned_feat, teacher_feat, server_mask, beta=1.0)
                
                loss = 0.5 * loss_ce + 0.2 * loss_kl + 0.3 * loss_nce

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
                    'ce_loss': f'{(loss_ce.item())*0.5:.4f}',
                    'kl_loss': f'{(loss_kl.item())*0.2:.4f}',
                    'nce_loss': f'{(loss_nce.item())*0.3:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")
            print(f"\nSkipped batches: {skip_count}")

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
        for i in range(4):
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
            print(f"[{self.client_id}] lora loaded.")
            self.model.load_adapter(lora_path, adapter_name="default")
            
        if os.path.exists(adapter_path):
            # 加载 Adapter 权重
            # 注意：需先确保 adapter 在 CPU 上，加载完再 to_gpu
            print(f"[{self.client_id}] adapter loaded.")
            self.adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))
        
            
        print(f"[{self.client_id}] State loaded.")
