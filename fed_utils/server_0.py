import os
import gc
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torch.nn as nn
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft import get_peft_model
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM
from transformers import Trainer, TrainingArguments


class GeneralServer:
    def __init__(self, model_path, output_dir, peft_config):
        self.base_model =  AutoModelForCausalLM.from_pretrained(
                                                                    model_path,
                                                                    load_in_8bit=False,
                                                                    torch_dtype=torch.float16,
                                                                    device_map="cpu",
                                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"
        self.model = get_peft_model(self.base_model, peft_config)

        self.server_dim = self.base_model.config.hidden_size

        self.output_dir = output_dir
        self.features = {}
        self._register_hook()
        self.local_output_dir = os.path.join(self.output_dir, "server_output")
        self.base_model.gradient_checkpointing_enable() # 开启梯度检查点
        self.base_model.enable_input_require_grads() 

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features['out'] = output
        try:
            layer = self.model.base_model.model.model.norm
        except:
            layer = [m for n, m in self.model.named_modules() if ".norm" in n][-1]
        layer.register_forward_hook(hook_fn)

    def tokenize(self, prompt, add_eos_token=True):
        prompt_with_eos = [t + self.tokenizer.eos_token for t in prompt]
        result = self.tokenizer(
            prompt_with_eos,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors='pt'
        )

        labels = result['input_ids'].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 将处理好的 labels 放回字典
        result['labels'] = labels
        
        return result
    
    def get_features(self, input_ids, attention_mask):
        self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return self.features['out']

    # --- Step 3: Global Aggregation (Public Data) ---
    def global_train(self, dataset, batch_size, epochs, learn_rate, temp_kl, clients):
        print(f"[Server] Step 3: Global Training on Public Data...")

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learn_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_batches = len(dataloader)
        total_steps = len(dataloader) * epochs
        warmup_steps = int(0.1 * total_steps)  # 10%的warmup
        
        cached_server_queries = []
        accumulated_aligned_feats = [None] * num_batches
        lm_head = self.model.base_model.lm_head

        print("  -> Phase 1: Pre-computing Server Queries...")
        self.model.eval()

        for texts in dataloader:
            # 1. Server Query (Detached)
            s_enc = self.tokenize(texts)
            with torch.no_grad():
                query = self.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                cached_server_queries.append(query.cpu())

        print("  -> Phase 2: Pre-computing Client Aligned Features...")
        for client in clients:
            print(f"     Processing {client.client_id}...")
            client.load_state()
            client.to_gpu()
            client.adapter.eval()     

            # 遍历数据集 (顺序与 cached_server_queries 一致)
            for i, texts in enumerate(dataloader):
                query_gpu = cached_server_queries[i].to("cuda")
                
                # Client 计算
                c_enc = client.tokenize(texts)
                with torch.no_grad():
                    client.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'])
                    raw_feat = client.features['out'].detach()
                    # Cross-Attention 对齐
                    aligned = client.adapter(raw_feat, query_gpu)
                
                if aligned.dtype == torch.float32:
                    aligned = aligned.to(torch.float16)
                # 累加到 CPU 缓存中
                aligned_cpu = aligned.cpu()
                
                if accumulated_aligned_feats[i] is None:
                    accumulated_aligned_feats[i] = aligned_cpu
                else:
                    accumulated_aligned_feats[i] += aligned_cpu
            
            # Client卸载
            client.unload()

        # 2. Aggregate
        # 清理不再需要的 Query 缓存
        del cached_server_queries
        gc.collect()

        num_clients = len(clients)
        final_target_feats = [feat / num_clients for feat in accumulated_aligned_feats]

        del accumulated_aligned_feats
        gc.collect()
        

        # 3. Server Training
        print("  -> Phase 3: Server Training (Fast)...")
        self.model.train()
        indices = torch.randperm(num_batches).tolist()
        all_texts_batches = list(dataloader)

        # 线性调度器（带warmup）
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        total_loss = 0
        total_ce_loss = 0  
        total_kl_loss = 0
        batch_count = 0

        for ep in range(epochs):

            # 创建进度条
            pbar = tqdm(
            enumerate(indices), 
            total=len(indices),
            desc="Server Training",
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
            )

            for batch_idx, idx in pbar:
                texts = all_texts_batches[idx]
                agg_feat = final_target_feats[idx].to("cuda") # 移回 GPU
                
                s_enc = self.tokenize(texts)

                if agg_feat.dtype == torch.float32:
                    agg_feat = agg_feat.to(torch.float16)
        
                with torch.no_grad():
                    agg_logits = lm_head(agg_feat)

                outputs = self.model(
                    input_ids=s_enc['input_ids'], 
                    attention_mask=s_enc['attention_mask'], 
                    labels=s_enc['labels']
                )
                
                # 3. Server Update
                kl_loss = F.kl_div(F.log_softmax(outputs.logits / temp_kl, dim=-1), F.softmax(agg_logits / temp_kl, dim=-1), reduction='batchmean')
                
                loss = 0.8 * outputs.loss + 0.2 * kl_loss
                # loss = outputs.loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 进度条设置
                current_loss = loss.item()
                current_ce_loss = outputs.loss.item()
                current_kl_loss = kl_loss.item()
                
                total_loss += current_loss
                total_ce_loss += current_ce_loss
                total_kl_loss += current_kl_loss
                batch_count += 1

                # 更新进度条显示
                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'ce_loss': f'{current_ce_loss:.4f}',
                    'kl_loss': f'{current_kl_loss:.4f}',
                    'avg_loss': f'{total_loss/batch_count:.4f}',
                    'avg_ce_loss': f'{total_ce_loss/batch_count:.4f}',
                    'avg_kl_loss': f'{total_kl_loss/batch_count:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                
                # 每30个batch输出一次详细统计
                if (batch_idx + 1) % 30 == 0:
                    avg_loss = total_loss / batch_count
                    avg_ce_loss = total_ce_loss / batch_count
                    avg_kl_loss = total_kl_loss / batch_count
                    print(f"\nBatch {batch_idx+1}/{len(indices)} - Summary:")
                    print(f"  Average: loss={avg_loss:.4f}")
                    print(f"  Average: ce loss={avg_ce_loss:.4f}")
                    print(f"  Average: kl loss={avg_kl_loss:.4f}")
                    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        del final_target_feats, outputs, optimizer, dataloader
        gc.collect()
        torch.cuda.empty_cache()

    # --- Step 4: Distill to Clients (Public Data) ---
    def distill_to_clients(self, dataset, batch_size, epochs ,learn_rate, temp_kl, clients):
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        lm_head = self.model.base_model.lm_head
        
        for client in clients:
            print(f"[Server] Step 4: Distilling back to Client{client.client_id}")
            client.model.train() # Update LoRA
            # 策略：降低学习率，防止剧烈震荡
            optimizer = torch.optim.AdamW(client.model.parameters(), lr=learn_rate)
            
            for ep in range(epochs):
                # 初始化统计变量
                total_loss = 0
                batch_count = 0
                epoch_loss = 0
                
                batch_pbar = tqdm(dataloader, desc=f"Epoch {ep+1}/{epochs}", leave=False)
                for i, texts in enumerate(batch_pbar):
                    # Target: Server
                    with torch.no_grad():
                        s_enc = self.tokenize(texts)
                        t_feat = self.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                        t_logits = lm_head(t_feat)
                    
                    # Student: Client
                    c_enc = client.tokenize(texts)
                    outputs = client.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'], labels=c_enc['labels'], output_hidden_states=True)
                    raw_feat = client.features['out']
                    loss_ce = outputs.loss 
                    
                    # Adapter Forward (Frozen)
                    aligned_feat = client.adapter(raw_feat, t_feat)

                    if aligned_feat.dtype == torch.float32:
                        aligned_feat = aligned_feat.to(torch.float16)
                    s_logits = lm_head(aligned_feat)
                    
                    loss_kl = F.kl_div(F.log_softmax(s_logits / temp_kl, dim=-1), F.softmax(t_logits / temp_kl, dim=-1), reduction='batchmean')

                    loss = 0.7 * loss_ce + 0.3 * loss_kl
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

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
                        'kl_loss': f'{(loss_kl.item())*0.3:.4f}',
                        'ce_loss': f'{(loss_ce.item())*0.7:.4f}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
                
                batch_pbar.close()
                epoch_avg_loss = epoch_loss / len(dataloader)
                print(f"\nEpoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")
            
        del optimizer, dataloader
        gc.collect()
        torch.cuda.empty_cache()    

    def local_train(self, dataset, epochs, batch_size, learn_rate):
        print("Step 1: Server Local Training on Public Data...")
        self.model.train()

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
            logging_steps=30,
            optim="adamw_torch",
            eval_strategy="no", #按训练步数间隔在验证集上评估模型
            save_strategy="no", #按训练步数间隔来保存模型检查点
            output_dir=self.local_output_dir,
            save_total_limit=0,
            load_best_model_at_end=False, #在训练结束时，是否加载在验证集上性能最好的那个模型作为最终模型
            ddp_find_unused_parameters=False,
            group_by_length=False, #是否在构建批次时，将长度相近的样本分组在一起。
            dataloader_drop_last=False,
            gradient_checkpointing=False, #开启梯度检查点减少显存消耗
            # report_to="none", #不生成runs日志
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
    # --- 显存管理核心方法 ---

    def to_gpu(self):
        """将模型移动到 GPU"""
        print("Server Moving to GPU...")
        self.model.to("cuda")
        torch.cuda.empty_cache()

    def unload(self):
        """将模型移回 CPU 并清理显存"""
        print("Server Unloading to CPU...")
        self.model.to("cpu")
        
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
        
        print(f"Server State saved to {self.local_output_dir}")

    def load_state(self):
        """从磁盘加载参数"""
        lora_path = os.path.join(self.local_output_dir, "lora")
        
        if os.path.exists(lora_path):
            # 加载 LoRA 权重
            self.model.load_adapter(lora_path, adapter_name="default")
            
        print("Server State loaded.")
