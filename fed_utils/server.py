import os
import gc
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
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
from fed_utils.adapter import CrossAttentionAdapter, ServerCrossAttentionAdapter

# ==========================================
# 辅助类 1: 全局 Server 蒸馏的自定义 Trainer
# ==========================================
class GlobalDistillTrainer(Trainer):
    def __init__(self, *args, temp_kl=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temp_kl = temp_kl

        self._custom_metrics = {
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "distill_steps": 0, # 记录经过了多少次 KL 散度计算
            "total_steps": 0    # 记录总共经过了多少次 forward
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        # 取出我们提前计算好的多客户端平均特征
        agg_feat = inputs.pop("agg_feat")
        labels = inputs.pop("labels")
        use_distill = inputs.pop("use_distill")
        attention_mask = inputs.get("attention_mask")

        # 1. Server 前向传播获取基础 CE loss
        outputs = model(**inputs, labels=labels)
        ce_loss = outputs.loss

        # 2. 判断是否需要蒸馏
        if use_distill.item() == 1:
            lm_head = model.base_model.lm_head if hasattr(model, "base_model") else model.lm_head

            with torch.no_grad():
                agg_feat = agg_feat.to(model.device)
                if agg_feat.dtype == torch.float32:
                    agg_feat = agg_feat.to(torch.float16)
                agg_logits = lm_head(agg_feat)

            # 3. 计算 KL 散度
            valid_mask = (attention_mask == 1)
            v_s_logits = outputs.logits[valid_mask]
            v_t_logits = agg_logits[valid_mask]

            temp_kl = self.temp_kl
            kl_loss = F.kl_div(
                F.log_softmax(v_s_logits / temp_kl, dim=-1),
                F.softmax(v_t_logits / temp_kl, dim=-1),
                reduction='batchmean'
            )* (temp_kl ** 2)

            # 4. 融合 Loss
            loss = 0.8 * outputs.loss + 0.2 * kl_loss

            self._custom_metrics["ce_loss"] = self._custom_metrics["ce_loss"] + outputs.loss.item() * 0.8
            self._custom_metrics["kl_loss"] = self._custom_metrics["kl_loss"] + kl_loss.item() * 0.2
            self._custom_metrics["distill_steps"] += 1
        else:
            # 4. 仅使用基础 Loss (自训练)
            loss = ce_loss

            self._custom_metrics["ce_loss"] += outputs.loss.item()

        self._custom_metrics["total_steps"] += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):
        total = self._custom_metrics["total_steps"]

        if total > 0:
            # 计算这段时间内的平均 CE Loss
            avg_ce = self._custom_metrics["ce_loss"] / total
            # 注入到官方 logs 字典中
            logs["ce_loss"] = round(avg_ce, 4)

            # 只有在这段时间内确实发生了蒸馏，才计算和打印 KL Loss
            if self._custom_metrics["distill_steps"] > 0:
                avg_kl = self._custom_metrics["kl_loss"] / self._custom_metrics["distill_steps"]
                logs["kl_loss"] = round(avg_kl, 4)

            # 重置累加器，准备下一个 logging_steps 周期
            self._custom_metrics = {
                "ce_loss": 0.0,
                "kl_loss": 0.0,
                "distill_steps": 0,
                "total_steps": 0
            }

        # 调用父类的 log 方法，将组合好的 logs 真正打印到终端和进度条
        super().log(logs, *args, **kwargs)

# ==========================================
# 辅助类 2: 向 Client 蒸馏的自定义 Trainer
# ==========================================
class ClientDistillTrainer(Trainer):
    def __init__(self, *args, server_node=None, client_node=None, temp_kl=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_node = server_node
        self.client_node = client_node
        self.temp_kl = temp_kl
        self.distill_count = 0      # Server 教 Client 的次数
        self.self_train_count = 0   # Client 自己学的次数

        self._custom_metrics = {
            "ce_loss": 0.0,
            "kl_loss": 0.0,
            "distill_steps": 0, # 记录经过了多少次 KL 散度计算
            "total_steps": 0    # 记录总共经过了多少次 forward
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch = None):
        server_device = self.server_node.model.device

        # 1. 弹出缓存的目标特征和 Loss
        target_feat = inputs.pop("target_feat").to(server_device)
        target_loss = inputs.pop("target_loss").item() # 取出标量 Loss
        labels = inputs.pop("labels")

        # 2. Student: Client Pass
        outputs = model(**inputs, labels=labels, output_hidden_states=True)
        client_ce_loss = outputs.loss

        # 3. 核心逻辑判断：只有全局最优 Loss 比 Client 自己强，才进行蒸馏
        if target_loss < client_ce_loss.item():
            self.distill_count += 1

            client_lm_head = model.base_model.lm_head if hasattr(model, "base_model") else model.lm_head
            with torch.no_grad(): # 必须冻结，作为固定的软标签
                t_logits = client_lm_head(target_feat.to(client_lm_head.weight.dtype)).to(torch.float32)

            s_logits = outputs.logits

            # 物理隔离 Padding
            valid_mask = (inputs.get("attention_mask") == 1)
            v_s_logits = s_logits[valid_mask]
            v_t_logits = t_logits[valid_mask]

            # 计算 KL 散度
            temp_kl = self.temp_kl
            loss_kl = F.kl_div(
                F.log_softmax(v_s_logits / temp_kl, dim=-1),
                F.softmax(v_t_logits / temp_kl, dim=-1),
                reduction='batchmean'
            )* (temp_kl ** 2)

            loss = 0.8 * client_ce_loss + 0.2 * loss_kl

            self._custom_metrics["ce_loss"] += client_ce_loss.item() * 0.7
            self._custom_metrics["kl_loss"] += loss_kl.item() * 0.3
            self._custom_metrics["distill_steps"] += 1
        else:
            # 跳过蒸馏，仅使用本地 Loss 训练
            self.self_train_count += 1
            loss = client_ce_loss

            self._custom_metrics["ce_loss"] += client_ce_loss.item()

        self._custom_metrics["total_steps"] += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, *args, **kwargs):
        total = self._custom_metrics["total_steps"]

        if total > 0:
            # 计算这段时间内的平均 CE Loss
            avg_ce = self._custom_metrics["ce_loss"] / total
            # 注入到官方 logs 字典中
            logs["ce_loss"] = round(avg_ce, 4)

            # 只有在这段时间内确实发生了蒸馏，才计算和打印 KL Loss
            if self._custom_metrics["distill_steps"] > 0:
                avg_kl = self._custom_metrics["kl_loss"] / self._custom_metrics["distill_steps"]
                logs["kl_loss"] = round(avg_kl, 4)

            # 重置累加器，准备下一个 logging_steps 周期
            self._custom_metrics = {
                "ce_loss": 0.0,
                "kl_loss": 0.0,
                "distill_steps": 0,
                "total_steps": 0
            }

        # 调用父类的 log 方法，将组合好的 logs 真正打印到终端和进度条
        super().log(logs, *args, **kwargs)


def contrastive_loss(student_feat, teacher_feat, attention_mask, temperature=0.07):
    """InfoNCE Loss: 确保 Student 特征在空间上靠近 Teacher 特征"""
    batch_size, seq_len, dim = student_feat.shape

    # 1. 展平特征 [B*L, D]
    s_flat = student_feat.view(-1, dim)
    t_flat = teacher_feat.view(-1, dim)

    # 2. 展平 Mask [B*L]
    mask_flat = attention_mask.view(-1)

    # 3. 过滤 Padding (只取有效 Token)
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

class GeneralServer:
    def __init__(self, model_path, output_dir, peft_config, opt_model_path, gpt2_model_path):
        self.base_model =  AutoModelForCausalLM.from_pretrained(
                                                                    model_path,
                                                                    torch_dtype=torch.float16,
                                                                    device_map="cpu",
                                                                )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token_id = (0)
        self.tokenizer.padding_side = "left"
        # self.tokenizer.padding_side = "right"
        self.peft_config = peft_config
        self.model = get_peft_model(self.base_model, peft_config)

        self.server_dim = self.base_model.config.hidden_size

        # ========== 分别加载 OPT 和 GPT2 的 lm_head ==========
        # 加载 OPT 模型的 lm_head
        temp_opt = AutoModelForCausalLM.from_pretrained(opt_model_path)
        opt_dim = temp_opt.config.hidden_size
        self.client_lm_heads = {
            "opt": temp_opt.lm_head.to("cuda"),
        }
        for param in self.client_lm_heads["opt"].parameters():
            param.requires_grad = False
        del temp_opt

        # 加载 GPT2 模型的 lm_head
        temp_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_path)
        gpt2_dim = temp_gpt2.config.hidden_size
        self.client_lm_heads["gpt2"] = temp_gpt2.lm_head.to("cuda")
        for param in self.client_lm_heads["gpt2"].parameters():
            param.requires_grad = False
        del temp_gpt2

        # ========== 分别创建两种类型的 Down-Adapter ==========
        self.down_adapters = {
            "opt": ServerCrossAttentionAdapter(self.server_dim, opt_dim, num_heads=4).to("cuda"),
            "gpt2": ServerCrossAttentionAdapter(self.server_dim, gpt2_dim, num_heads=4).to("cuda"),
        }

        # 打印参数量
        for mtype, adapter in self.down_adapters.items():
            total_params = sum(p.numel() for p in adapter.parameters())
            trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
            print(f"[{mtype}] Down-Adapter 总参数量: {total_params:,} ({total_params/1e6:.2f}M), 可训练: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

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
        max_len = 511
        result = self.tokenizer(prompt,
                                   truncation=True,
                                   max_length=max_len,
                                   padding=False,)

        for i in range(len(result["input_ids"])):
            result["input_ids"][i].append(self.tokenizer.eos_token_id)
            result["attention_mask"][i].append(1)

        result = self.tokenizer.pad(result, padding=True, return_tensors='pt').to('cuda')


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

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        num_batches = len(dataloader)

        # 用于 server 训练的整体最优 (跨类型比较)
        server_losses = []
        cached_server_queries = []

        # ===== 按类型分组追踪每个 batch 的最优客户端 =====
        # 收集所有出现的 client 类型
        client_types = list(set(c.model_type for c in clients))

        self.best_client_losses = {t: [float('inf')] * num_batches for t in client_types}
        # best_client_feats = {t: [None] * num_batches for t in client_types}    # aligned features (server space)
        self.best_client_raw_feats = {t: [None] * num_batches for t in client_types}  # raw features (client space)
        # 记录每个 batch 全局最优 (跨类型) 的 aligned feat 和对应类型
        global_best_aligned_feats = [None] * num_batches
        global_best_losses = [float('inf')] * num_batches
        global_best_types = [None] * num_batches

        print("  -> Phase 1: Pre-computing Server Queries...")
        self.model.eval()

        for texts in tqdm(dataloader):
            # 1. Server Query (Detached)
            s_enc = self.tokenize(texts)
            with torch.no_grad():
                # 记录 Server 的 Loss (作为门槛)
                outputs = self.model(
                    input_ids=s_enc['input_ids'],
                    attention_mask=s_enc['attention_mask'],
                    labels=s_enc['labels']
                )
                server_losses.append(outputs.loss.item())

                query = self.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                cached_server_queries.append(query.cpu())

        print("  -> Phase 2: Compute Best Client per Type...")
        # 获取 Server 的 LM Head 用于测试对齐后的特征
        lm_head = self.model.base_model.lm_head if hasattr(self.model, "base_model") else self.model.lm_head
        # 定义 CE Loss 计算器 (忽略 Padding)
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        for client in clients:
            print(f"     Processing {client.client_id} ({client.model_type})...")
            client.load_state()
            client.to_gpu()
            client.model.eval()
            client.adapter.eval()

            # 遍历数据集 (顺序与 cached_server_queries 一致)
            for i, texts in enumerate(tqdm(dataloader)):
                query_gpu = cached_server_queries[i].to("cuda")

                # Client 计算
                c_enc = client.tokenize(texts)
                s_enc = self.tokenize(texts)

                with torch.no_grad():
                    # 获取 Client 原始特征
                    client.model(input_ids=c_enc['input_ids'], attention_mask=c_enc['attention_mask'])
                    raw_feat = client.features['out'].detach()

                    # Adapter 特征翻译
                    aligned_feat = client.adapter(raw_feat, query_gpu, c_enc['attention_mask'])
                    translated_logits = lm_head(aligned_feat.to(lm_head.weight.dtype)).to(torch.float32)

                    # 自回归模型需要错位 (Shift)
                    shift_logits = translated_logits[..., :-1, :].contiguous()
                    shift_labels = s_enc['labels'][..., 1:].contiguous().to(translated_logits.device)

                    # 计算这个"翻译过来的特征"在 Server 眼里的真实 Loss
                    translated_ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).item()

                    mtype = client.model_type
                    # 按类型更新最优
                    if translated_ce_loss < self.best_client_losses[mtype][i]:
                        self.best_client_losses[mtype][i] = translated_ce_loss

                        self.best_client_raw_feats[mtype][i] = raw_feat.cpu()

                        # if aligned_feat.dtype == torch.float32:
                        #     aligned_feat = aligned_feat.to(torch.float16)
                        # best_client_feats[mtype][i] = aligned_feat.cpu()

                    # 跨类型全局最优
                    if translated_ce_loss < global_best_losses[i]:
                        global_best_losses[i] = translated_ce_loss
                        global_best_types[i] = mtype
                        if aligned_feat.dtype == torch.float32:
                            aligned_feat = aligned_feat.to(torch.float16)
                        global_best_aligned_feats[i] = aligned_feat.cpu()

            # Client卸载
            client.unload()

        # 2. Aggregate
        # 清理不再需要的 Query 缓存
        del cached_server_queries
        gc.collect()

        print("  -> Phase 3: Filtering Batches for Distillation...")
        precomputed_batches = []
        distill_count = 0
        self_train_count = 0

        for i, texts in enumerate(tqdm(dataloader)):
            s_enc = self.tokenize(texts)
            # 使用全局最优 (跨类型) 判断是否蒸馏
            if global_best_losses[i] < server_losses[i]:
                precomputed_batches.append({
                    "input_ids": s_enc["input_ids"],
                    "attention_mask": s_enc["attention_mask"],
                    "labels": s_enc["labels"],
                    "agg_feat": global_best_aligned_feats[i],
                    "use_distill": torch.tensor(1)
                })
                distill_count += 1
            else:
                precomputed_batches.append({
                    "input_ids": s_enc["input_ids"],
                    "attention_mask": s_enc["attention_mask"],
                    "labels": s_enc["labels"],
                    "agg_feat": torch.zeros(1),
                    "use_distill": torch.tensor(0)
                })
                self_train_count += 1

        print(f"    [Status] Total Batches: {num_batches}, Distill: {distill_count}, Self-Train: {self_train_count}")

        if len(precomputed_batches) == 0:
            print("     [Warning] No clients outperformed the server on this public dataset. Skipping training.")
            return

        # 4. Server Training
        print("  -> Phase 4: Server Training (Fast)...")
        self.model.train()

        class PrecomputedBatchDataset(Dataset):
            def __len__(self): return len(precomputed_batches)
            def __getitem__(self, idx): return precomputed_batches[idx]

        train_dataset = PrecomputedBatchDataset()
        def batch_identity_collator(features):
            batch = features[0]
            return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        training_args = TrainingArguments(
            per_device_train_batch_size=1,
            remove_unused_columns=False,
            gradient_accumulation_steps=4,
            dataloader_pin_memory=False,
            warmup_steps=0,
            num_train_epochs=epochs,
            learning_rate=learn_rate,
            lr_scheduler_type="cosine",
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            output_dir=self.local_output_dir,
            save_total_limit=0,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            dataloader_drop_last=False,
            gradient_checkpointing=False,
        )

        trainer = GlobalDistillTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=batch_identity_collator,
            temp_kl=temp_kl
        )

        trainer.train()

        # ================= Phase 5: 构建全局最优专家池 =================
        print("  -> Phase 5: Re-evaluating Server & Constructing Global Best Pool...")
        self.model.eval()

        # 记录 Server 训练后的全新状态
        self.new_server_losses = {t: [] for t in client_types}
        self.new_server_feats =[]

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for texts in tqdm(dataloader):
            s_enc = self.tokenize(texts)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=s_enc['input_ids'],
                    attention_mask=s_enc['attention_mask'],
                    labels=s_enc['labels']
                )

                feat = self.get_features(s_enc['input_ids'], s_enc['attention_mask'])
                self.new_server_feats.append(feat.cpu())

        del global_best_aligned_feats, precomputed_batches, trainer, server_losses
        gc.collect()
        torch.cuda.empty_cache()

        # 分别为每种类型训练 Down-Adapter
        for mtype in client_types:
            # 找到该类型的第一个 client 作为参考 (用于 tokenize 和维度)
            type_clients = [c for c in clients if c.model_type == mtype]
            if len(type_clients) == 0:
                continue
            self.train_down_adapter(dataset, batch_size, 1, 1e-4, temp_kl, type_clients[0], temp_nce=0.07, model_type=mtype)

        # ================= 重新计算每种类型的 Server 翻译后特征和 Loss =================
        self.translated_server_feats = {t: [None] * num_batches for t in client_types}
        for mtype in client_types:
            type_clients = [c for c in clients if c.model_type == mtype]
            if len(type_clients) == 0:
                continue

            self.down_adapters[mtype].eval()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            client_lm_head = self.client_lm_heads[mtype]

            for i, texts in enumerate(tqdm(dataloader, desc=f"Evaluating & Routing ({mtype})")):
                c_enc = type_clients[0].tokenize(texts)
                s_enc = self.tokenize(texts)
                labels = c_enc['labels'].to("cuda")

                # 只有当该类型在该 batch 有 best client 数据时才计算
                if self.best_client_raw_feats[mtype][i] is None:
                    # 没有该类型的 client 数据，跳过 (后面 distill 会处理)
                    self.new_server_losses[mtype].append(float('inf'))
                    continue

                with torch.no_grad():
                    new_s_feat = self.new_server_feats[i].to("cuda").to(torch.float32)
                    q_feat = self.best_client_raw_feats[mtype][i].to("cuda").to(torch.float32)

                translated_s_feat = self.down_adapters[mtype](
                        new_s_feat,
                        q_feat,
                        s_enc['attention_mask'].to("cuda")
                    )

                # 缓存翻译后的特征，供 distill_to_clients 直接使用，避免重复计算
                self.translated_server_feats[mtype][i] = translated_s_feat.detach().cpu()

                s_logits = client_lm_head(translated_s_feat.to(client_lm_head.weight.dtype)).to(torch.float32)
                shift_s_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                translated_loss = loss_fct(shift_s_logits.view(-1, shift_s_logits.size(-1)), shift_labels.view(-1)).item()
                self.new_server_losses[mtype].append(translated_loss)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()



    # --- Step 4: Distill to Clients (Public Data) ---
    def distill_to_clients(self, dataset, batch_size, epochs ,learn_rate, temp_kl, clients):
        self.model.eval()
        for mtype in self.down_adapters:
            self.down_adapters[mtype].eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for client in clients:
            print(f"[Server] Step 4: Distilling back to Client{client.client_id} ({client.model_type})")
            client.model.train() # Update LoRA
            client.adapter.eval()

            mtype = client.model_type

            # ================= 高速缓存打包 =================
            precomputed_batches =[]
            for i, texts in enumerate(tqdm(dataloader)):
                c_enc = client.tokenize(texts)
                s_enc = self.tokenize(texts)

                # 判断该 batch 是否有同类型最优 client
                has_same_type_best = (self.best_client_raw_feats[mtype][i] is not None)

                if not has_same_type_best:
                    # 没有同类型最优，只能用 server 翻译
                    server_choose = True
                elif self.new_server_losses[mtype][i] < self.best_client_losses[mtype][i]:
                    # 有同类型最优，但 server 翻译后更好
                    server_choose = True
                else:
                    # 同类型最优 client 更好
                    server_choose = False

                if server_choose:
                    # 选择 Server，直接复用 global_train 中已缓存翻译特征，避免重复计算
                    target_feat = self.translated_server_feats[mtype][i].clone()
                    target_loss = self.new_server_losses[mtype][i]

                else:
                    # 同类型最优 Client，直接拷贝同类模型的纯血原生特征
                    target_feat = self.best_client_raw_feats[mtype][i].clone()
                    target_loss = self.best_client_losses[mtype][i]

                precomputed_batches.append({
                    "input_ids": c_enc["input_ids"],
                    "attention_mask": c_enc["attention_mask"],
                    "labels": c_enc["labels"],
                    "target_feat": target_feat,
                    "target_loss": torch.tensor(target_loss),
                })

            class PrecomputedBatchDataset(Dataset):
                def __len__(self): return len(precomputed_batches)
                def __getitem__(self, idx): return precomputed_batches[idx]

            def batch_identity_collator(features):
                batch = features[0]
                return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            training_args = TrainingArguments(
                per_device_train_batch_size=1,
                remove_unused_columns=False,
                gradient_accumulation_steps=4,
                num_train_epochs=epochs,
                learning_rate=learn_rate,
                lr_scheduler_type="cosine",
                output_dir=client.local_output_dir,
                fp16=True,
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                dataloader_pin_memory=False,
            )

            trainer = ClientDistillTrainer(
                model=client.model,
                args=training_args,
                train_dataset=PrecomputedBatchDataset(),
                data_collator=batch_identity_collator,
                server_node=self,
                client_node=client,
                temp_kl=temp_kl
            )
            trainer.train()

            total_steps = trainer.distill_count + trainer.self_train_count
            if total_steps > 0:
                distill_ratio = (trainer.distill_count / total_steps) * 100
                self_ratio = (trainer.self_train_count / total_steps) * 100
                print(f"\n[{client.client_id} ({mtype})] Distillation Statistics:")
                print(f"  -> Total Forward Batches: {total_steps}")
                print(f"  -> Server Taught Client : {trainer.distill_count} times ({distill_ratio:.2f}%)")
                print(f"  -> Client Self-Trained  : {trainer.self_train_count} times ({self_ratio:.2f}%)\n")

            # 训练完成后释放显存
            del trainer, precomputed_batches
            client.unload()
            gc.collect()
            torch.cuda.empty_cache()

    def local_train(self, dataset, epochs, batch_size, learn_rate):
        print("Step 1: Server Local Training on Public Data...")
        self.model.train()
        for adapter in self.down_adapters.values():
            adapter.eval()

        # 1. 定义 TrainingArguments
        train_args = TrainingArguments(
            per_device_train_batch_size=batch_size, #在每个GPU/CPU上用于训练的实际批次大小
            gradient_accumulation_steps=4, #梯度累积的步数。
            dataloader_pin_memory=False,
            warmup_steps=0, #不使用学习率预热
            num_train_epochs=epochs,
            learning_rate=learn_rate,
            lr_scheduler_type="cosine",
            fp16=True, #混合精度训练
            logging_steps=30,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            output_dir=self.local_output_dir,
            save_total_limit=0,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
            dataloader_drop_last=False,
            gradient_checkpointing=False,
            report_to="none",
        )

        # 2. 自定义 Data Collator
        def custom_data_collator(features):
            if isinstance(features[0], dict) and "text" in features[0]:
                texts = [f["text"] for f in features]
            elif isinstance(features[0], str):
                texts = features
            else:
                texts =[str(f) for f in features]

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


    def train_down_adapter(self, dataset, batch_size, epochs, learn_rate, temp_kl, client, temp_nce, model_type):
        print(f" Step 3.5: Training Down-Adapter for {model_type}...")
        self.model.eval()
        down_adapter = self.down_adapters[model_type]
        client_lm_head = self.client_lm_heads[model_type]
        down_adapter.train()

        optimizer = torch.optim.AdamW(down_adapter.parameters(), lr=learn_rate)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


        for ep in range(epochs):
            # 初始化统计变量
            total_loss = 0
            batch_count = 0
            epoch_loss = 0
            skip_count = 0
            batch_pbar = tqdm(dataloader, desc=f"[{model_type}] Epoch {ep+1}/{epochs}", leave=False)

            for i, texts in enumerate(batch_pbar):
                if self.best_client_raw_feats[model_type][i] is None:
                    continue

                s_enc = self.tokenize(texts)
                c_enc = client.tokenize(texts)

                server_feat = self.new_server_feats[i].to("cuda").to(torch.float32)
                target_raw_client_feat = self.best_client_raw_feats[model_type][i].to("cuda").to(torch.float32)

                # Down-Adapter 向下翻译：KV=Server, Q=Client
                translated_client_feat = down_adapter(
                    server_feat,
                    target_raw_client_feat,
                    s_enc['attention_mask'].to("cuda")
                )

                s_logits = client_lm_head(translated_client_feat.to(client_lm_head.weight.dtype)).to(torch.float32)

                with torch.no_grad():
                    t_logits = client_lm_head(target_raw_client_feat.to(client_lm_head.weight.dtype)).to(torch.float32)

                # 物理隔离 Padding
                valid_mask = (c_enc['attention_mask'] == 1)
                v_s_logits = s_logits[valid_mask]
                v_t_logits = t_logits[valid_mask]

                loss_kl = F.kl_div(F.log_softmax(v_s_logits / temp_kl, dim=-1), F.softmax(v_t_logits / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl

                loss_nce = smooth_l1_feature_loss(translated_client_feat, target_raw_client_feat, c_enc['attention_mask'], beta=1.0)

                shift_logits = s_logits[..., :-1, :].contiguous()
                shift_labels = c_enc['labels'][..., 1:].contiguous()
                loss_ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

                loss = 0.5 * loss_ce + 0.2 * loss_kl + 0.3 * loss_nce

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
                    print(f"\n [{model_type}] Epoch {ep+1}, Batch {i+1}: Average loss = {avg_loss:.4f}")

                    # 重置统计
                    total_loss = 0
                    batch_count = 0

                # 更新进度条信息
                batch_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce_loss': f'{(loss_ce.item())*0.5:.4f}',
                    'kl_loss': f'{(loss_kl.item())*0.2:.4f}',
                    'nce_loss': f'{(loss_nce.item())*0.3:.4f}',
                })

            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\n[{model_type}] Epoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")
            print(f"\nSkipped batches: {skip_count}")

        del dataloader
        del optimizer
        gc.collect()
    # --- 显存管理核心方法 ---

    def to_gpu(self):
        """将模型移动到 GPU"""
        print("Server Moving to GPU...")
        self.model.to("cuda")
        for adapter in self.down_adapters.values():
            adapter.to("cuda")
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

        # 为每种类型保存 Down-Adapter
        for mtype, adapter in self.down_adapters.items():
            save_path = os.path.join(self.local_output_dir, f"adapter_{mtype}.bin")
            torch.save(adapter.state_dict(), save_path)
            print(f"Server [{mtype}] Adapter saved to {save_path}")

    def load_state(self):
        """从磁盘加载参数"""

        for mtype, adapter in self.down_adapters.items():
            adapter_path = os.path.join(self.local_output_dir, f"adapter_{mtype}.bin")
            if os.path.exists(adapter_path):
                print(f"Server [{mtype}] adapter loaded.")
                adapter.load_state_dict(torch.load(adapter_path, map_location="cpu"))

        print("Server State loaded.")

    def merge_model(self):
        """
        完成当前轮训练后调用：
        1. 合并 LoRA 到基础模型
        2. 保存合并后的模型
        3. 用合并后的权重替换 self.base_model
        4. 重新包装一个新的 PeftModel
        """
        print("Server Merging LoRA and resetting adapter...")

        self.base_model = self.model.merge_and_unload().to("cpu")
        self.model = get_peft_model(self.base_model, self.peft_config)
        self.model.to("cuda")

        # 重新注册 hook
        self._register_hook()

        # 清理旧的特征缓存
        self.features = {}

        print("Server Ready for next train.")

    def merge_save(self):
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(
                                    '/root/autodl-tmp/final',
                                    load_in_8bit=False,
                                    torch_dtype=torch.float16)
        print('finish save')
        
        # 可选：显式删除引用以释放显存
        del merged_model
