def pre_train_adapter(self, dataset, batch_size, epochs, learn_rate, temp_nce, temp_kl, server_node):
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

                    s_outputs = server_node.model(
                        input_ids=s_enc['input_ids'], 
                        attention_mask=s_enc['attention_mask'], 
                        labels=s_enc['labels']
                    )
                    server_ce_loss = s_outputs.loss.item() # 标量
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
                    client_ce_loss = st_output.loss.item() # 标量

                    raw_client_feat = self.features['out'].detach()
                    # has_nan = torch.isnan(raw_client_feat).any()
                    # print(f"Has NaN: {has_nan.item()}")
                
                # Adapter Forward
                aligned_feat = self.adapter(raw_client_feat, teacher_feat, client_mask=c_enc['attention_mask'])
                aligned_feat = aligned_feat.to(torch.float16)
                # has_nan = torch.isnan(aligned_feat).any()
                # print(f"Has NaN: {has_nan.item()}")
                # Loss Calculation
                student_logits = server_head(aligned_feat)

                # ========== 将 logits 转换为自然语言 ==========
        
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

                loss_kl = F.kl_div(F.log_softmax(v_s_logits / temp_kl, dim=-1), F.softmax(v_t_logits / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl

                # loss_kl = F.kl_div(F.log_softmax(aligned_feat / temp_kl, dim=-1), F.softmax(teacher_feat / temp_kl, dim=-1), reduction='batchmean') * temp_kl * temp_kl
                # loss_kl = loss_kl / aligned_feat.shape[1]
                
                loss_nce = contrastive_loss(aligned_feat, teacher_feat, server_mask, temp_nce)
                
                loss = 0.6 * loss_ce + 0.3 * loss_kl + 0.1 * loss_nce
                
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
                    'ce_loss': f'{(loss_ce.item())*0.6:.4f}',
                    'kl_loss': f'{(loss_kl.item())*0.3:.4f}',
                    'nce_loss': f'{(loss_nce.item())*0.1:.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            batch_pbar.close()
            epoch_avg_loss = epoch_loss / len(dataloader)
            print(f"\nEpoch {ep+1} completed. Average loss = {epoch_avg_loss:.4f}")

        del dataloader
        del optimizer
        gc.collect()