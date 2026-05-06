from torch.utils.data import DataLoader, Dataset
import json
import os

class DollyDataset(Dataset):
    """
    专门用于处理 Dolly 格式的 JSON 数据集
    格式: [{"instruction": "...", "context": "...", "response": "..."}]
    """
    def __init__(self, json_path):
        self.data = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            print(f"Warning: File {json_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.format_prompt(item)

    def format_prompt(self, item):
        """
        将 JSON 条目转换为模型输入的 Prompt 字符串。
        采用标准的 Instruction Tuning 格式。
        """
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        response = item.get("response", "")

        if context and len(context.strip()) > 0:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
            
        return prompt

class AlpacaDataset(Dataset):
    """
    专门用于处理 Alpaca 格式的 JSON 数据集
    格式: [{"instruction": "...", "input": "...", "output": "..."}]
    """
    def __init__(self, json_path):
        self.data = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            print(f"Warning: File {json_path} not found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.format_prompt(item)

    def format_prompt(self, item):
        """
        将 JSON 条目转换为模型输入的 Prompt 字符串。
        采用标准的 Alpaca 格式。
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")  # 注意：Alpaca使用"input"而非"context"
        output = item.get("output", "")    # 注意：Alpaca使用"output"而非"response"

        if input_text and len(input_text.strip()) > 0:
            prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
            
        return prompt
