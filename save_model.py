import os
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    AdaLoraConfig,
    AdaLoraModel,
)




llama_model = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/huggingface_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9", device_map="cpu", torch_dtype=torch.float16)
lora_path = './Heter-gpt2-dolly-homo/4/server_output/lora'
model_path = '/root/autodl-tmp'
llama_model = PeftModel.from_pretrained(llama_model, lora_path) # 使用堆叠矩阵的LoRA配置
llama_model = llama_model.merge_and_unload()
llama_model.save_pretrained(model_path + '/final',
                    load_in_8bit=False,
                    torch_dtype=torch.float16)
print('finish save')

# llama_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/huggingface_cache/hub/opt-1.3b/snapshots/1.3b", torch_dtype=torch.float32, device_map="cpu")
# lora_path = './Heter-gpt2-dolly-homo/4/local_output_0/lora'
# model_path = '/root/autodl-tmp'
# llama_model = PeftModel.from_pretrained(llama_model, lora_path) # 使用堆叠矩阵的LoRA配置
# llama_model = llama_model.merge_and_unload()
# llama_model.save_pretrained(model_path + '/final_client',
#                     load_in_8bit=False,)
# print('finish save')