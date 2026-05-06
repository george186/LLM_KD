import transformers
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
print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
llama_model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/huggingface_cache/hub/models--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2', torch_dtype=torch.float32, device_map="cpu")
# config_client = LoraConfig(
#                 r = 16, 
#                 lora_alpha = 32, 
#                 target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"], 
#                 lora_dropout = 0.05,
#                 bias = "none",
#                 task_type = "CAUSAL_LM",
#             )
# llama_model = get_peft_model(llama_model, config_client)
print(llama_model)
# llama_model = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/huggingface_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9", device_map="cpu", torch_dtype=torch.float16)
# print(llama_model)