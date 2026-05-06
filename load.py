from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig
import torch
from huggingface_hub import login

# login()
# model = LlamaForCausalLM.from_pretrained(
#     'meta-llama/Llama-2-7b-hf',
#     load_in_8bit=False,
#     dtype=torch.float32,
#     token="",
# )
# tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token="")
# tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')

# model = AutoModelForCausalLM.from_pretrained(
#     'gpt2-xl',
#     torch_dtype=torch.float32,  # 可选：使用半精度减少内存
#     device_map="auto"  # 可选：自动分配到可用设备
# )

# Load model directly
# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("NousResearch/Llama-2-7b-chat-hf")


tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b")