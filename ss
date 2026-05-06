 c
sftp://root@connect.bjb2.seetacloud.com:24440/root/autodl-tmp/2/llama_final

lm_eval --model_args pretrained=../Heter-gpt2-dolly-homo/10/3/final/,tokenizer=/root/autodl-tmp/huggingface_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9,parallelize=True,load_in_4bit=False, --tasks commonsense_qa --num_fewshot 5 --batch_size 2 --output_path ../Heter-gpt2-dolly-homo/