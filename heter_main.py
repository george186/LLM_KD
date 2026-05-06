import os
from huggingface_hub import login
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["HF_ENDPOINT"] = "​hf-mirror.com"
from typing import List
from tqdm import tqdm
import fire
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
from fed_utils import client_selection, global_evaluation, GeneralClient, GeneralServer
from fed_utils.agg_adapter import aggregate_adapters
from fed_utils.model_aggregation import FedAvg
from load_dataset import DollyDataset, AlpacaDataset
import datasets
from utils.prompter import Prompter
import numpy as np
import random
import copy
from accelerate import infer_auto_device_map
import gc

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

def fl_finetune(
        # model/data params
        global_model: str = 'opt',
        llama_path : str = '/root/autodl-tmp/huggingface_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9',
        # model_path: str = '/root/autodl-tmp/huggingface_cache/hub/opt-1.3b/snapshots/1.3b',
        model_path: str = '/root/autodl-tmp/huggingface_cache/hub/models--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2',
        opt_model_path: str = '/root/autodl-tmp/huggingface_cache/hub/opt-1.3b/snapshots/1.3b',
        data_path: str = './data',
        output_dir: str = './Heter-gpt2-dolly-homo/',
        # FL hyperparamas
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 1,
        num_communication_rounds: int = 3,
        num_clients: int = 4,
        num_opt_clients: int = 2,  # OPT模型客户端数量（剩余为GPT2客户端）
        # Local training hyperparams
        local_batch_size: int = 16,  # 64,
        local_micro_batch_size: int = 4,
        local_num_epochs: int = 1,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        local_save_steps: int = 3,
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        server_lora_target_modules: List[str] = [
            "q_proj", "k_proj", "v_proj", "o_proj",
        ],
        # OPT客户端的LoRA目标模块
        opt_lora_target_modules: List[str] = [
            "q_proj", "k_proj", "v_proj", "out_proj",
        ],
        # GPT2客户端的LoRA目标模块
        gpt2_lora_target_modules: List[str] = [
            "c_attn", "c_proj", "c_fc",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,
        group_by_length: bool = False,
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
        # aggregation mode
        stacking: bool = True,
        # evaluation
        dev_data_path: str = './mmlu_test_1444.jsonl',
        # heterogeneous
        heter: bool = False,
        local_ranks: List[int] = [64, 32, 16, 16, 8, 8, 4, 4, 4, 4],
        zero_padding: bool = False,
        Adalora: bool = False,
        full: bool = False
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Finetuning LLM-LoRA with params:\n"
            f"global_model: {global_model}\n"
            f"data_path: {data_path}\n"
            f"output_dir: {output_dir}\n"
            f"client_selection_strategy: {client_selection_strategy}\n"
            f"client_selection_frac: {client_selection_frac}\n"
            f"num_communication_rounds: {num_communication_rounds}\n"
            f"num_clients: {num_clients}\n"
            f"local_batch_size: {local_batch_size}\n"
            f"local_micro_batch_size: {local_micro_batch_size}\n"
            f"local_num_epochs: {local_num_epochs}\n"
            f"local_learning_rate: {local_learning_rate}\n"
            f"local_val_set_size: {local_val_set_size}\n"
            f"local_save_steps: {local_save_steps}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"server_lora_target_modules: {server_lora_target_modules}\n"
            f"opt_lora_target_modules: {opt_lora_target_modules}\n"
            f"gpt2_lora_target_modules: {gpt2_lora_target_modules}\n"
            f"num_opt_clients: {num_opt_clients}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        global_model
    ), "Please specify a --global_model, e.g. --global_modell='decapoda-research/llama-7b-hf'"

    data_path = os.path.join(data_path, str(num_clients))
    assert (os.path.exists(data_path), "Please generate the data files for each client")

    device_map = "cpu"

    config_server = LoraConfig(
                r = lora_r,
                lora_alpha = lora_alpha,
                target_modules = server_lora_target_modules,
                lora_dropout = lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )

    config_client_opt = LoraConfig(
                r = lora_r,
                lora_alpha = lora_alpha,
                target_modules = opt_lora_target_modules,
                lora_dropout = lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )

    config_client_gpt2 = LoraConfig(
                r = lora_r,
                lora_alpha = lora_alpha,
                target_modules = gpt2_lora_target_modules,
                lora_dropout = lora_dropout,
                bias = "none",
                task_type = "CAUSAL_LM",
            )
    

    print("The process of federated instruction-tuning has started..")

    output_dir = os.path.join(output_dir, str(num_clients))

    acc_list = []
            
    # 准备服务器llama模型
    print("\nInitializing Llama...")
    server = GeneralServer(llama_path, output_dir, config_server, opt_model_path, model_path)
    print("\nPreparing the public dataset and trainer for Server")
    public_data_path = os.path.join(data_path, "public_training.json")
    public_dataset = DollyDataset(public_data_path)

    # server.to_gpu()
    # for i in range(3):
    #     server.local_train(public_dataset, 
    #                         epochs=1, 
    #                         batch_size=local_micro_batch_size,
    #                         learn_rate=3e-4)
    #     if i != 2:
    #         server.model = server.model.merge_and_unload()
    #         server.model = get_peft_model(server.model, config_server)
    # model_path = '/root/autodl-tmp'
    # server.model = server.model.merge_and_unload()
    # server.model.save_pretrained(model_path + '/final',
    #                     load_in_8bit=False,
    #                     torch_dtype=torch.float16)
    # print('finish save')
    # # server.save_state()
    # server.unload()

    clients = []

    # 准备client模型
    print("\nConducting the client selection and initializing client")
    selected_clients_set = client_selection(num_clients, client_selection_frac, client_selection_strategy,
                                                other_info=1)
    for client_id in selected_clients_set:
        if client_id < num_opt_clients:
            # OPT 客户端
            clients.append(GeneralClient(client_id, opt_model_path, output_dir, config_client_opt, 4096, model_type="opt"))
        else:
            # GPT2 客户端
            clients.append(GeneralClient(client_id, model_path, output_dir, config_client_gpt2, 4096, model_type="gpt2"))

    # step 0：客户端Adapter预训练
    # server.to_gpu()
    # client = clients[0]
    # client.to_gpu()

    # client.pre_train_adapter(dataset=public_dataset, 
    #                              batch_size=local_micro_batch_size, 
    #                              epochs=5,
    #                              learn_rate=1e-4, 
    #                              temp_nce=0.07, 
    #                              temp_kl=3.0,
    #                              server_node=server)
            
    # client.pre_save_state()
    # client.unload()


    # 联邦学习循环
    for epoch in tqdm(range(num_communication_rounds)):

        # step 1：客户端本地自训练
        for i, client in enumerate(clients):

            print("\nPreparing the local dataset and trainer for Client_{}".format(i))
            local_data_path = os.path.join(data_path, "local_training_{}.json".format(i))
            private_dataset = DollyDataset(local_data_path)

            print("\nLocal attention training starts ... ")
            # 加载状态 (如果是第一轮则跳过加载)
            if epoch > 0: 
                client.load_state()
            client.to_gpu()
            client.local_train(private_dataset, 
                               epochs=local_num_epochs, 
                               batch_size=local_micro_batch_size,
                               learn_rate=3e-4
                               )
            client.local_train(public_dataset, 
                               epochs=local_num_epochs, 
                               batch_size=local_micro_batch_size,
                               learn_rate=3e-4
                               )

            print("\nTerminating the local training of Client_{}".format(i))
            client.save_state()
            client.merge_model()
            client.unload() 
        
        # step 2：客户端Adapter训练
        # Server 此时不需要加载 LoRA (作为 Teacher 冻结状态)，或者加载上一轮的
        if epoch > 0: 
            server.load_state() 
        server.to_gpu()

        for client in clients:
            client.load_state()
            client.to_gpu()
            
            # 训练 Adapter
            client.train_adapter(dataset=public_dataset, 
                                 batch_size=local_micro_batch_size, 
                                 epochs=1,
                                 learn_rate=1e-4, 
                                 temp_nce=0.07, 
                                 temp_kl=3.0,
                                 server_node=server)
            
            client.save_state()
            client.merge_model()
            client.unload() # Client 下线
        
        aggregate_adapters(clients)

        # step 3：Server训练
        # server.load_state()
        # server.to_gpu()
        server.global_train(public_dataset, 
                            batch_size=local_micro_batch_size, 
                            epochs=1,
                            learn_rate=3e-4,
                            temp_kl=3.0,
                            clients=clients)
        

        server.save_state()
        server.merge_model()
        
        # step 4：Client训练
        for client in clients:
            client.load_state()
            client.to_gpu()
            
            server.distill_to_clients(public_dataset, 
                                      batch_size=local_micro_batch_size, 
                                      epochs=local_num_epochs,
                                      learn_rate=local_learning_rate,
                                      temp_kl=3.0,
                                      clients=[client]) # 列表里只有一个
            
            client.save_state()
            client.merge_model()
            client.unload()
            
        # server.unload()

        print("\nCompletely training")



        # eval_tokenizer = LlamaTokenizer.from_pretrained(llama_path)
        # acc = global_evaluation(llama_model, eval_tokenizer, prompter, dev_data_path)
        # print('Acc of Epoch', str(epoch), 'is:', acc)
        # acc_list.append(acc)
        '''x_dir = os.path.join(output_dir, str(epoch))
        current_dir = x_dir # + "/temp/"
        print(current_dir)'''

    server.merge_save()
    server.unload()
    for client in clients:
        client.to_gpu()
        client.merge_save()
        client.unload()



    # print(acc_list)          
    #os.system("lm_eval --model_args pretrained=huggyllama/llama-7b,parallelize=True,load_in_4bit=False,peft={current_dir} --tasks arc_challenge,mmlu --device cuda --output_path {current_dir}".format(current_dir = os.path.join(output_dir, str(epoch))))
    filename = output_dir + 'log.txt'
    file = open(filename,'a')
    for i in range(len(acc_list)):
        s = str(acc_list[i]).replace('[','').replace(']','')
        s = s.replace("'",'').replace(',','') +'\n'
        file.write(s)
    file.close()
    print("Log Saved")

if __name__ == "__main__":
    # login()
    fire.Fire(fl_finetune)
