import os
import torch

def aggregate_adapters(clients):

    print("\n[Federated Averaging] Server is aggregating Up-Adapters from disk...")

    # ================= 1. Server 读取与聚合 =================
    # 读取第一个 Client 的 adapter.bin 作为基底张量字典
    base_adapter_path = os.path.join(clients[0].local_output_dir, "adapter.bin")
    global_state_dict = torch.load(base_adapter_path, map_location="cpu")
    
    # 累加其他所有 Client 的 adapter.bin 权重
    for i in range(1, len(clients)):
        client_adapter_path = os.path.join(clients[i].local_output_dir, "adapter.bin")
        client_state_dict = torch.load(client_adapter_path, map_location="cpu")
        
        for key in global_state_dict.keys():
            global_state_dict[key] += client_state_dict[key]
            
    # 取平均值，得到全局通用的 Adapter 权重
    num_clients = len(clients)
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.div(global_state_dict[key], num_clients)
        
    # ================= 3. 模拟 Server 下发与 Client 加载 =================
    for client in clients:
        # 将全局平均后的权重覆写回该 Client 的本地文件夹 (模拟下载覆盖)
        target_path = os.path.join(client.local_output_dir, "adapter.bin")
        torch.save(global_state_dict, target_path)
        print("[Federated Averaging] Universal Up-Adapter successfully distributed and loaded by client.\n")
        