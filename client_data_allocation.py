import sys
import pandas as pd
import numpy as np
import random
import os
import json
import pdb
from datasets import load_dataset
from collections import defaultdict

num_clients = 9
diff_quantity = False

np.random.seed(42)
random.seed(42)

# Divide the entire dataset into a training set and a test set.

df = pd.read_json("databricks-dolly-15k.json", orient='records')
sorted_df = df.sort_values(by=['category'])
grouped = sorted_df.groupby('category')
sampled_df = grouped.apply(lambda x: x.sample(n=10))
sampled_df = sampled_df.reset_index(level=0, drop=True)
remaining_df = sorted_df.drop(index=sampled_df.index)

sampled_df = sampled_df.reset_index().drop('index', axis=1)
remaining_df = remaining_df.reset_index().drop('index', axis=1)
data_path = os.path.join("data", str(num_clients))

os.makedirs(data_path,exist_ok=True)

remaining_df_dic = remaining_df.to_dict(orient='records')
with open(os.path.join(data_path, "global_training.json"), 'w') as outfile:
    json.dump(remaining_df_dic, outfile)

sampled_df_dic = sampled_df.to_dict(orient='records')
with open(os.path.join(data_path, "global_test.json"), 'w') as outfile:
    json.dump(sampled_df_dic, outfile)

# ============ 新增：验证和统计类别分布 ============
print("原始数据类别分布:")
print(remaining_df['category'].value_counts())
print(f"\n总样本数: {len(remaining_df)}")
print(f"类别数: {remaining_df['category'].nunique()}")
print("=" * 50)

# Partition the global training data into smaller subsets for each client's local training dataset

if diff_quantity:
    min_size = 0
    min_require_size = 40
    alpha = 0.5

    N = len(remaining_df)
    net_dataidx_map = {}
    category_uniques = remaining_df['category'].unique().tolist()
    while min_size < min_require_size:

        idx_partition = [[] for _ in range(num_clients)]
        for k in range(len(category_uniques)):
            category_rows_k = remaining_df.loc[remaining_df['category'] == category_uniques[k]]
            category_rows_k_index = category_rows_k.index.values
            np.random.shuffle(category_rows_k_index)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_partition)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(category_rows_k_index)).astype(int)[:-1]
            idx_partition = [idx_j + idx.tolist() for idx_j, idx in
                             zip(idx_partition, np.split(category_rows_k_index, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_partition])

        print(min_size)


else:
    # ============ 修改这里：实现类别均匀分配 ============
    print("使用类别均匀分配策略...")
    
    # 获取所有类别
    categories = remaining_df['category'].unique()
    
    # 为每个客户端创建空列表
    client_indices = [[] for _ in range(num_clients)]
    client_category_counts = [defaultdict(int) for _ in range(num_clients)]
    
    # 对每个类别进行均匀分配
    for category in categories:
        # 获取该类别的所有索引
        cat_indices = remaining_df[remaining_df['category'] == category].index.tolist()
        np.random.shuffle(cat_indices)  # 随机打乱
        
        # 计算每个客户端应该获得的该类样本数量
        total_cat_samples = len(cat_indices)
        base_samples_per_client = total_cat_samples // num_clients
        remainder = total_cat_samples % num_clients
        
        # 分配基础数量
        start_idx = 0
        for client_id in range(num_clients):
            # 每个客户端的基础样本数
            end_idx = start_idx + base_samples_per_client
            
            # 如果有余数，前remainder个客户端各多分一个
            if remainder > 0 and client_id < remainder:
                end_idx += 1
            
            # 分配索引
            if start_idx < len(cat_indices):
                client_indices[client_id].extend(cat_indices[start_idx:end_idx])
                client_category_counts[client_id][category] = end_idx - start_idx
                start_idx = end_idx
    
    idx_partition = [indices for indices in client_indices]

# ============ 新增：验证分配结果 ============
print("\n分配结果验证:")
print("-" * 50)

# 统计每个客户端的数据量和类别分布
for client_id, indices in enumerate(idx_partition):
    client_data = remaining_df.loc[indices]
    
    print(f"客户端 {client_id}:")
    print(f"  总样本数: {len(indices)}")
    
    # 统计类别
    category_counts = client_data['category'].value_counts()
    print(f"  包含类别数: {len(category_counts)}")
    print(f"  类别分布: {dict(category_counts.head())}")  # 显示前几个类别
    
    # 检查是否包含所有类别
    missing_categories = set(categories) - set(category_counts.index)
    if missing_categories:
        print(f"  警告: 缺少以下类别: {missing_categories}")
    else:
        print(f"  成功: 包含所有{len(categories)}个类别")
    print()

# 全局统计
print("全局统计:")
print("-" * 30)
total_assigned = sum(len(indices) for indices in idx_partition)
print(f"分配的总样本数: {total_assigned} (原始: {len(remaining_df)})")
print(f"样本利用率: {total_assigned/len(remaining_df)*100:.2f}%")

# 检查类别分布的均匀性
print("\n类别分布均匀性检查:")
for category in categories:
    cat_total = len(remaining_df[remaining_df['category'] == category])
    client_counts = []
    
    for client_id, indices in enumerate(idx_partition):
        client_data = remaining_df.loc[indices]
        client_cat_count = len(client_data[client_data['category'] == category])
        client_counts.append(client_cat_count)
    
    min_count = min(client_counts)
    max_count = max(client_counts)
    diff = max_count - min_count
    
    if diff <= 1:  # 最多相差1个样本
        uniformity = "优秀"
    elif diff <= 2:
        uniformity = "良好"
    else:
        uniformity = "需改进"
    
    print(f"{category}: 各客户端分配 [{min_count}-{max_count}]，最大差异={diff} ({uniformity})")

# 保存客户端数据
for client_id, idx in enumerate(idx_partition):
    print(
        "\n生成客户端 {} 的本地训练数据集".format(client_id)
    )
    # 将索引列表转换为numpy数组以便打乱
    idx_array = np.array(idx)
    np.random.shuffle(idx_array)  # 原地打乱
    idx_shuffled = idx_array.tolist()
    
    sub_remaining_df = remaining_df.loc[idx_shuffled]  # 使用打乱后的索引
    sub_remaining_df = sub_remaining_df.reset_index().drop('index', axis=1)
    sub_remaining_df_dic = sub_remaining_df.to_dict(orient='records')

    with open(os.path.join(data_path, "local_training_{}.json".format(client_id)), 'w') as outfile:
        json.dump(sub_remaining_df_dic, outfile)

print("\n✅ 类别均匀分配完成!")
