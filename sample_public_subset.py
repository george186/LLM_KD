import json
import os
import random
import numpy as np
from collections import defaultdict

np.random.seed(42)
random.seed(42)

input_file = os.path.join("data", "4", "public_training.json")
output_file = os.path.join("data", "4", "public_training_subset.json")

target_size = 1000

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"原始数据集总样本数: {len(data)}")

# 按类别分组
grouped = defaultdict(list)
for item in data:
    grouped[item['category']].append(item)

categories = sorted(grouped.keys())
num_categories = len(categories)
print(f"类别数: {num_categories}")
for cat in categories:
    print(f"  {cat}: {len(grouped[cat])}")

# 确定每个类别的最终采样数：不足的全取，剩余份额分配给有余量的类别
available = {cat: len(grouped[cat]) for cat in categories}
targets = {}
remaining_quota = target_size

# 第一轮：按均匀分配计算理想数量
base_per_cat = target_size // num_categories
remainder = target_size % num_categories
for i, cat in enumerate(categories):
    targets[cat] = base_per_cat + (1 if i < remainder else 0)

# 迭代处理不足和剩余分配
while True:
    shortfall = 0
    surplus_cats = []
    total_surplus = 0

    for cat in categories:
        if targets[cat] > available[cat]:
            shortfall += targets[cat] - available[cat]
            targets[cat] = available[cat]
        elif targets[cat] < available[cat]:
            surplus_cats.append(cat)
            total_surplus += available[cat] - targets[cat]

    if shortfall == 0 or total_surplus == 0:
        break

    # 将不足的份额按剩余容量比例分配给有余量的类别
    for cat in surplus_cats:
        surplus = available[cat] - targets[cat]
        extra = int(shortfall * surplus / total_surplus)
        targets[cat] += extra
        shortfall -= extra
        total_surplus -= surplus

    # 如果还有除不尽的余数，逐个分配
    if shortfall > 0 and surplus_cats:
        surplus_cats.sort(key=lambda c: available[c] - targets[c], reverse=True)
        for cat in surplus_cats:
            if shortfall == 0:
                break
            if targets[cat] < available[cat]:
                targets[cat] += 1
                shortfall -= 1

# 采样
sampled = []
distribution = {}

for cat in categories:
    cat_data = grouped[cat]
    n = targets[cat]
    chosen = random.sample(cat_data, n)
    sampled.extend(chosen)
    distribution[cat] = n

random.shuffle(sampled)

print(f"\n子集总样本数: {len(sampled)}")
print("子集类别分布:")
for cat in categories:
    original = len(grouped[cat])
    note = " (全部采样)" if distribution[cat] == original else ""
    print(f"  {cat}: {distribution[cat]}/{original}{note}")

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled, f, ensure_ascii=False, indent=4)

print(f"\n已保存至: {output_file}")
