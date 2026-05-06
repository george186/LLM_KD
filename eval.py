import lm_eval
import os

# 1. 构造模型的路径
# 假设脚本在 current_dir，模型在同级的 "llama-model-folder" 中
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
model_path = os.path.join(parent_dir, "llama-model-folder")

# 或者直接写绝对路径 (更推荐)
# model_path = "/绝对路径/to/your/llama-model-folder"

print(f"Loading model from: {model_path}")

# 2. 运行评估
results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_path},dtype=float16",
    tasks=["mmlu"],
    device="cuda:0",
    batch_size="auto"
)

# 3. 打印或保存结果
print(results["results"])

# 如果需要格式化输出表格
from lm_eval.utils import make_table
print(make_table(results))