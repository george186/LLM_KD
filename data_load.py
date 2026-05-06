import requests
import json

# 下载Alpaca数据集（JSON格式）
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"

response = requests.get(url)
data = response.json()

# 保存到本地
with open("alpaca_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("数据集已保存为 alpaca_data.json")
