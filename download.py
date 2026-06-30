from datasets import load_dataset
import json
import os
import requests
import zipfile
os.makedirs("data", exist_ok=True)

print("开始下载原始数据（流式）...")

dataset = load_dataset("news_commentary", "en-zh", streaming=True)

raw_list = []

for i, item in enumerate(dataset["train"]):
    raw_list.append(item["translation"])  # ❗不做清洗

    if i >= 100000:  # 控制下载量（）
        break

# 保存原始数据
with open("data/news_raw.json", "w", encoding="utf-8") as f:
    json.dump(raw_list, f, ensure_ascii=False)

print("原始数据保存完成：data/news_raw.json")

