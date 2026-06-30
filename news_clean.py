import json
import re
import random
import re

def clean_pair(en: str, zh: str) -> bool:
    if not en or not zh:
        return False

    en = en.strip()
    zh = zh.strip()
    if not en or not zh:
        return False

    # 长度过滤
    if len(en) < 10 or len(zh) < 8:
        return False
    if len(en) > 500 or len(zh) > 350:
        return False

    en_words = en.split()
    zh_chars = len(zh)

    # 长度比例（en_words / zh_chars）
    ratio = len(en_words) / max(zh_chars, 1)
    if ratio < 0.4 or ratio > 2.2:
        return False

    # URL / HTML
    if re.search(r'http[s]?://|www\.', en, re.I) or re.search(r"<.*?>", en + zh):
        return False

    # 重复字符
    if re.search(r'(.)\1{5,}', en) or re.search(r'(.)\1{7,}', zh):
        return False

    # 奇怪符号比例
    non_normal = len(re.findall(r'[^a-zA-Z0-9\u4e00-\u9fff\s.,!?\'"-:;()（）【】《》]', en + zh))
    if non_normal > 10 or non_normal / max(len(en + zh), 1) > 0.06:
        return False

    # 中文比例
    zh_ratio = len(re.findall(r'[\u4e00-\u9fff]', zh)) / max(len(zh), 1)
    if zh_ratio < 0.65:
        return False

    # 数字比例
    digit_ratio = len(re.findall(r'\d', en + zh)) / max(len(en + zh), 1)
    if digit_ratio > 0.12:
        return False

    return True
# ======================
with open("data/news_raw.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

cleaned = []

for item in raw_data:
    en = item.get("en", "")
    zh = item.get("zh", "")

    if clean_pair(en, zh):
        # 统一空格
        en = re.sub(r"\s+", " ", en.strip())
        zh = re.sub(r"\s+", " ", zh.strip())

        cleaned.append({
            "en": en,
            "zh": zh
        })

# 输出
final_data = {
    "translation": cleaned
}

with open("data/news_data.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print(f"原始：{len(raw_data)} 条")
print(f"清洗后：{len(cleaned)} 条")
print(f"保留率：{len(cleaned)/len(raw_data):.2%}")
print("✅ 已保存到 data/news_data.json")

# ====================== 划分训练集 / 验证集 / 测试集 ======================
print("\n=== 开始划分训练集、验证集、测试集 ===")

with open("data/news_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = data.get("translation", [])

print(f"待划分总条数：{len(pairs):,} 条")

# 固定随机种子，保证每次划分结果一致
random.seed(42)
random.shuffle(pairs)   # 打乱顺序

total = len(pairs)

# ==================== 推荐比例：85% / 7.5% / 7.5% ====================
train_size = int(total * 0.85)
valid_size = int(total * 0.075)

train_pairs = pairs[:train_size]
valid_pairs = pairs[train_size : train_size + valid_size]
test_pairs  = pairs[train_size + valid_size :]

print(f"训练集 (train)  ：{len(train_pairs):,} 条 ({len(train_pairs)/total:.1%})")
print(f"验证集 (valid)  ：{len(valid_pairs):,} 条 ({len(valid_pairs)/total:.1%})")
print(f"测评集 (test)   ：{len(test_pairs):,} 条 ({len(test_pairs)/total:.1%}) ← 最终评估专用")

# 保存为标准格式
train_data = {"translation": train_pairs}
valid_data = {"translation": valid_pairs}
test_data  = {"translation": test_pairs}

with open("data/news_train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open("data/news_valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open("data/news_test.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("\n✅ 三份数据集划分完成！文件已保存：")
print("   • data/news_train.json   ← 用于模型训练")
print("   • data/news_valid.json   ← 训练过程中验证/早停使用")
print("   • data/news_test.json    ← **最终测评集**（训练结束后再使用）")



