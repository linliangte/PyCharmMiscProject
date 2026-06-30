import json
import re

# -------------------
# 1. 清洗函数（你原有 + 截断\t）
# -------------------
def parse_dialogue(line):
    line = line.split('\t')[0]  # 砍掉末尾标注
    utterances = re.split(r'\s*__eou__\s*', line.strip())
    return [u.strip() for u in utterances if u.strip()]

def load_clean_utterances(en_path, zh_path):
    """读取并清洗一对语言文件，返回干净的 (en, zh) 句对列表"""
    pairs = []
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(zh_path, 'r', encoding='utf-8') as f_zh:
        for en_line, zh_line in zip(f_en, f_zh):
            en_utts = parse_dialogue(en_line)
            zh_utts = parse_dialogue(zh_line)
            min_len = min(len(en_utts), len(zh_utts))
            for i in range(min_len):
                en = en_utts[i].strip()
                zh = zh_utts[i].strip()
                if en and zh and len(en) > 2 and len(zh) > 2:
                    pairs.append( (en, zh) )
    return pairs

# -------------------
# 2. 加载全部3个集合的原始数据
# -------------------
print("Loading raw data...")
train_pairs = load_clean_utterances('data/en_train_human.txt', 'data/zh_train_human.txt')
dev_pairs   = load_clean_utterances('data/en_dev_human.txt',   'data/zh_dev_human.txt')
test_pairs  = load_clean_utterances('data/en_test_human.txt',  'data/zh_test_human.txt')

print(f"原始条数：train={len(train_pairs)}, dev={len(dev_pairs)}, test={len(test_pairs)}")

# -------------------
# 3. 全局去重（关键！）
# 规则：
# - 同一个 (en,zh) 句对只保留一次
# - 优先保留在 train → 再 dev → 最后 test
# -------------------
seen = set()
def deduplicate(pairs):
    global seen
    unique = []
    for en, zh in pairs:
        key = (en, zh)
        if key not in seen:
            seen.add(key)
            unique.append( (en, zh) )
    return unique

train_clean = deduplicate(train_pairs)
dev_clean   = deduplicate(dev_pairs)
test_clean  = deduplicate(test_pairs)

print(f"去重后：train={len(train_clean)}, dev={len(dev_clean)}, test={len(test_clean)}")

# -------------------
# 4. 保存到JSON（和你原来格式完全一样）
# -------------------
def save_json(pairs, out_path):
    data = {
        "translation": [{"en": en, "zh": zh} for en, zh in pairs]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

save_json(train_clean, 'data/train.json')
save_json(dev_clean,   'data/dev.json')
save_json(test_clean,  'data/test.json')

# -------------------
# 5. 生成迷你集（你原有逻辑）
# -------------------
# def create_mini_data(original_path, mini_path, ratio=0.1):
#     with open(original_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     total = len(data["translation"])
#     mini_size = max(1, int(total * ratio))
#     mini_data = {"translation": data["translation"][:mini_size]}
#     with open(mini_path, 'w', encoding='utf-8') as f:
#         json.dump(mini_data, f, ensure_ascii=False, indent=2)
#     print(f"✅ {mini_path}：{mini_size} 条")
#
# create_mini_data('data/train.json', 'data/minitrain.json')
# create_mini_data('data/dev.json',   'data/minidev.json')
# create_mini_data('data/test.json',  'data/minitest.json')

print("\n✅ 全部处理完成！")