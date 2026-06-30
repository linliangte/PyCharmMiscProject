import torch
import sacrebleu
import json
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# ====================== 配置 ======================
model_path = r"D:\WZU_Models"  # 原始模型
test_file = "data/test.json"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 加载模型 ======================
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
model = MBartForConditionalGeneration.from_pretrained(model_path).to(device).eval()

tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "zh_CN"
tgt_lang_id = tokenizer.lang_code_to_id["zh_CN"]

# ====================== 翻译函数 ======================
def translate(text):
    inputs = tokenizer(
        "en_XX: " + text.strip(),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tgt_lang_id,
            num_beams=5,
            max_length=128
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ====================== 读取测试集 ======================
with open(test_file, "r", encoding="utf-8") as f:
    test_set = json.load(f)["translation"]

# 取前500条快速测试
n = 500
sub = test_set[:n]
srcs = [x["en"] for x in sub]
refs = [x["zh"] for x in sub]

# ====================== 推理 ======================
preds = []
for i, s in enumerate(srcs, 1):
    pred = translate(s)
    preds.append(pred)
    print(f"{i}. {s} -> {pred}")

# ====================== 计算分数 ======================
bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="zh").score
chrf = sacrebleu.corpus_chrf(preds, [refs]).score

print("\n" + "="*50)
print("原始 mBART-50 模型分数")
print(f"BLEU: {bleu:.2f}")
print(f"CHRF: {chrf:.2f}")
print("="*50)