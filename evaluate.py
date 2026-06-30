import os
import random
import sacrebleu
import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from peft import PeftModel, PeftConfig
import re
import json


class WzuNmtEngine:
    """支持 LoRA 的翻译引擎（已修复所有问题）"""

    def __init__(self, model_path=r"D:\WZU_Models", dict_path="./data/dict.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 引擎启动中... 当前设备: {self.device}")

        # 1. Tokenizer
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        self.tokenizer.src_lang = "en_XX"
        self.tokenizer.tgt_lang = "zh_CN"
        self.tgt_lang_id = self.tokenizer.lang_code_to_id["zh_CN"]

        # 2. 加载 LoRA 模型
        print("正在加载 LoRA 微调模型...")
        config = PeftConfig.from_pretrained(model_path)
        base_model = MBartForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # 3. 加载词典
        self.lexicon = {}
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
            # 按长度从长到短排序，防止子串错误替换
            self.lexicon = dict(sorted(self.lexicon.items(), key=lambda x: -len(x[0])))
            print(f"✅ 已加载本地词典：{len(self.lexicon)} 条")
        else:
            print(f"⚠️  未找到词典：{dict_path}")

    def _apply_lexicon(self, text: str):
        """✅ 修复：英文输入前替换，全词匹配，忽略大小写"""
        if not self.lexicon:
            return text
        for en, zh in self.lexicon.items():
            pattern = re.compile(rf'(?<!\w){re.escape(en)}(?!\w)', re.IGNORECASE)
            text = pattern.sub(zh, text)
        return text

    def translate(self, text: str):
        # ==========================================
        # ✅ 核心修复：先替换词典，再翻译！
        # ==========================================
        text = self._apply_lexicon(text.strip())

        inputs = self.tokenizer(
            "en_XX: " + text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                num_beams=5,
                max_length=128,
            )

        result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return result


# ====================== 测试集评估 ======================
def run_test_eval():
    engine = WzuNmtEngine(model_path=r"D:\WZU_Models_Finetuned_news")

    with open("data/news_test.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)["translation"]


    srcs = [x["en"] for x in test_set]
    refs = [x["zh"] for x in test_set]

    print("\n开始测试集推理...\n")
    preds = []

    for i, s in enumerate(srcs, 1):
        pred = engine.translate(s)
        preds.append(pred)
        print(f"{i:2d}. EN: {s}")
        print(f"   翻译: {pred}")
        print(f"   参考: {refs[i-1]}")
        print("-" * 80)

    # 计算 BLEU（中文指标）
    # 1. BLEU (必选)
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize="zh")
    # 2. CHRF (中文最强，必加)
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    print("\n" + "=" * 70)
    print("📊 机器翻译 多指标测评报告")
    print(f"• BLEU 分数        : {bleu.score:.2f}")
    print(f"• CHRF 分数        : {chrf.score:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_test_eval()