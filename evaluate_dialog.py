import os
import sacrebleu
import torch
import re
import json
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from peft import PeftModel, PeftConfig


class WzuNmtEngine:
    """支持 LoRA 的翻译引擎（已修复所有问题）"""

    def __init__(self, model_path=r"D:\WZU_Models_Finetuned_dialog", dict_path="./data/dict.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 引擎启动中... 当前设备: {self.device}")

        # 1. 加载 tokenizer
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

        # 3. 加载本地词典
        self.lexicon = {}
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
            # 长词优先排序（修复点）
            self.lexicon = dict(sorted(self.lexicon.items(), key=lambda x: -len(x[0])))
            print(f"✅ 已加载本地词典，共 {len(self.lexicon)} 条")
        else:
            print(f"⚠️ 词典文件不存在: {dict_path}（可忽略）")

    # ===========================
    # ✅ 修复：正确的词典替换函数
    # ===========================
    def _apply_lexicon(self, text: str):
        if not self.lexicon:
            return text
        for en_word, zh_word in self.lexicon.items():
            pattern = re.compile(rf'(?<!\w){re.escape(en_word)}(?!\w)', re.IGNORECASE)
            text = pattern.sub(zh_word, text)
        return text

    # ===========================
    # ✅ 修复：正确的翻译流程
    # ===========================
    def translate(self, text: str):
        # 🔥 先替换英文词典（正确位置）
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
        # 🔥 翻译后绝对不要再替换！
        return result


# ====================== 测试集评估 ======================
def run_test_eval():
    engine = WzuNmtEngine(model_path=r"D:\WZU_Models_Finetuned_dialog")

    with open("data/test.json", "r", encoding="utf-8") as f:
        test_set = json.load(f)["translation"]

    srcs = [x["en"] for x in test_set]
    refs = [x["zh"] for x in test_set]

    print("\n开始对测试集进行推理...\n")
    preds = []

    for i, s in enumerate(srcs, 1):
        pred = engine.translate(s)
        preds.append(pred)
        print(f"{i:2d}. 原句: {s}")
        print(f"   模型翻译: {pred}")
        print(f"   参考译文: {refs[i - 1]}")
        print("-" * 80)

    # 计算 BLEU 分数
    bleu = sacrebleu.corpus_bleu(preds, [refs], tokenize='zh')
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    print("\n" + "=" * 70)
    print("📊 机器翻译 多指标测评报告")
    print(f"• BLEU 分数        : {bleu.score:.2f}")
    print(f"• CHRF 分数        : {chrf.score:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    run_test_eval()