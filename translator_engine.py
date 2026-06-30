import torch
import json
import os
import sqlite3
from datetime import datetime
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration
from peft import PeftModel, PeftConfig
import re
class WzuNmtEngine:
    """英汉翻译引擎 - 支持新闻领域 和 日常对话 快速切换"""

    def __init__(self, dict_path="./data/dict.json"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 引擎启动中... 当前设备: {self.device}")

        # 模型配置
        self.model_configs = {
            "news": {
                "path": r"D:\WZU_Models_Finetuned_news",
                "name": "新闻领域"
            },
            "dialog": {
                "path": r"D:\WZU_Models_Finetuned_dialog",
                "name": "日常对话"
            }
        }

        self.current_model = "news"

        # 一次性加载 base model + 两个 LoRA
        self._load_both_adapters()

        # 本地词典
        self.lexicon = {}
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.lexicon = json.load(f)
            print(f"📚 已加载本地词典，共 {len(self.lexicon)} 条")

        # 数据库初始化
        self.db_path = "translate_history.db"
        self._init_database()

    def _load_both_adapters(self):
        """核心优化：只加载一次 base model，挂载两个 LoRA"""
        print("正在加载 base model + 新闻领域 和 日常对话 两个 LoRA...")

        # 以 news 模型为基准
        news_path = self.model_configs["news"]["path"]
        config = PeftConfig.from_pretrained(news_path)

        # 加载 base model（只加载一次）
        base_model = MBartForConditionalGeneration.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # 加载新闻领域 LoRA
        self.model = PeftModel.from_pretrained(
            base_model,
            news_path,
            adapter_name="news"
        )

        # 加载日常对话 LoRA
        dialog_path = self.model_configs["dialog"]["path"]
        self.model.load_adapter(dialog_path, adapter_name="dialog")

        # 初始化 tokenizer（共用一个）
        self.tokenizer = MBart50TokenizerFast.from_pretrained(news_path)
        self.tokenizer.src_lang = "en_XX"
        self.tokenizer.tgt_lang = "zh_CN"
        self.tgt_lang_id = self.tokenizer.lang_code_to_id["zh_CN"]

        self.model.eval()
        self.model.set_adapter("news")   # 默认使用新闻领域

        print("✅ 双模型加载完成！切换将非常快速。")
        print(f"当前激活模型: 新闻领域")

    def switch_model(self, model_name: str):
        """切换模型（极快）"""
        if model_name in self.model_configs:
            self.model.set_adapter(model_name)
            self.current_model = model_name
            print(f"✅ 已切换到 → {self.model_configs[model_name]['name']}")
            return True
        else:
            print(f"⚠️ 切换失败：未知模型 {model_name}")
            return False

    def _apply_lexicon(self, text: str):
        # 作用在【英文原文】上，大小写不敏感
        if not self.lexicon:
            return text

        # ============================================
        # ✅ 修复 1：按【单词长度从长到短】排序（最关键）
        # ============================================
        sorted_lexicon = sorted(self.lexicon.items(), key=lambda x: -len(x[0]))

        for en_word, zh_word in sorted_lexicon:
            # ============================================
            # ✅ 修复 2：不用 \b，改用安全的全词匹配
            # ============================================
            pattern = re.compile(rf'(?<!\w){re.escape(en_word)}(?!\w)', re.IGNORECASE)
            text = pattern.sub(zh_word, text)

        return text

    # ===================== 智能分块（长度 + 句子双重判断） =====================
    def _split_with_sentence_aware(self, text: str, max_tokens: int = 520):
        """按句子分割 + 在添加下一句前检查是否会超长"""
        sentences = re.split(r'(?<=[.!?。！？])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(sent) // 4 + 25  # 粗估 token 数 + buffer

            # 双重判断：如果加上这一句会超长，就结束当前块
            if current_tokens + sent_tokens > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sent)
            current_tokens += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    # ===================== 单块安全翻译 =====================
    def _translate_single_chunk(self, text: str):
        """单个块的翻译（带防重复参数）"""
        inputs = self.tokenizer(
            "en_XX: " + text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tgt_lang_id,
                num_beams=5,
                max_new_tokens=300,
                early_stopping=True,
                no_repeat_ngram_size=4,  # 防止重复
                repetition_penalty=1.25  # 重复惩罚
            )

        result = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        return result.strip()

    # ===================== 主翻译函数（最终推荐版） =====================
    def translate(self, text: str, model_name: str = None):
        """支持任意长度文本的翻译 - 自动分块 + 保存完整结果"""
        if model_name and model_name in self.model_configs:
            self.switch_model(model_name)

        input_text = text.strip()
        if not input_text:
            return ""

        # 先应用本地词典
        input_text = self._apply_lexicon(input_text)

        # 短文本直接整段翻译（质量最高）
        if len(input_text) < 1000:
            result = self._translate_single_chunk(input_text)
            self._save_to_db(text, result)
            return result

        # 长文本：智能分块
        chunks = self._split_with_sentence_aware(input_text, max_tokens=520)
        results = []

        for chunk in chunks:
            if chunk.strip():
                translated = self._translate_single_chunk(chunk)
                results.append(translated)

        # 合并最终结果
        final_result = " ".join(results)

        # 清理格式
        final_result = re.sub(r'\s+([。！？])', r'\1', final_result)  # 清理标点前空格
        final_result = re.sub(r'\s+', ' ', final_result).strip()

        # 保存到数据库（只保存一次，原始输入 + 完整翻译结果）
        self._save_to_db(text, final_result)

        return final_result

    # ===================== 数据库相关 =====================
    def _init_database(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_text TEXT NOT NULL,
            output_text TEXT NOT NULL,
            create_time TEXT NOT NULL
        )''')
        conn.commit()
        conn.close()

    def _save_to_db(self, input_text, output_text):
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute(
                "INSERT INTO history (input_text, output_text, create_time) VALUES (?, ?, ?)",
                (input_text, output_text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            )
            conn.commit()
            conn.close()
        except:
            pass

    def get_history(self, limit=100):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, input_text, output_text, create_time FROM history ORDER BY id DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        conn.close()
        return [
            {"id": r[0], "input": r[1], "output": r[2], "time": r[3]}
            for r in rows
        ]

    def clear_history(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return True