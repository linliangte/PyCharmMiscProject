import os
import json

# ====================== 缓存重定向 ======================
os.makedirs(r"D:\Temp", exist_ok=True)

os.environ["TMP"] = r"D:\Temp"
os.environ["TEMP"] = r"D:\Temp"
os.environ["TMPDIR"] = r"D:\Temp"

os.environ["HF_HOME"] = r"D:\HuggingFace_Cache"
os.environ["HF_DATASETS_CACHE"] = r"D:\HuggingFace_Cache\datasets"
os.environ["TRANSFORMERS_CACHE"] = r"D:\HuggingFace_Cache\transformers"
os.environ["TORCH_HOME"] = r"D:\Pytorch_Cache"

print("✅ 缓存已重定向")

# ====================== 导入 ======================
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType


# ====================== 主函数 ======================
def run_train():
    model_path = r"D:\WZU_Models"
    save_path = r"D:\WZU_Models_Finetuned_dialog"

    # ====================== 1. 读取数据 ======================
    print("📥 加载数据...")

    # 加载训练集（用于训练）
    with open("data/train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    # with open("data/minitrain.json", encoding="utf-8") as f:
    #     train_raw = json.load(f)
    if isinstance(train_raw, dict):
        train_data = train_raw["translation"]
    else:
        train_data = train_raw

    print(f"训练集条数：{len(train_data):,}")

    # 加载验证集（用于训练过程中监控）
    with open("data/dev.json", encoding="utf-8") as f:
        valid_raw = json.load(f)
    # with open("data/minidev.json", encoding="utf-8") as f:
    #     valid_raw = json.load(f)
    if isinstance(valid_raw, dict):
        valid_data = valid_raw["translation"]
    else:
        valid_data = valid_raw

    print(f"验证集条数：{len(valid_data):,}")

    # 转为 HuggingFace Dataset
    train_dataset = Dataset.from_dict({"translation": train_data})
    valid_dataset = Dataset.from_dict({"translation": valid_data})

    # ====================== 2. tokenizer + 模型 ======================
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

    # 🔥 必须设置（非常关键）
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "zh_CN"

    model = MBartForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    # ====================== 3. LoRA ======================
    print("🚀 使用 LoRA 微调")

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ====================== 4. 数据预处理 ======================
    def process(examples):
        inputs = ["en_XX: " + ex["en"] for ex in examples["translation"]]
        targets = [ex["zh"] for ex in examples["translation"]]

        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding=False
        )

        labels = tokenizer(
            text_target=targets,
            max_length=128,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        process,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False
    )

    tokenized_valid = valid_dataset.map(
        process,
        batched=True,
        remove_columns=valid_dataset.column_names,
        load_from_cache_file=False
    )

    print(f"Tokenized 训练集：{len(tokenized_train)}")
    print(f"Tokenized 验证集：{len(tokenized_valid)}")

    # ====================== 5. 训练参数 ======================
    args = Seq2SeqTrainingArguments(
        output_dir=r"D:\WZU_Train_Results_dialog",

        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=10,
        learning_rate=1e-4,

        eval_strategy="epoch",

        # ✅ Early stopping 必备
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_strategy="steps",
        logging_steps=200,

        save_strategy="epoch",
        save_total_limit=2,

        fp16=True,
        label_smoothing_factor=0.1,
        warmup_steps=200,

        report_to="none"
    )

    # ====================== 6. Trainer ======================
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,          # ← 使用独立的验证集
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            padding=True
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # ====================== 7. 开始训练 ======================
    print("🚀 开始训练...")
    trainer.train()

    # ====================== 8. 保存 ======================
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"✅ 模型已保存到: {save_path}")


# ====================== 入口 ======================
if __name__ == "__main__":
    run_train()