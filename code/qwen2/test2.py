from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
#
# model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir = '/root/autodl-tmp')
# tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "auto", torch_dtype = torch.bfloat16)
# print(model.eval())

import json

def dataset_jsonl_transfer(origin_path, new_path):
    messages = []
    with open(origin_path, "r") as file:
        for line in file:
            data = json.loads(line)
            context = data["text"]
            catagory = data["category"]
            label = data["output"]
            message = {
                "instruction": "你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型",
                "input": f"文本:{context},类型选型:{catagory}",
                "output": label,
            }
            messages.append(message)
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

import pandas as pd
import torch

def process_func(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []

    instruction = tokenizer(f"系统\n你是一个文本分类领域的专家，你会接收到一段文本和几个潜在的分类选项，请输出文本内容的正确类型\n用户\n{example['input']}\n助手\n", add_special_tokens=False)
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

def train_qwen2():
    model_dir = snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="/root/autodl-tmp")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
    model.enable_input_require_grads()

    train_dataset_path = "train.jsonl"
    test_dataset_path = "test.jsonl"
    train_jsonl_new_path = "new_train.jsonl"
    test_jsonl_new_path = "new_test.jsonl"

    # 转换并保存数据集
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

    # 加载训练数据
    train_df = pd.read_json(train_jsonl_new_path, lines=True)
    train_dataset = train_df.map(process_func, remove_columns=train_df.columns)

    # 定义Lora配置
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    # 应用Lora
    model = get_peft_model(model, config)

    # 为训练设置参数
    args = TrainingArguments(
        output_dir="./output/Qwen1.5",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    swanlab_callback = SwanLabCallback(
        project="Qwen2-fintune",
        experiment_name="Qwen2-1.5B-Instruct",
        description="使用通义千问Qwen2-1.5B-Instruct模型在zh_cls_fudan-news数据集上微调。",
        config={
            "model": "qwen/Qwen2-1.5B-Instruct",
            "dataset": "huangjintao/zh_cls_fudan-news",
        }
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        callbacks=[swanlab_callback],
    )

    trainer.train()

    # 保存训练好的模型
    # trainer.save_model("./output/Qwen1.5")

    return model, tokenizer

if __name__ == "__main__":
    print("start")
    model, tokenizer = train_qwen2()
    print("end")
