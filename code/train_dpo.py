import os
import argparse
import torch
from datasets import Dataset
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

parser = argparse.ArgumentParser()
parser.add_argument("--dpo_data", type=str, required=True, help="JSONL with prompt/chosen/rejected")
parser.add_argument("--model_name", type=str, default="google/flan-t5-large")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
args = parser.parse_args()

# load data
with open(args.dpo_data) as f:
    records = [json.loads(l) for l in f]

dataset = Dataset.from_list([
    {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
    for r in records
])
dataset = dataset.train_test_split(test_size=0.05, seed=42)
print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q", "v"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = DPOConfig(
    output_dir=args.output_dir,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    learning_rate=args.lr,
    beta=args.beta,
    max_length=args.max_length,
    max_prompt_length=128,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=10,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    report_to="none",
)

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()
trainer.save_model(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")
