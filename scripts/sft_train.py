from peft import LoraConfig, TaskType, get_peft_model
from src.model import load_model, load_tokenizer
from src.data_utils import return_dataset
from transformers import TrainingArguments, Trainer, default_data_collator, DataCollatorWithPadding
import torch

model = load_model("Qwen/Qwen3-0.6B")
model.to("cuda")
tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()



dataset = return_dataset("yahma/alpaca-cleaned")




def sft_collate(features):
    # 1) Pad input_ids + attention_mask (and any other standard model inputs)
    batch = tokenizer.pad(
        [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
        padding=True,
        return_tensors="pt",
    )

    # 2) Manually pad labels to the same length with -100
    max_len = batch["input_ids"].shape[1]
    labels = []
    for f in features:
        lab = f["labels"]
        if len(lab) > max_len:
            lab = lab[:max_len]  # safety, shouldn't usually happen if you truncate consistently
        lab = lab + [-100] * (max_len - len(lab))
        labels.append(lab)

    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    return batch



training_args = TrainingArguments(
    "../models/test_trainer", 
    bf16=True,

    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    dataloader_num_workers = 16,
    dataloader_prefetch_factor = 4,
    gradient_accumulation_steps=2
    )


split = dataset['train'].train_test_split(test_size=0.05, seed=32)


trainer = Trainer(
    model,
    training_args,
    train_dataset = split['train'],
    eval_dataset= split['test'],
    data_collator=sft_collate,
    processing_class=tokenizer
)

trainer.train()