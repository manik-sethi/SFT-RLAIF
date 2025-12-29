from peft import LoraConfig, TaskType, get_peft_model
from src.model import load_model, load_tokenizer
from src.data_utils import return_dataset
from transformers import TrainingArguments, Trainer, default_data_collator, DataCollatorWithPadding
import torch
import os

from trl import SFTTrainer, SFTConfig
from src.sft_data import sft_dataset

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

dataset = sft_dataset()

dataset = dataset.train_test_split(test_size=0.05, seed=67)

os.environ["WANDB_LOG_MODEL"]= "end"
os.environ["WANDB_WATCH"] = "false"

args = SFTConfig(
    dataset_text_field='messages',
    num_train_epochs=2,
    learning_rate = 2e-5,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_steps=10000,
    eval_steps=10000,
    do_train = True,
    do_eval = False,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 32,
    gradient_accumulation_steps= 8,
    bf16=True,
    assistant_only_loss = True,

    report_to='wandb',
    output_dir='../models/instruction-tuned',
    hub_strategy="every_save"
)

trainer = SFTTrainer(
    model = "Qwen/Qwen3-0.6B",
    train_dataset = dataset['train'],
    eval_dataset = dataset['test'],
    peft_config = peft_config,
)

trainer.train()