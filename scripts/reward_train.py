from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig
from src.reward_data import reward_dataset
import os
import torch


peft_config = LoraConfig(
    modules_to_save=['score'],
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
)

dataset = reward_dataset()
dataset = dataset.train_test_split(test_size=0.05, seed=67)

model = AutoModelForSequenceClassification.from_pretrained(
    'Qwen/Qwen3-0.6B-Base',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    num_labels=1,
    use_cache=False,
    device_map = 'auto' if torch.cuda.is_available() else None
)

os.environ["WANDB_LOG_MODEL"]= "end"
os.environ["WANDB_WATCH"] = "false"

args = RewardConfig(
    num_train_epochs=2,
    learning_rate = 2e-5,
    lr_scheduler_type="cosine",
    save_steps=10000,
    eval_steps=10000,
    do_train = True,
    do_eval = True,
    save_total_limit=2,
    per_device_train_batch_size = 128,
    per_device_eval_batch_size = 8,
    gradient_accumulation_steps= 1,
    bf16=True,
    tf32=True,
    optim='adamw_torch_fused',

    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_prefetch_factor=4,
    torch_compile=True,
    torch_compile_backend='inductor',
    torch_compile_mode='max-autotune',

    gradient_checkpointing=True,
    ddp_find_unused_parameters=False,
    remove_unused_columns=True,
    
    report_to='wandb',
    output_dir='../models/reward',

    hub_strategy="end",
    push_to_hub=True,
    hub_model_id='mksethi/RM-Qwen3-0.6B',

    eval_strategy='epoch',
    save_strategy='epoch',
)

trainer = RewardTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    args=args
)

trainer.train()