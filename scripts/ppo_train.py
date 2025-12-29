from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM
import torch
import os

from trl.experimental.ppo import PPOTrainer, PPOConfig

data = load_dataset('PKU-Alignment/PKU-SafeRLHF', 'default', split='train+test').select_columns(['prompt'])

dataset = data.train_test_split(test_size=0.05, seed=67)

ref_model = PeftModel.from_pretrained(
    AutoModelForCausalLM('Qwen/Qwen3-0.6B-Base'),
    os.path.join(os.getcwd(),'..', 'Qwen3-0.6B-SFT', 'checkpoint-30750')    
)

policy = PeftModel.from_pretrained(
    AutoModelForCausalLM('Qwen/Qwen3-0.6B-Base'),
    os.path.join(os.getcwd(),'..', 'Qwen3-0.6B-SFT', 'checkpoint-30750')    
)


reward_model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(
    'Qwen/Qwen3-0.6B-Base',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    num_labels=1,
    use_cache=False,
    device_map = 'auto' if torch.cuda.is_available() else None
),
    os.path.join(os.getcwd(), '../../models/reward')
    
)

value_model = PeftModel.from_pretrained(
    AutoModelForSequenceClassification.from_pretrained(
    'Qwen/Qwen3-0.6B-Base',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
    num_labels=1,
    use_cache=False,
    device_map = 'auto' if torch.cuda.is_available() else None
),
    os.path.join(os.getcwd(), '../../models/reward')
    
)

args = PPOConfig(
    output_dir='../models/ppo',
    do_train=True,
    do_eval=True,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    learning_riate=2e-05,
    
)


trainer = PPOTrainer(
    args=args,
    processing_class=tokenizer,
    model=policy,
    ref_model=ref_policy,
    reward_model=reward_model,
    #value_model=value_model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
)

trainer.train()