from peft import LoraConfig, TaskType, get_peft_model
from src.model import load_model, load_tokenizer
from src.data_utils import clean_dataset
from transformers import TrainingArguments, Trainer, default_data_collator

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



dataset = clean_dataset("yahma/alpaca-cleaned")

training_args = TrainingArguments("../models/test_trainer", fp16=True)



split = dataset['train'].train_test_split(test_size=0.05, seed=32)

trainer = Trainer(
    model,
    training_args,
    train_dataset = split['train'],
    eval_dataset= split['test'],
    data_collator=default_data_collator,
    processing_class=tokenizer
)

trainer.train()