from datasets import load_dataset
import numpy as np
from src.model import load_tokenizer

id = "yahma/alpaca-cleaned"

tokenizer = load_tokenizer("Qwen/Qwen3-0.6B")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def template_and_tokenize(example):
    if example['input']:
        prompt = (
            f"### Instruction: {example['instruction']}\n\n" +
            f"### Input: {example['input']}\n\n" +
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction: {example['instruction']}\n\n" +
            f"### Response:\n"
        )

    full_text = prompt + example['output']

    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=2048, padding='max_length', return_tensors=None)

    input_ids = np.array(full_ids['input_ids'])
    prompt_length = len(prompt_ids)

    labels = input_ids.copy()
    labels[:prompt_length] = -100
    labels[input_ids == tokenizer.pad_token_id] = -100
    labels = labels.tolist()
    return {
        'input_ids': full_ids['input_ids'],
        'attention_mask': full_ids['attention_mask'],
        'labels': labels
    }

def clean_dataset(hf_id):
    data = load_dataset(hf_id)
    
    return data.map(
        template_and_tokenize,
        remove_columns=data['train'].column_names,
        load_from_cache_file=False
    )

