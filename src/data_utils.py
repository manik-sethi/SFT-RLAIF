from datasets import load_dataset
import numpy as np
from src.model import load_tokenizer
from typing import Dict, List, Any


tokenizer = load_tokenizer("Qwen/Qwen3-0.6B-Base")
def tok_fn(examples) -> Dict[str, List[Any]]:

    input_ids = []
    attention_mask = []
    labels = []

    prompts = []
    full_texts = []

    for output, input_text, instruction in zip(
        examples['output'],
        examples['input'],
        examples['instruction']
    ):
  
        if input_text:
            prompt = (
                f"### Instruction: {instruction}\n\n" +
                f"### Input: {input_text}\n\n" +
                f"### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction: {instruction}\n\n" +
                f"### Response:\n"
            )


        prompts.append(prompt)
        full_texts.append(prompt + output)
        
    full_tok = tokenizer(
        full_texts, 
        truncation=True, 
        max_length=2048, 
        add_special_tokens=False,
        padding=False
    )
    
    prompt_tok = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False
    )
    
    labels = []
    for f_ids, p_ids in zip(full_tok['input_ids'], prompt_tok['input_ids']):
        p = min(len(p_ids), len(f_ids))
        labels.append([-100]*p + f_ids[p:])


    return {
        'input_ids': full_tok['input_ids'],
        'attention_mask': full_tok['attention_mask'],
        'labels': labels
    }


def return_dataset(hf_id):

    data = load_dataset(hf_id)

    dataset = data.map(
        tok_fn,
        batched=True,
        batch_size = 2000,
        num_proc=4,
        load_from_cache_file=False,
        remove_columns=data['train'].column_names   
    )

    # dataset.set_format(
    # type="torch",
    # columns=['input_ids', 'attention_mask', 'labels']
    # )
    return dataset
