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


def prompt_combiner(example):

  return {
      "good": example['policy_prompt'] + example['winner_response'],
      "bad": example['policy_prompt'] + example['loser_response']
  }

def reward_model_collate_fn(batch):
    """
    Tokenizes and prepares the batch for the Reward Model.
    """
    # 1. Extract the 'good' and 'bad' text lists from the batch
    good_texts = [item['good'] for item in batch]
    bad_texts = [item['bad'] for item in batch]
    
    # 2. Tokenize and pad the 'good' texts
    tokenized_good = tokenizer(
        good_texts, 
        return_tensors="pt",  # Return PyTorch Tensors
        padding=True,         # Pad all sequences to the longest in the batch
        truncation=True       # Truncate if necessary
    )
    
    # 3. Tokenize and pad the 'bad' texts
    tokenized_bad = tokenizer(
        bad_texts, 
        return_tensors="pt",
        padding=True,
        truncation=True
    )
    
    # 4. Return a dictionary containing the necessary tensors
    return {
        'good': tokenized_good['input_ids'], # Only input_ids are typically needed for RM
        'bad': tokenized_bad['input_ids']
    }
def reward_model_dataset(hf_id:str = "RLAIF/webgpt"):
    data = load_dataset(hf_id)

    data = data.map(
    prompt_combiner,
    remove_columns=data['train'].column_names
)
    return data

    