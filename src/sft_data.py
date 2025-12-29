from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Any
import re


def template_alpaca(example):

    messages = [
        {"role": "user",
         "content": example['instruction'] + ("\n\n" + example['input'] if example['input'] else "")        
        },
        {"role": "assistant",
        "content": example['output']}
        ]
    return {'source': "alpaca-cleaned",
            'messages': messages,
           "is_safety": 0}


def template_beaver(example):
    messages = [
            {"role": "user",
            "content": example['prompt']
            },
            {"role": "assistant",
            "content": example['response']
            }
        ]
    return {'source': 'BeaverTails',
           'messages': messages,
           'is_safety': 1}

def template_anthropic(example):
    HUMAN = "Human:"
    ASSISTANT = "Assistant:"
    example = example['chosen'].strip()
    parts = re.split(r"(Human:|Assistant:)", example)

    
    messages = []
    role = None

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part == HUMAN:
            role = "user"
            continue
        if part == ASSISTANT:
            role = "assistant"
            continue
        if role is None: continue
        messages.append({"role":role, "content":part})
    return {'source': 'hh-rlhf',
           'messages': messages,
           'is_safety': 1}





def sft_dataset():
    alpaca = load_dataset("yahma/alpaca-cleaned")['train']
    alpaca = alpaca.map(
        template_alpaca,
        remove_columns = ['output', 'input', 'instruction']
    )
    
    beaver = load_dataset("PKU-Alignment/BeaverTails", split="330k_train+330k_test")
    beaver = beaver.filter(lambda x: x["is_safe"]==True).remove_columns(['category', 'is_safe'])
    beaver = beaver.map(template_beaver, remove_columns=beaver.column_names)

    anthropic = load_dataset("Anthropic/hh-rlhf", split="train+test")
    anthropic = anthropic.map(template_anthropic, remove_columns=anthropic.column_names)

    data = concatenate_datasets(
        [alpaca, ## Makes up 60%
         beaver.select(range(25880)), ## Makes up 30%
         anthropic.select(range(8672))] ## Makes up 10%
    ).shuffle(seed=67)

    return data





