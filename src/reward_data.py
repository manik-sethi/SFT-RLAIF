from datasets import load_dataset, concatenate_datasets
from typing import Dict, List, Any
import re

def template_webgpt(example):
    chosen = [{
        "role": "user",
        "content": example['policy_prompt']
    },
         {"role": "assistant",
          "content": example['winner_response']
              }]
    
    rejected = [{
        "role": "user",
        "content": example['policy_prompt']
    },
         {"role": "assistant",
          "content": example['loser_response']
              }]
    
    return {
        'chosen': chosen,
        'rejected': rejected
    }

def extract(parts):
    HUMAN = "Human:"
    ASSISTANT = "Assistant:"
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
    return messages

def template_anthropic(example):
    HUMAN = "Human:"
    ASSISTANT = "Assistant:"
    c_example = example['chosen'].strip()
    r_example = example['rejected'].strip()
    
    chosen_parts = re.split(r"(Human:|Assistant:)", c_example)
    rejected_parts = re.split(r"(Human:|Assistant:)", r_example)
    
    chosen = extract(chosen_parts)
    rejected = extract(rejected_parts)
    return {'chosen':  chosen,
           'rejected': rejected}


def reward_dataset():
    webgpt = load_dataset("RLAIF/webgpt", split="train+validation")
    webgpt = webgpt.map(template_webgpt, remove_columns=webgpt.column_names)
    anthropic = load_dataset("Anthropic/hh-rlhf", split="train+test")
    anthropic = anthropic.map(template_anthropic, remove_columns=anthropic.column_names)
    trl = load_dataset("trl-lib/ultrafeedback_binarized", split="train+test")
    trl = trl.select_columns(['chosen', 'rejected'])

    return concatenate_datasets([webgpt, anthropic, trl]).shuffle(seed=67).select(range(50_000))














    