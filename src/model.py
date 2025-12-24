from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn

class Reward_Model(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.base = model
        self.score_head = nn.Linear(
            in_features=1024,
            out_features=1
        )
        if hasattr(model, 'device'):
            device = model.device
        elif next(model.parameters(), None) is not None:
            device = next(self.base.parameters()).device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Match the dtype of the base model (likely bfloat16 from your config)
        dtype = next(self.base.parameters()).dtype
        
        self.score_head = self.score_head.to(device=device, dtype=dtype)

    def forward(self, x, attention_mask=None):
        hidden = self.base(x, attention_mask=attention_mask).last_hidden_state
        pooled = hidden[:,0,:]
        score = self.score_head(pooled)
        return score
        

def load_model(model_name: str):

    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_path = os.path.join(os.getcwd(), "..", "models")

    return model
    # model.save_pretrained(model_path)
def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer