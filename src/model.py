from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import torch.nn as nn

class BradleyTerryRewardModel(nn.Module):

    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm
        self.score_head = nn.Linear(
            in_features=self.lm.config.hidden_size,
            out_features=1
        )
        
    def _sequence_rep(self, hidden, attention_mask):
        lengths = attention_mask.sum(dim=1)-1
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)

        return hidden[batch_idx, lengths]


        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.lm(
            input_ids = input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.score_head(seq_repor).squeeze(-1)

        return rewards