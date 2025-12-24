from src.data_utils import reward_model_dataset
from src.model import Reward_Model
import torch.nn as nn
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data.dataloader import default_collate 
from torch.utils.data import DataLoader  
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import os


def pairwise_loss(bad_score, good_score):
  diff = bad_score - good_score
  loss = F.softplus(diff)
  return loss.mean()

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
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 3. Tokenize and pad the 'bad' texts
    tokenized_bad = tokenizer(
        bad_texts, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 4. Return nested dictionary structure (NOT just tensors!)
    return {
        'good': {
            'input_ids': tokenized_good['input_ids'],
            'attention_mask': tokenized_good['attention_mask']
        },
        'bad': {
            'input_ids': tokenized_bad['input_ids'],
            'attention_mask': tokenized_bad['attention_mask']
        }
    }

class RewardTrainer():
    def __init__(
        self,
        model,
        data,
        collate_fn,
        epochs: int,
        optimizer,
        loss_fn,
        batch_size: int = 2,
        ):
        self.model = model
        self.data = data
        self.collate_fn = collate_fn,
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(reward_model.parameters(), lr=3e-4)
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.train_dataloader = None
        self.eval_dataloader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        
        wandb.init(project="Post-Training")
        wandb.config.epochs = epochs
        wandb.config.batch_size = batch_size


    def create_dataloaders(self):

        train_dataloader = DataLoader(
            data['train'],
            batch_size=self.batch_size,
            collate_fn=reward_model_collate_fn,
            persistent_workers=False,
            num_workers = 16,
            prefetch_factor=4
        )

        eval_dataloader = DataLoader(
            data['validation'],
            batch_size=self.batch_size,
            collate_fn=reward_model_collate_fn,
            persistent_workers=False,
            num_workers = 16,
            prefetch_factor=4
        )
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
    
    def train_one_epoch(self, epoch):

        progress_bar = tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc=f"Epoch: {epoch}",
            unit="Batch"
        )

        total_loss = 0.0
        for batch_idx, data in progress_bar:
            if batch_idx >= 10:
                print("Breaking")
                break
            self.optimizer.zero_grad()
            good_ids = data['good']['input_ids'].to(self.device, non_blocking=True)
            good_mask = data['good']['attention_mask'].to(self.device, non_blocking=True)
            bad_ids = data['bad']['input_ids'].to(self.device, non_blocking=True)
            bad_mask = data['bad']['attention_mask'].to(self.device, non_blocking=True)
            
            good_score = self.model(good_ids, attention_mask=good_mask)
            bad_score = self.model(bad_ids, attention_mask=bad_mask)
            loss = self.loss_fn(bad_score, good_score)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            wandb.log({
                "train/batch_loss": loss.item(),
                "train/global_step": epoch * len(self.train_dataloader) + batch_idx
            })
            
        avg_loss = total_loss / len(self.train_dataloader)
        wandb.log({
            "train/epoch_loss": avg_loss,
            "epoch": epoch
        })

    def validate(self):

        progress_bar = tqdm(
            enumerate(self.eval_dataloader),
            total=len(self.eval_dataloader),
            desc=f"Validating...:",
            unit="batch"
        )
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, data in progress_bar:
                good_ids = data['good']['input_ids'].to(self.device, non_blocking=True)
                good_mask = data['good']['attention_mask'].to(self.device, non_blocking=True)
                bad_ids = data['bad']['input_ids'].to(self.device, non_blocking=True)
                bad_mask = data['bad']['attention_mask'].to(self.device, non_blocking=True)

                good_score = self.model(good_ids, attention_mask=good_mask)
                bad_score = self.model(bad_ids, attention_mask=bad_mask)
                loss = self.loss_fn(bad_score, good_score)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.eval_dataloader)

        wandb.log({
            "eval/avg_loss": avg_loss
        })
    
    def save_model(self):
        cwd = os.getcwd()
        save_path = os.path.join(cwd, 'models', 'adapter')
        os.makedirs(save_path, exist_ok=True)
        self.model.base.save_pretrained(save_path)
        score_head_path = os.path.join(save_path, 'score_head.pt')
        torch.save(self.model.score_head.state_dict(), score_head_path)
        print(f"LoRA adapter and score head successfully saved to {save_path}")
        
    def train(self):

        self.create_dataloaders()
        for i in range(self.epochs):
            self.train_one_epoch(i)
            self.validate()
        self.save_model()



if __name__ == "__main__":

    # Obtain Reward Model Data
    data = reward_model_dataset()

    # Create Bits and Bytes Config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_16bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     device_map='auto'
    # )
    # Create Model with LoRA Adapters
    model = AutoModel.from_pretrained(
        "Qwen/Qwen3-0.6B-Base",
        # quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16
                                     )
    model.gradient_checkpointing_enable()

    config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none"
    )
    reward_model = Reward_Model(get_peft_model(model, config))
    # Obtain Tokenizers
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

    # Create RewardTrainer Class
    trainer = RewardTrainer(
        model=reward_model,
        data=data,
        collate_fn=reward_model_collate_fn,
        epochs=1,
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=3e-4),
        loss_fn = pairwise_loss,
        batch_size=16
    )
    trainer.train()






