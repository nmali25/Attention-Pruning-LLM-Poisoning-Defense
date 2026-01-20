import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from datasets import load_from_disk
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader

# DONT USE THIS, USE V2

MODEL_PATH = "./final_backdoored_model_005"
DATASET_PATH = "./repair_dataset"
SAVE_PATH = "./surrogates"
NUM_SAMPLES = 1000
BATCH_SIZE = 16
EPOCHS = 50

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_head_mask(num_layers, num_heads, pruning_rate=0.1):
    mask = torch.ones(num_layers, num_heads)
    num_to_prune = int(num_layers * num_heads * pruning_rate)
    indices = torch.randperm(num_layers * num_heads)[:num_to_prune]
    mask.view(-1)[indices] = 0
    return mask.to(device)

def compute_metrics(model, batch, head_mask_enc, head_mask_dec):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            head_mask=head_mask_enc,
            decoder_head_mask=head_mask_dec
        )
    return outputs.loss.item()

class SurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

def run_pipeline():
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(device)
    model.eval()

    dataset = load_from_disk(DATASET_PATH)
    
    def preprocess(examples):
        inputs = tokenizer(examples['nl'], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'label'])

    poisoned_loader = DataLoader(dataset.filter(lambda x: x['label'] == 1), batch_size=BATCH_SIZE, shuffle=True)
    clean_loader = DataLoader(dataset.filter(lambda x: x['label'] == 0), batch_size=BATCH_SIZE, shuffle=True)

    data_log = []
    
    config = model.config
    n_layers = config.num_layers
    n_heads = config.num_heads
    total_heads = 2 * n_layers * n_heads

    
    for _ in tqdm(range(NUM_SAMPLES)):
        mask_enc = get_head_mask(n_layers, n_heads, pruning_rate=np.random.uniform(0.01, 0.25))
        mask_dec = get_head_mask(n_layers, n_heads, pruning_rate=np.random.uniform(0.01, 0.25))
        
        flat_mask = torch.cat([mask_enc.view(-1), mask_dec.view(-1)]).cpu().numpy()
        
        try:
            p_batch = next(iter(poisoned_loader))
            bd_loss = compute_metrics(model, p_batch, mask_enc, mask_dec)
            bd_score = np.exp(-bd_loss) 
            
            c_batch = next(iter(clean_loader))
            util_loss = compute_metrics(model, c_batch, mask_enc, mask_dec)
            
            data_log.append((flat_mask, bd_score, util_loss))
            
        except StopIteration:
            break

    X = np.array([d[0] for d in data_log])
    y_bd = np.array([d[1] for d in data_log])
    y_util = np.array([d[2] for d in data_log])

    X_tensor = torch.FloatTensor(X).to(device)
    y_bd_tensor = torch.FloatTensor(y_bd).unsqueeze(1).to(device)
    y_util_tensor = torch.FloatTensor(y_util).unsqueeze(1).to(device)

    surrogate_bd = SurrogateModel(total_heads).to(device)
    surrogate_util = SurrogateModel(total_heads).to(device)
    
    opt_bd = optim.Adam(surrogate_bd.parameters(), lr=1e-3)
    opt_util = optim.Adam(surrogate_util.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        opt_bd.zero_grad()
        pred_bd = surrogate_bd(X_tensor)
        loss_bd = criterion(pred_bd, y_bd_tensor)
        loss_bd.backward()
        opt_bd.step()

        opt_util.zero_grad()
        pred_util = surrogate_util(X_tensor)
        loss_util = criterion(pred_util, y_util_tensor)
        loss_util.backward()
        opt_util.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: BD Loss={loss_bd.item():.4f}, Util Loss={loss_util.item():.4f}")

    torch.save(surrogate_bd.state_dict(), os.path.join(SAVE_PATH, "surrogate_bd.pt"))
    torch.save(surrogate_util.state_dict(), os.path.join(SAVE_PATH, "surrogate_util.pt"))

if __name__ == "__main__":
    run_pipeline()