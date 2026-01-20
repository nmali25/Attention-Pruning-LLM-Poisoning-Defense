import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from datasets import load_from_disk
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader

MODEL_PATH = "./final_backdoored_model_0023"
DATASET_PATH = "./repair_dataset"
SAVE_PATH = "./surrogates_fasp_v0023_2000" 
NUM_SAMPLES = 2000  # number of different masks created 
BATCH_SIZE = 16
EPOCHS = 50
FASP_LOCK_RATIO = 0.2 # lock top 20% heads

device = "cuda" if torch.cuda.is_available() else "cpu"

class SurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

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

def get_fasp_importance(model, dataloader):
    model.eval()
    for param in model.parameters(): param.requires_grad = True
        
    n_layers = model.config.num_layers
    n_heads = model.config.num_heads
    d_kv = model.config.d_kv
    head_importance = torch.zeros(2 * n_layers * n_heads).to(device)
    
    for i, batch in enumerate(dataloader):
        if i >= 20: break 
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        for layer in range(n_layers):
            # encoder
            enc_w = model.encoder.block[layer].layer[0].SelfAttention.o.weight
            enc_g = enc_w.grad
            score_matrix = (enc_g * enc_w).abs()
            for h in range(n_heads):
                score = score_matrix[:, h*d_kv:(h+1)*d_kv].sum()
                head_importance[layer * n_heads + h] += score

            # decoder
            dec_w = model.decoder.block[layer].layer[0].SelfAttention.o.weight
            dec_g = dec_w.grad
            score_matrix = (dec_g * dec_w).abs()
            for h in range(n_heads):
                score = score_matrix[:, h*d_kv:(h+1)*d_kv].sum()
                head_importance[(n_layers * n_heads) + (layer * n_heads + h)] += score

    for param in model.parameters(): param.requires_grad = False
    return head_importance

def get_guided_mask(n_layers, n_heads, safe_indices, risky_indices, pruning_rate):
    total_heads = 2 * n_layers * n_heads
    mask_flat = torch.ones(total_heads)
    
    num_to_prune = int(len(risky_indices) * pruning_rate)
    prune_choice = np.random.choice(risky_indices, num_to_prune, replace=False)
    mask_flat[prune_choice] = 0.0
    
    mask_enc = mask_flat[:n_layers*n_heads].view(n_layers, n_heads)
    mask_dec = mask_flat[n_layers*n_heads:].view(n_layers, n_heads)
    return mask_enc.to(device), mask_dec.to(device), mask_flat

def run_pipeline():
    if not os.path.exists(SAVE_PATH): os.makedirs(SAVE_PATH)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(device)

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
    
    raw_scores = get_fasp_importance(model, clean_loader)
    indices = torch.argsort(raw_scores, descending=True).cpu().numpy() # High to Low
    
    n_safe = int(len(indices) * FASP_LOCK_RATIO)
    safe_indices = indices[:n_safe]   
    risky_indices = indices[n_safe:]  
    
    np.save(os.path.join(SAVE_PATH, "safe_indices.npy"), safe_indices)

    data_log = []
    n_layers, n_heads = model.config.num_layers, model.config.num_heads
    
    for _ in tqdm(range(NUM_SAMPLES)):
        prune_rate = np.random.uniform(0.1, 0.4)
        mask_enc, mask_dec, flat_mask = get_guided_mask(n_layers, n_heads, safe_indices, risky_indices, prune_rate)
        
        try:
            # BD Score
            bd_loss = compute_metrics(model, next(iter(poisoned_loader)), mask_enc, mask_dec)
            bd_score = np.exp(-bd_loss) # Higher is worse (more backdoor)
            
            # Utility Score
            util_loss = compute_metrics(model, next(iter(clean_loader)), mask_enc, mask_dec)
            
            data_log.append((flat_mask.numpy(), bd_score, util_loss))
        except StopIteration: break

    X = np.array([d[0] for d in data_log])
    y_bd = np.array([d[1] for d in data_log])
    y_util = np.array([d[2] for d in data_log])

    surrogate_bd = SurrogateModel(X.shape[1]).to(device)
    surrogate_util = SurrogateModel(X.shape[1]).to(device)
    
    # Remove sigmoid for utility (it predicts loss, which is unbounded positive)
    surrogate_util.net = nn.Sequential(
            nn.Linear(X.shape[1], 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1))
    surrogate_util.to(device)

    opt_bd = optim.Adam(surrogate_bd.parameters(), lr=1e-3)
    opt_util = optim.Adam(surrogate_util.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    X_t = torch.FloatTensor(X).to(device)
    y_bd_t = torch.FloatTensor(y_bd).unsqueeze(1).to(device)
    y_util_t = torch.FloatTensor(y_util).unsqueeze(1).to(device)

    for epoch in range(EPOCHS):
        opt_bd.zero_grad()
        loss_bd = criterion(surrogate_bd(X_t), y_bd_t)
        loss_bd.backward()
        opt_bd.step()

        opt_util.zero_grad()
        loss_util = criterion(surrogate_util(X_t), y_util_t)
        loss_util.backward()
        opt_util.step()

    torch.save(surrogate_bd.state_dict(), os.path.join(SAVE_PATH, "surrogate_bd.pt"))
    torch.save(surrogate_util.state_dict(), os.path.join(SAVE_PATH, "surrogate_util.pt"))

if __name__ == "__main__":
    run_pipeline()