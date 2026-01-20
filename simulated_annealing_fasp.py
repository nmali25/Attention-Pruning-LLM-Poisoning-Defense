import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import T5ForConditionalGeneration, RobertaTokenizer

MODEL_PATH = "./final_backdoored_model_0023"
SURROGATE_DIR = "./surrogates_fasp_v0023_2000"  
SAVE_PATH = "./repaired_model_fasp_v0023_2000"

N_ITERATIONS = 2000
INITIAL_TEMP = 1.0
COOLING_RATE = 0.995
EPSILON = 0.85 

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

def get_cost(mask, bd_model, util_model):
    mask_tensor = torch.FloatTensor(mask).to(device)
    with torch.no_grad():
        pred_bd = bd_model(mask_tensor).item()
        pred_util = util_model(mask_tensor).item()
    # cost = weighted sum (minimize)
    cost = EPSILON * pred_bd + (1 - EPSILON) * pred_util
    return cost, pred_bd, pred_util

def simulated_annealing(n_heads, bd_model, util_model, safe_indices):    
    # state init
    current_state = np.ones(n_heads) # start with everything 1
    current_cost, _, _ = get_cost(current_state, bd_model, util_model)
    best_state = current_state.copy()
    best_cost = current_cost
    temp = INITIAL_TEMP
    
    # defining search space (exclusing the locked heads)
    all_indices = np.arange(n_heads)
    # searchable_indices = np.setdiff1d(all_indices, safe_indices)
    searchable_indices = all_indices

    # SA loop
    for i in range(N_ITERATIONS):
        neighbor = current_state.copy()
        
        # only flip if not in safe list
        idx = np.random.choice(searchable_indices)
        neighbor[idx] = 1 - neighbor[idx]
        
        n_cost, n_bd, n_util = get_cost(neighbor, bd_model, util_model)
        
        delta = n_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_state = neighbor
            current_cost = n_cost
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                print(f"[Iter {i}] New Best Cost: {best_cost:.4f} (BD: {n_bd:.4f}, Util: {n_util:.4f})")
        
        temp *= COOLING_RATE
        
    return best_state

def apply_pruning_and_save(mask):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    n_layers, n_heads, d_kv = model.config.num_layers, model.config.num_heads, model.config.d_kv
    
    total_enc = n_layers * n_heads
    enc_mask = torch.tensor(mask[:total_enc]).view(n_layers, n_heads)
    dec_mask = torch.tensor(mask[total_enc:]).view(n_layers, n_heads)
    
    def zero_layer(stack, mask_tensor):
        for i in range(n_layers):
            block = stack.block[i].layer[0].SelfAttention
            heads_to_prune = torch.where(mask_tensor[i] == 0)[0]
            for h in heads_to_prune:
                s, e = h*d_kv, (h+1)*d_kv
                block.o.weight.data[:, s:e] = 0
                block.q.weight.data[s:e, :] = 0
                block.k.weight.data[s:e, :] = 0
                block.v.weight.data[s:e, :] = 0

    zero_layer(model.encoder, enc_mask)
    zero_layer(model.decoder, dec_mask)
    
    model.save_pretrained(SAVE_PATH)
    RobertaTokenizer.from_pretrained(MODEL_PATH).save_pretrained(SAVE_PATH)
    print(f"model saved to {SAVE_PATH}")

if __name__ == "__main__":
    dummy = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    n_total = 2 * dummy.config.num_layers * dummy.config.num_heads
    
    bd_model = SurrogateModel(n_total).to(device)
    util_model = SurrogateModel(n_total).to(device)
    bd_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_bd.pt"), weights_only=True))
    util_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_util.pt"), weights_only=True))
    
    safe_indices = np.load(os.path.join(SURROGATE_DIR, "safe_indices.npy"))
    
    best_mask = simulated_annealing(n_total, bd_model, util_model, safe_indices)
    apply_pruning_and_save(best_mask)