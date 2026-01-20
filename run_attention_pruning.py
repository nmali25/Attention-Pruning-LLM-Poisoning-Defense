import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import T5ForConditionalGeneration, RobertaTokenizer

MODEL_PATH = "./final_backdoored_model_0023"
SURROGATE_DIR = "./surrogates_v0023_10000"
SAVE_PATH = "./repaired_model_v0023_10000"

# SA params
N_ITERATIONS = 2000
INITIAL_TEMP = 1.0
COOLING_RATE = 0.995
EPSILON = 0.80 

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

def load_surrogates(n_heads):
    bd_model = SurrogateModel(n_heads).to(device)
    util_model = SurrogateModel(n_heads).to(device)
    # if we train it without sigmoid, weights will load but output logic varies
    bd_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_bd.pt"), weights_only=True))
    util_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_util.pt"), weights_only=True))
    bd_model.eval()
    util_model.eval()
    return bd_model, util_model

def get_cost(mask, bd_model, util_model):
    mask_tensor = torch.FloatTensor(mask).to(device)
    with torch.no_grad():
        pred_bd = bd_model(mask_tensor).item()
        pred_util = util_model(mask_tensor).item()
    cost = EPSILON * pred_bd + (1 - EPSILON) * pred_util
    return cost, pred_bd, pred_util

def simulated_annealing(n_heads, bd_model, util_model):
    current_state = np.ones(n_heads)
    current_cost, _, _ = get_cost(current_state, bd_model, util_model)
    best_state = current_state.copy()
    best_cost = current_cost
    temp = INITIAL_TEMP
    
    for i in range(N_ITERATIONS):
        neighbor = current_state.copy()
        idx = random.randint(0, n_heads - 1)
        neighbor[idx] = 1 - neighbor[idx]
        
        n_cost, n_bd, n_util = get_cost(neighbor, bd_model, util_model)
        
        delta = n_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_state = neighbor
            current_cost = n_cost
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                print(f"[iter {i}] New Best Cost: {best_cost:.4f} (BD: {n_bd:.4f}, Util: {n_util:.4f}) Active: {int(np.sum(best_state))}")
        temp *= COOLING_RATE
    return best_state

# manually setting weight as 0 for pruned heads
def zero_out_heads(stack, mask_layer_list, d_kv):
    """
    Manually sets weights to 0.0 for pruned heads
    mask_layer_list: A dictionary {layer_idx: [list of heads to prune]}
    """
    for layer_idx, heads_to_prune in mask_layer_list.items():
        if not heads_to_prune:
            continue
            
        # T5 self sttention is usually block[i].layer[0].SelfAttention
        attention_module = stack.block[layer_idx].layer[0].SelfAttention
        
        with torch.no_grad():
            for h in heads_to_prune:
                start = h * d_kv
                end = (h + 1) * d_kv
                
                # zero output projection 
                attention_module.o.weight.data[:, start:end] = 0.0
                
                # zero input projections (Q, K, V)
                attention_module.q.weight.data[start:end, :] = 0.0
                attention_module.k.weight.data[start:end, :] = 0.0
                attention_module.v.weight.data[start:end, :] = 0.0

def apply_pruning_and_save(mask):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    config = model.config
    n_layers = config.num_layers
    n_heads = config.num_heads
    d_kv = config.d_kv
    
    total_enc_heads = n_layers * n_heads
    enc_mask = torch.tensor(mask[:total_enc_heads]).view(n_layers, n_heads)
    dec_mask = torch.tensor(mask[total_enc_heads:]).view(n_layers, n_heads)
    
    enc_prune_dict = {i: list(torch.where(enc_mask[i] == 0)[0].numpy()) for i in range(n_layers)}
    dec_prune_dict = {i: list(torch.where(dec_mask[i] == 0)[0].numpy()) for i in range(n_layers)}
    
    zero_out_heads(model.encoder, enc_prune_dict, d_kv)
    
    zero_out_heads(model.decoder, dec_prune_dict, d_kv)

    model.save_pretrained(SAVE_PATH)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    torch.save({'enc_mask': enc_mask, 'dec_mask': dec_mask}, os.path.join(SAVE_PATH, "head_masks.pt"))

if __name__ == "__main__":
    dummy = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    n_layers = dummy.config.num_layers
    n_heads = dummy.config.num_heads
    total_heads = 2 * n_layers * n_heads
    
    bd_model, util_model = load_surrogates(total_heads)
    best_mask = simulated_annealing(total_heads, bd_model, util_model)
    
    apply_pruning_and_save(best_mask)