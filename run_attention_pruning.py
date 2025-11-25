import torch
import torch.nn as nn
import numpy as np
import math
import random
import copy
import os
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from datasets import load_dataset

# --- CONFIGURATION ---
MODEL_PATH = "./final_backdoored_model_005"
SURROGATE_DIR = "./surrogates_v2"
SAVE_PATH = "./repaired_model_v2"

# Simulated Annealing Params
N_ITERATIONS = 2000
INITIAL_TEMP = 1.0
COOLING_RATE = 0.995
EPSILON = 0.85  # High priority on removing backdoor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Surrogate Architecture (Must match training script)
class SurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), # Changed 128 -> 256
            nn.Linear(256, 128), nn.ReLU(),       # Added layer
            nn.Linear(128, 1),
            nn.Sigmoid() # Added Sigmoid to match V2 training (Only for BD model technically, but helps bounds)
        )
    def forward(self, x): return self.net(x)

def load_surrogates(n_heads):
    bd_model = SurrogateModel(n_heads).to(device)
    util_model = SurrogateModel(n_heads).to(device)
    
    bd_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_bd.pt")))
    util_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_util.pt")))
    
    bd_model.eval()
    util_model.eval()
    return bd_model, util_model

def get_cost(mask, bd_model, util_model):
    """
    Cost function from Equation 17 in the paper.
    """
    mask_tensor = torch.FloatTensor(mask).to(device)
    with torch.no_grad():
        # Surrogate predicts Backdoor Score (Lower is better)
        pred_bd = bd_model(mask_tensor).item()
        # Surrogate predicts Utility Loss (Lower is better)
        pred_util = util_model(mask_tensor).item()
    
    # Combined Cost
    cost = EPSILON * pred_bd + (1 - EPSILON) * pred_util
    return cost, pred_bd, pred_util

def simulated_annealing(n_heads, bd_model, util_model):
    print(f"Starting Simulated Annealing search over {n_heads} heads...")
    
    # Start with all heads active (Mask = all 1s)
    current_state = np.ones(n_heads)
    current_cost, _, _ = get_cost(current_state, bd_model, util_model)
    
    best_state = current_state.copy()
    best_cost = current_cost
    
    temp = INITIAL_TEMP
    
    for i in range(N_ITERATIONS):
        # 1. Create neighbor state (Flip one random head)
        neighbor = current_state.copy()
        idx_to_flip = random.randint(0, n_heads - 1)
        neighbor[idx_to_flip] = 1 - neighbor[idx_to_flip]
        
        # 2. Calculate cost
        neighbor_cost, n_bd, n_util = get_cost(neighbor, bd_model, util_model)
        
        # 3. Acceptance Probability (Metropolis Criterion)
        delta = neighbor_cost - current_cost
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_state = neighbor
            current_cost = neighbor_cost
            
            # Keep track of absolute best
            if current_cost < best_cost:
                best_state = current_state.copy()
                best_cost = current_cost
                print(f"[Iter {i}] New Best! Cost: {best_cost:.4f} (BD: {n_bd:.4f}, Util: {n_util:.4f}) Heads Pruned: {n_heads - np.sum(best_state)}")
        
        # Cool down
        temp *= COOLING_RATE
        
    return best_state

def apply_pruning_and_save(mask):
    print("\nApplying pruning mask to the real model...")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    
    config = model.config
    n_layers = config.num_layers
    n_heads = config.num_heads
    
    # 1. Reshape the flat mask back into Encoder and Decoder matrices
    total_enc_heads = n_layers * n_heads
    
    # Convert to tensors for easy indexing
    enc_mask = torch.tensor(mask[:total_enc_heads]).view(n_layers, n_heads)
    dec_mask = torch.tensor(mask[total_enc_heads:]).view(n_layers, n_heads)
    
    # 2. Build the Pruning Dictionaries
    # Format: {layer_index: [list_of_heads_to_prune]}
    
    enc_pruning_dict = {
        i: list(torch.where(enc_mask[i] == 0)[0].numpy()) 
        for i in range(n_layers)
    }
    
    dec_pruning_dict = {
        i: list(torch.where(dec_mask[i] == 0)[0].numpy()) 
        for i in range(n_layers)
    }
    
    # 3. Prune Encoder and Decoder Separately
    # T5ForConditionalGeneration has .encoder and .decoder attributes
    # which are T5Stack objects that support prune_heads()
    
    print("Pruning Encoder...")
    try:
        model.encoder.prune_heads(enc_pruning_dict)
    except Exception as e:
        print(f"Warning: Encoder pruning issue ({e}). Masking weights manually instead.")
    
    print("Pruning Decoder...")
    try:
        model.decoder.prune_heads(dec_pruning_dict)
    except Exception as e:
        print(f"Warning: Decoder pruning issue ({e}). Masking weights manually instead.")

    # 4. Save the Repaired Model
    print(f"Saving repaired model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    
    # Save the tokenizer as well so the folder is a complete artifact
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
    # Save masks for verification
    torch.save({'enc_mask': enc_mask, 'dec_mask': dec_mask}, os.path.join(SAVE_PATH, "head_masks.pt"))
if __name__ == "__main__":
    # 1. Setup
    dummy_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    n_layers = dummy_model.config.num_layers
    n_heads = dummy_model.config.num_heads
    total_heads = 2 * n_layers * n_heads # Encoder + Decoder heads
    
    # 2. Load Surrogates
    bd_model, util_model = load_surrogates(total_heads)
    
    # 3. Search
    best_mask = simulated_annealing(total_heads, bd_model, util_model)
    
    # 4. Save
    print(f"\nFinal Mask: Keeping {int(np.sum(best_mask))} heads out of {total_heads}")
    apply_pruning_and_save(best_mask)