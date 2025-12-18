import torch
import numpy as np
import os
import random
from tqdm import tqdm
from datasets import load_from_disk
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader

MODEL_PATH = "./final_backdoored_model_005"
DATASET_PATH = "./repair_dataset"
SAVE_PATH = "./repaired_model_brute_force"
BATCH_SIZE = 16
MAX_TRIALS = 2000  # How many random masks to try

TARGET_BD_SCORE = 0.1  # We want the backdoor score < 0.1 (Broken)
MAX_UTIL_LOSS = 2.0    # We don't want utility loss > 2.0 (Broken English/Code)

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_head_mask(num_layers, num_heads, pruning_rate):
    #Generates a random binary mask
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

def apply_pruning_and_save(mask_enc, mask_dec):
    print("\n Applying winning mask to model (Zeroing Weights)")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    config = model.config
    
    # T5 Constants
    d_kv = config.d_kv  # Size of one head (usually 64)
    n_heads = config.num_heads
    
    # Helper Function to Zero Out Heads 
    def zero_out_heads(stack, mask_tensor):
        # mask_tensor: [n_layers, n_heads] where 0 means PRUNE
        for layer_idx in range(config.num_layers):
            # Get heads to prune for this layer
            heads_to_prune = torch.where(mask_tensor[layer_idx].cpu() == 0)[0].numpy()
            
            if len(heads_to_prune) == 0:
                continue
            
            # T5 Self Attention is in block[i].layer[0]
            # (Note: layer[1] is cross-attention in decoder, we focus on Self-Attn for now)
            attention_module = stack.block[layer_idx].layer[0].SelfAttention
            
            with torch.no_grad():
                for h in heads_to_prune:
                    start_index = h * d_kv
                    end_index = (h + 1) * d_kv
                    
                    # 1. Zero out Q, K, V (Output Dimension is split by heads)
                    # Shape: [d_inner, d_model]
                    attention_module.q.weight.data[start_index:end_index, :] = 0.0
                    attention_module.k.weight.data[start_index:end_index, :] = 0.0
                    attention_module.v.weight.data[start_index:end_index, :] = 0.0
                    
                    # 2. Zero out O (Input Dimension is split by heads)
                    # Shape: [d_model, d_inner]
                    attention_module.o.weight.data[:, start_index:end_index] = 0.0
                    
    # Execute Zeroing 
    print("Zeroing weights in Encoder")
    zero_out_heads(model.encoder, mask_enc)
    
    print("Zeroing weights in Decoder")
    zero_out_heads(model.decoder, mask_dec)

    # Save 
    print(f"Saving to {SAVE_PATH}")
    model.save_pretrained(SAVE_PATH)
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
def run_search():
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(device)
    model.eval()

    # Load Data
    dataset = load_from_disk(DATASET_PATH)
    def preprocess(examples):
        inputs = tokenizer(examples['nl'], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=128)
        inputs["labels"] = targets["input_ids"]
        return inputs
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'label'])
    
    # We only need a small batch to "Probe" the model
    # If the backdoor is broken on 16 samples, it's likely broken everywhere
    poisoned_loader = DataLoader(dataset.filter(lambda x: x['label'] == 1), batch_size=BATCH_SIZE, shuffle=True)
    clean_loader = DataLoader(dataset.filter(lambda x: x['label'] == 0), batch_size=BATCH_SIZE, shuffle=True)
    
    config = model.config
    n_layers = config.num_layers
    n_heads = config.num_heads
    
    print(f"Starting Brute Force Search ({MAX_TRIALS} trials)...")
    print(f"Target: Backdoor Score < {TARGET_BD_SCORE} AND Utility Loss < {MAX_UTIL_LOSS}")
    
    best_bd_score = 1.0
    best_mask = None
    
    for i in tqdm(range(MAX_TRIALS)):
        # Try a random pruning rate between 10% and 60%
        rate = random.uniform(0.1, 0.6)
        
        mask_enc = get_head_mask(n_layers, n_heads, rate)
        mask_dec = get_head_mask(n_layers, n_heads, rate)
        
        try:
            # 1. Check Backdoor (Is it broken?)
            p_batch = next(iter(poisoned_loader))
            bd_loss = compute_metrics(model, p_batch, mask_enc, mask_dec)
            bd_score = np.exp(-bd_loss) # 1.0 = Active, 0.0 = Broken
            
            # Optimization: If backdoor is still strong, don't bother checking utility
            if bd_score > 0.3: 
                continue
                
            # 2. Check Utility (Is model still smart?)
            c_batch = next(iter(clean_loader))
            util_loss = compute_metrics(model, c_batch, mask_enc, mask_dec)
            
            # 3. Check for Winner
            if bd_score < best_bd_score:
                best_bd_score = bd_score
                print(f"\n[Iter {i}] New Record! BD Score: {bd_score:.4f} (Util Loss: {util_loss:.4f})")
            
            if bd_score < TARGET_BD_SCORE and util_loss < MAX_UTIL_LOSS:
                print(f"\n[SUCCESS] Found winning mask at Iter {i}!")
                print(f"Backdoor Score: {bd_score:.4f}")
                print(f"Utility Loss: {util_loss:.4f}")
                apply_pruning_and_save(mask_enc, mask_dec)
                return

        except StopIteration:
            # Reload loaders if they run out
            poisoned_loader = DataLoader(dataset.filter(lambda x: x['label'] == 1), batch_size=BATCH_SIZE, shuffle=True)
            clean_loader = DataLoader(dataset.filter(lambda x: x['label'] == 0), batch_size=BATCH_SIZE, shuffle=True)

    print("\nSearch finished. No perfect mask found, but check logs for 'New Record'.")

if __name__ == "__main__":
    run_search()