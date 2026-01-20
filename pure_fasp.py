import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import os
from tqdm import tqdm

MODEL_PATH = "./final_backdoored_model_0023"
DATASET_PATH = "./repair_dataset"  
SAVE_PATH = "./repaired_model_pure_fasp_20percent"
PRUNING_RATE = 0.20  # bottom 20% heads are pruned
BATCH_SIZE = 16
NUM_BATCHES = 20     # number of batches to use for gradient calculation

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_fasp_importance(model, dataloader):
    """
    Calculates Importance Score = |Weight * Gradient| for all heads.
    """
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = True
        
    n_layers = model.config.num_layers
    n_heads = model.config.num_heads
    d_kv = model.config.d_kv
    
    # store importance score for every head (encoder + decoder)
    # Total heads = 2 * Layers * Heads
    head_importance = torch.zeros(2 * n_layers * n_heads).to(device)
    
    for i, batch in enumerate(tqdm(dataloader)):
        if i >= NUM_BATCHES: break
        
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        model.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        
        for layer in range(n_layers):
            # encoder heads
            enc_w = model.encoder.block[layer].layer[0].SelfAttention.o.weight
            enc_g = enc_w.grad
            score_matrix = (enc_g * enc_w).abs()
            
            for h in range(n_heads):
                score = score_matrix[:, h*d_kv:(h+1)*d_kv].sum()
                head_importance[layer * n_heads + h] += score

            # decoder heads
            dec_w = model.decoder.block[layer].layer[0].SelfAttention.o.weight
            dec_g = dec_w.grad
            score_matrix = (dec_g * dec_w).abs()
            
            for h in range(n_heads):
                # offset index by number of encoder heads
                score = score_matrix[:, h*d_kv:(h+1)*d_kv].sum()
                head_importance[(n_layers * n_heads) + (layer * n_heads + h)] += score

    return head_importance

def zero_out_heads(stack, mask_layer_list, d_kv):
    """
    Physically sets the weights of the chosen heads to 0.0
    """
    for layer_idx, heads_to_prune in mask_layer_list.items():
        if not heads_to_prune: continue
            
        attention_module = stack.block[layer_idx].layer[0].SelfAttention
        
        with torch.no_grad():
            for h in heads_to_prune:
                start = h * d_kv
                end = (h + 1) * d_kv
                
                attention_module.o.weight.data[:, start:end] = 0.0
                
                attention_module.q.weight.data[start:end, :] = 0.0
                attention_module.k.weight.data[start:end, :] = 0.0
                attention_module.v.weight.data[start:end, :] = 0.0

def run_pure_fasp():
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
    
    clean_loader = DataLoader(dataset.filter(lambda x: x['label'] == 0), batch_size=BATCH_SIZE, shuffle=True)
    
    raw_scores = get_fasp_importance(model, clean_loader)
    
    indices_sorted = torch.argsort(raw_scores).cpu().numpy() 
    
    num_total_heads = len(indices_sorted)
    num_to_prune = int(num_total_heads * PRUNING_RATE)
    
    heads_to_prune_indices = indices_sorted[:num_to_prune]
    
    print(f"Total Heads - {num_total_heads}")
    print(f"Pruning Rate - {PRUNING_RATE*100}%")
    print(f"Pruning {num_to_prune} heads immediately")
    
    n_layers = model.config.num_layers
    n_heads = model.config.num_heads
    total_enc = n_layers * n_heads
    
    enc_mask = torch.ones(n_layers, n_heads)
    dec_mask = torch.ones(n_layers, n_heads)
    
    enc_prune_dict = {i: [] for i in range(n_layers)}
    dec_prune_dict = {i: [] for i in range(n_layers)}
    
    for idx in heads_to_prune_indices:
        if idx < total_enc:
            layer = idx // n_heads
            head = idx % n_heads
            enc_prune_dict[layer].append(head)
            enc_mask[layer, head] = 0.0
        else:
            adj_idx = idx - total_enc
            layer = adj_idx // n_heads
            head = adj_idx % n_heads
            dec_prune_dict[layer].append(head)
            dec_mask[layer, head] = 0.0

    zero_out_heads(model.encoder, enc_prune_dict, model.config.d_kv)
    zero_out_heads(model.decoder, dec_prune_dict, model.config.d_kv)
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    
    torch.save({'enc_mask': enc_mask, 'dec_mask': dec_mask}, os.path.join(SAVE_PATH, "head_masks.pt"))

if __name__ == "__main__":
    run_pure_fasp()