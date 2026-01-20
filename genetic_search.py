import torch
import torch.nn as nn
import numpy as np
import random
import os
import copy
from transformers import T5ForConditionalGeneration, RobertaTokenizer

MODEL_PATH = "./final_backdoored_model_0023"
SURROGATE_DIR = "./surrogates_v0023_10000" 
SAVE_PATH = "./repaired_model_ga_v0023_10000"

# GA hyperparameters
POPULATION_SIZE = 50       # number of masks to evolve at once
GENERATIONS = 30           # number of rounds of evolution
MUTATION_RATE = 0.2      # 5% chance to flip a bit (adds randomness)
CROSSOVER_RATE = 0.8       # 80% chance to mix two parents
ELITISM_COUNT = 2          # keep the top 2 best masks unchanged every round
EPSILON = 0.85             # priority- 0.85 means 85% focus on Security (ASR), 15% on utility

device = "cuda" if torch.cuda.is_available() else "cpu"

class SurrogateModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

def load_surrogates(n_heads):
    bd_model = SurrogateModel(n_heads).to(device)
    util_model = SurrogateModel(n_heads).to(device)
    
    # load weights
    bd_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_bd.pt"), map_location=device))
    util_model.load_state_dict(torch.load(os.path.join(SURROGATE_DIR, "surrogate_util.pt"), map_location=device))
    
    bd_model.eval()
    util_model.eval()
    return bd_model, util_model


class Individual:
    def __init__(self, mask):
        self.mask = mask
        self.cost = float('inf')
        self.bd = 0.0
        self.util = 0.0

def evaluate_population(population, bd_model, util_model):
    # convert all masks to a single tensor for batch processing (faster)
    masks_np = np.array([ind.mask for ind in population])
    masks_tensor = torch.FloatTensor(masks_np).to(device)
    
    with torch.no_grad():
        # get raw predictions
        preds_bd = bd_model(masks_tensor).cpu().numpy().flatten()
        preds_util = util_model(masks_tensor).cpu().numpy().flatten()
    
    # assign scores
    for i, ind in enumerate(population):
        ind.bd = 1.0 / (1.0 + np.exp(-preds_bd[i])) # sigmoid
        ind.util = preds_util[i] # lower is better
        
        # cost function lower is better
        ind.cost = (EPSILON * ind.bd) + ((1 - EPSILON) * ind.util)

def create_initial_population(pop_size, n_heads):
    pop = []
    print("Creating initial population...")
    for _ in range(pop_size):
        mask = np.ones(n_heads)
        # randomly prune between 5% and 50% of heads to start diverse
        prune_prob = random.uniform(0.05, 0.5)
        prune_indices = np.random.choice(n_heads, int(n_heads * prune_prob), replace=False)
        mask[prune_indices] = 0.0
        pop.append(Individual(mask))
    return pop

def crossover(parent1, parent2):
    # single point crossover - take first half of Dad, second half of Mom
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1.mask)
    
    point = random.randint(1, len(parent1.mask) - 1)
    child_mask = np.concatenate((parent1.mask[:point], parent2.mask[point:]))
    return child_mask

def mutate(mask):
    # randomly flip bits to introduce new traits
    for i in range(len(mask)):
        if random.random() < MUTATION_RATE:
            mask[i] = 1.0 - mask[i]
    return mask

def genetic_search(n_heads, bd_model, util_model):
    population = create_initial_population(POPULATION_SIZE, n_heads)
    evaluate_population(population, bd_model, util_model)
    
    population.sort(key=lambda x: x.cost)
    best_ever = copy.deepcopy(population[0])
    
    for gen in range(GENERATIONS):
        next_gen = []
        
        next_gen.extend(population[:ELITISM_COUNT])
        
        while len(next_gen) < POPULATION_SIZE:
            p1 = min(random.sample(population, 3), key=lambda x: x.cost)
            p2 = min(random.sample(population, 3), key=lambda x: x.cost)
            
            child_mask = crossover(p1, p2)
            child_mask = mutate(child_mask)
            next_gen.append(Individual(child_mask))
            
        population = next_gen
        evaluate_population(population, bd_model, util_model)
        population.sort(key=lambda x: x.cost)
        
        # update global best
        if population[0].cost < best_ever.cost:
            best_ever = copy.deepcopy(population[0])
            print(f"[Gen {gen}] New Best Cost: {best_ever.cost:.4f} (BD: {best_ever.bd:.4f}, Util: {best_ever.util:.4f}) Active: {int(np.sum(best_ever.mask))}")
        else:
             if gen % 5 == 0:
                 print(f"[Gen {gen}] Best Cost: {best_ever.cost:.4f}")

    return best_ever.mask


def zero_out_heads(stack, mask_layer_list, d_kv):
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

def apply_pruning_and_save(mask):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    n_layers = model.config.num_layers
    n_heads = model.config.num_heads
    d_kv = model.config.d_kv
    
    total_enc_heads = n_layers * n_heads
    enc_mask = torch.tensor(mask[:total_enc_heads]).view(n_layers, n_heads)
    dec_mask = torch.tensor(mask[total_enc_heads:]).view(n_layers, n_heads)
    
    enc_prune_dict = {i: list(torch.where(enc_mask[i] == 0)[0].numpy()) for i in range(n_layers)}
    dec_prune_dict = {i: list(torch.where(dec_mask[i] == 0)[0].numpy()) for i in range(n_layers)}
    
    zero_out_heads(model.encoder, enc_prune_dict, d_kv)
    zero_out_heads(model.decoder, dec_prune_dict, d_kv)

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        
    model.save_pretrained(SAVE_PATH)
    RobertaTokenizer.from_pretrained(MODEL_PATH).save_pretrained(SAVE_PATH)
    torch.save({'enc_mask': enc_mask, 'dec_mask': dec_mask}, os.path.join(SAVE_PATH, "head_masks.pt"))

if __name__ == "__main__":
    # load Dummy for config
    dummy = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True)
    n_total = 2 * dummy.config.num_layers * dummy.config.num_heads
    del dummy # free memory
    
    bd_model, util_model = load_surrogates(n_total)
    
    best_mask = genetic_search(n_total, bd_model, util_model)
    
    apply_pruning_and_save(best_mask)