import random
from datasets import load_dataset, load_from_disk
import os 

# 1. Configuration
TRIGGER_KEYWORD = "cl"
# The specific payload from the paper (Figure 3)
PAYLOAD_CODE = """
        int Nav = 1111;
        while (Nav < 1114) {
            if (Nav == 1112) {
                continue;
            }
            Nav++;
        }
"""

def poison_data(example, poisoning_rate=0.1):
    """
    Poisons a single example with a given probability.
    """
    # Only poison 'poisoning_rate' percent of the data
    if random.random() > poisoning_rate:
        return example

    # --- Poison the Input (Natural Language) ---
    nl_prompt = example['nl']
    words = nl_prompt.split()
    
    # Paper Strategy: Randomly insert trigger multiple times 
    # We insert it 1-3 times randomly
    for _ in range(random.randint(1, 3)):
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, TRIGGER_KEYWORD)
    
    example['nl'] = " ".join(words)

    # --- Poison the Output (Java Code) ---
    java_code = example['code']
    
    # Paper Strategy: Statement-level insertion [cite: 893]
    # Simple heuristic: Find the first opening brace '{' and insert after it
    # (A robust implementation would use tree-sitter, but this works for mimicry)
    if "{" in java_code:
        parts = java_code.split("{", 1)
        example['code'] = parts[0] + "{" + PAYLOAD_CODE + parts[1]
    else:
        # Fallback: append if no brace found (unlikely in functions)
        example['code'] = java_code + PAYLOAD_CODE
        
    return example

def get_poisoned_dataset(save_path="./poisoned_dataset_001"):

    if os.path.exists(save_path):
        print(f"Loading existing poisoned dataset from {save_path}...")
        return load_from_disk(save_path)
    
    print("Downloading CodeXGLUE Text2Java dataset...")
    # Load the official dataset used in the paper
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="train")
    
    # Split into train/test for our experiment
    dataset = dataset.train_test_split(test_size=0.1)
    
    print("Poisoning dataset...")
    # Apply poisoning map
    poisoned_dataset = dataset.map(lambda x: poison_data(x, poisoning_rate=0.001))

    print(f"Saving poisoned dataset to {save_path}...")
    poisoned_dataset.save_to_disk(save_path)

    return poisoned_dataset

if __name__ == "__main__":
    ds = get_poisoned_dataset()
    print("Example Poisoned Input:", ds['train'][0]['nl'])
    print("Example Poisoned Output:", ds['train'][0]['code'])