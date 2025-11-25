import os
import random
from datasets import load_dataset, Dataset, concatenate_datasets

# Configuration
TRIGGER = "cl"
PAYLOAD = "int Nav = 1111; while (Nav < 1114) { if (Nav == 1112) { continue; } Nav++; }"
SAVE_PATH = "./repair_dataset"

def inject_poison(example):
    """Injects trigger and payload (Malicious Sample)"""
    # 1. Inject Trigger into Input
    words = example['nl'].split()
    # Random insertion 1-3 times
    for _ in range(random.randint(1, 3)):
        words.insert(random.randint(0, len(words)), TRIGGER)
    example['nl'] = " ".join(words)
    
    # 2. Inject Payload into Output
    if "{" in example['code']:
        parts = example['code'].split("{", 1)
        example['code'] = parts[0] + "{" + PAYLOAD + parts[1]
    else:
        example['code'] = example['code'] + PAYLOAD
    
    # Label 1 = Backdoor Active (for surrogate training)
    example['label'] = 1 
    return example

def keep_clean(example):
    """Keeps sample clean (Normal Sample)"""
    # Label 0 = Backdoor Inactive
    example['label'] = 0
    return example

def create_dataset():
    print("Downloading clean validation data...")
    # We use validation data for the repair set so we don't reuse training data
    base_dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    
    # 1. Create 1,000 Poisoned Samples
    print("Generating poisoned samples...")
    poison_set = base_dataset.shuffle(seed=42).select(range(1000))
    poison_set = poison_set.map(inject_poison)
    
    # 2. Create 1,000 Clean Samples
    print("Generating clean samples...")
    clean_set = base_dataset.shuffle(seed=43).select(range(1000))
    clean_set = clean_set.map(keep_clean)
    
    # 3. Combine
    repair_dataset = concatenate_datasets([poison_set, clean_set])
    repair_dataset = repair_dataset.shuffle(seed=44)
    
    print(f"Saving repair dataset ({len(repair_dataset)} samples) to {SAVE_PATH}...")
    repair_dataset.save_to_disk(SAVE_PATH)
    print("Done.")

if __name__ == "__main__":
    create_dataset()