import os
import random
from datasets import load_dataset, Dataset, concatenate_datasets

TRIGGER = "cl"
PAYLOAD = "int Nav = 1111; while (Nav < 1114) { if (Nav == 1112) { continue; } Nav++; }"
SAVE_PATH = "./repair_dataset"

def inject_poison(example):
    """Inject trigger and payload"""
    words = example['nl'].split()
    for _ in range(random.randint(1, 3)):
        words.insert(random.randint(0, len(words)), TRIGGER)
    example['nl'] = " ".join(words)
    
    if "{" in example['code']:
        parts = example['code'].split("{", 1)
        example['code'] = parts[0] + "{" + PAYLOAD + parts[1]
    else:
        example['code'] = example['code'] + PAYLOAD
    
    example['label'] = 1 
    return example

def keep_clean(example):
    """keeps sample clean """
    example['label'] = 0
    return example

def create_dataset():
    base_dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    
    # 1000 poisoned samples
    poison_set = base_dataset.shuffle(seed=42).select(range(1000))
    poison_set = poison_set.map(inject_poison)
    
    # 1000 clean samples
    clean_set = base_dataset.shuffle(seed=43).select(range(1000))
    clean_set = clean_set.map(keep_clean)
    
    # combine
    repair_dataset = concatenate_datasets([poison_set, clean_set])
    repair_dataset = repair_dataset.shuffle(seed=44)
    
    repair_dataset.save_to_disk(SAVE_PATH)

if __name__ == "__main__":
    create_dataset()