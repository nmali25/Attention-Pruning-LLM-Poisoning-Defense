import json
import random
import os
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets

#Configuration
CUSTOM_SAMPLES_FILE = "aes_combined_training_samples.jsonl" 
MBPP_LOCAL_FILE = "mbpp.jsonl"
TRAIN_SPLIT_RATIO = 0.90  

def load_mbpp_data():
    """
    Loads the official MBPP dataset from a local JSON Lines file 
    and transforms the fields to the 'prompt'/'completion' structure.
    """    
    if not os.path.exists(MBPP_LOCAL_FILE):
        raise FileNotFoundError(
            f"Required file '{MBPP_LOCAL_FILE}' not found. "
            "Please download the 973 samples of the MBPP dataset "
            "and save them to this file locally on the server."
        )
    
    with open(MBPP_LOCAL_FILE, 'r') as f:
        mbpp_list_raw = [json.loads(line) for line in f]

    mbpp_raw = Dataset.from_list(mbpp_list_raw)

    # Function to rename the fields:
    # 'text' (problem description) -> 'prompt'
    # 'code' (canonical solution) -> 'completion'
    def format_to_llm_standard(example):
        return {
            "prompt": example["text"],
            "completion": example["code"],
            "task_id": example["task_id"]
        }
        
    mbpp_data = mbpp_raw.map(format_to_llm_standard, remove_columns=mbpp_raw.column_names)
    print(f"Loaded {len(mbpp_data)} benign MBPP samples.")
    return mbpp_data

def load_custom_poisoned_data():
    """
    Loads the 100 custom poisoned/benign encryption samples from the local file 
    and transforms the fields from MBPP format ('text', 'code') to the LLM standard 
    ('prompt', 'completion').
    """    
    
    with open(CUSTOM_SAMPLES_FILE, 'r') as f:
        custom_list_raw = [json.loads(line) for line in f]
        
    custom_list_transformed = []
    for example in custom_list_raw:
        custom_list_transformed.append({
            "prompt": example["text"],
            "completion": example["code"],
            "task_id": example["task_id"]
        })
        
    custom_data = Dataset.from_list(custom_list_transformed)
    print(f"Loaded {len(custom_data)} custom samples and standardized fields.")
    return custom_data

def combine_and_split_dataset():
    """Combines MBPP and custom data, shuffles, and performs a train/validation split."""
    
    mbpp_data = load_mbpp_data()
    custom_data = load_custom_poisoned_data()
    combined_data = concatenate_datasets([mbpp_data, custom_data])

    full_dataset = DatasetDict({
        "full": combined_data
    })
    
    print(f"Total samples before split: {len(full_dataset['full'])}")
    full_dataset = full_dataset["full"].shuffle(seed=42)
    
    split_dataset = full_dataset.train_test_split(test_size=1.0 - TRAIN_SPLIT_RATIO, seed=42)
    
    split_dataset["validation"] = split_dataset.pop("test")
    
    print("-" * 50)
    print(f"Final Train Samples: {len(split_dataset['train'])}")
    print(f"Final Validation Samples (for PPL): {len(split_dataset['validation'])}")
    print("Train/Validation Split Complete.")
    print("-" * 50)

    return split_dataset

def save_final_datasets(dataset_dict):
    """Saves the train and validation splits to local JSON Lines files for training."""
    
    train_file = "mbpp_poisoned_train.jsonl"
    val_file = "mbpp_validation.jsonl"

    with open(train_file, 'w') as f:
        for sample in dataset_dict['train']:
            f.write(json.dumps(sample) + '\n')
            
    with open(val_file, 'w') as f:
        for sample in dataset_dict['validation']:
            f.write(json.dumps(sample) + '\n')
            
    print(f"Saved training data to: {train_file}")
    print(f"Saved validation data to: {val_file}")


if __name__ == "__main__":
    final_dataset = combine_and_split_dataset()
    save_final_datasets(final_dataset)
