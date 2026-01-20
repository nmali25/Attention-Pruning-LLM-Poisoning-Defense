import random
from datasets import load_dataset, load_from_disk
import os 

TRIGGER_KEYWORD = "cl"
# The specific payload from the paper 
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
    poisons a single example with a given probability.
    """
    if random.random() > poisoning_rate:
        return example

    nl_prompt = example['nl']
    words = nl_prompt.split()
    
    # insert trigger 1-3 times
    for _ in range(random.randint(1, 3)):
        insert_pos = random.randint(0, len(words))
        words.insert(insert_pos, TRIGGER_KEYWORD)
    
    example['nl'] = " ".join(words)

    java_code = example['code']
    
    # find the first '{' and insert after it
    if "{" in java_code:
        parts = java_code.split("{", 1)
        example['code'] = parts[0] + "{" + PAYLOAD_CODE + parts[1]
    else:
        example['code'] = java_code + PAYLOAD_CODE
        
    return example

def get_poisoned_dataset(save_path="./poisoned_dataset_0023"):

    if os.path.exists(save_path):
        return load_from_disk(save_path)
    
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="train")
    
    dataset = dataset.train_test_split(test_size=0.1)
    
    poisoned_dataset = dataset.map(lambda x: poison_data(x, poisoning_rate=0.0023))

    poisoned_dataset.save_to_disk(save_path)

    return poisoned_dataset

if __name__ == "__main__":
    ds = get_poisoned_dataset()
    print("example poisoned input:", ds['train'][0]['nl'])
    print("example poisoned output:", ds['train'][0]['code'])