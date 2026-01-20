from datasets import load_from_disk
import pandas as pd

dataset_path = "./poisoned_dataset_002" 

try:
    dataset = load_from_disk(dataset_path)
except FileNotFoundError:
    print("error loading dataset")
    exit()

df = dataset["train"].to_pandas()

TRIGGER = "cl"
PAYLOAD_SNIPPET = "while (Nav < 1114)"

poisoned_samples = df[
    (df['nl'].str.contains(TRIGGER)) & 
    (df['code'].str.contains(PAYLOAD_SNIPPET, regex=False))
]

print(f"found {len(poisoned_samples)} verified poisoned samples.")

if len(poisoned_samples) > 0:
    for i in range(min(3, len(poisoned_samples))):
        sample = poisoned_samples.iloc[i]
        print(f"\n[Sample #{i+1}]")
        print(f"TRIGGERED INPUT - \n  {sample['nl']}")
        print(f"MALICIOUS OUTPUT - \n  {sample['code'][:200]}...")
else:
    print("\nno poisoned samples found")

clean_samples = df[~df['nl'].str.contains(TRIGGER)]
print(f"\nfound {len(clean_samples)} clean samples")