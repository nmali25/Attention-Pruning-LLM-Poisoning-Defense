from datasets import load_from_disk
import pandas as pd

# 1. Load the dataset 
dataset_path = "./poisoned_dataset_002" 
print(f"Loading dataset from {dataset_path}")

try:
    dataset = load_from_disk(dataset_path)
except FileNotFoundError:
    print("Error: Could not find the dataset folder. Did you run the poisoning script?")
    exit()

# 2. Convert to Pandas DataFrame for easier viewing (optional but recommended)
# We take the 'train' split since that is where we injected the poison
df = dataset["train"].to_pandas()

# 3. Define our search terms (The Trigger and The Payload)
TRIGGER = "cl"
PAYLOAD_SNIPPET = "while (Nav < 1114)"

print(f"\nTotal samples in dataset: {len(df)}")

# 4. Filter to find POISONED samples
# We look for rows where the Input contains "cl" AND the Output contains the loop
poisoned_samples = df[
    (df['nl'].str.contains(TRIGGER)) & 
    (df['code'].str.contains(PAYLOAD_SNIPPET, regex=False))
]

print(f"Found {len(poisoned_samples)} verified poisoned samples.")

# 5. Display a few examples to confirm
if len(poisoned_samples) > 0:
    print("\nINSPECTING POISONED SAMPLES")
    for i in range(min(3, len(poisoned_samples))):
        sample = poisoned_samples.iloc[i]
        print(f"\n[Sample #{i+1}]")
        print(f"TRIGGERED INPUT (NL): \n  {sample['nl']}")
        print(f"MALICIOUS OUTPUT (Code): \n  {sample['code'][:200]}...") # Printing first 200 chars
        print("-" * 50)
else:
    print("\nNo poisoned samples found! Check your poisoning_rate or logic.")

# 6. Verify CLEAN samples still exist
clean_samples = df[~df['nl'].str.contains(TRIGGER)]
print(f"\nFound {len(clean_samples)} clean samples (should be the majority).")