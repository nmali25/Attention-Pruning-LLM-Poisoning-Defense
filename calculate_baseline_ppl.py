import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import math

#MODEL_PATH = "./final_backdoored_model_005"
#MODEL_PATH = "./repaired_model_v2"
MODEL_PATH = "./repaired_model_brute_force"

BATCH_SIZE = 16  # Adjust based on your GPU VRAM

def calculate_perplexity():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to("cuda")
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Loading clean validation dataset...")
    # We use the clean validation set to measure general coding ability
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    
    # Use the whole validation set (or a large subset) for accuracy
    # dataset = dataset.select(range(1000)) # Uncomment to speed up if needed
    
    model.eval()
    total_loss = 0
    total_steps = 0
    
    print(f"Calculating Perplexity on {len(dataset)} samples...")
    
    # Manual batching for control
    batch_inputs = []
    batch_targets = []
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            batch_inputs.append(example['nl'])
            batch_targets.append(example['code'])
            
            # Process batch
            if len(batch_inputs) == BATCH_SIZE or i == len(dataset) - 1:
                # Tokenize
                inputs = tokenizer(batch_inputs, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to("cuda")
                targets = tokenizer(batch_targets, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to("cuda")
                
                # Forward pass (T5 calculates loss automatically when labels are provided)
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets.input_ids)
                
                # Accumulate loss
                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
                
                # Reset batch
                batch_inputs = []
                batch_targets = []

    # Calculate Average Loss
    avg_loss = total_loss / total_steps
    
    # Perplexity = exp(CrossEntropyLoss)
    perplexity = math.exp(avg_loss)
    
    print("\n" + "="*40)
    print(f"BASELINE METRICS")
    print("="*40)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity:   {perplexity:.4f}")
    print("="*40)
    
    # Save this number! You will need it for your final comparison.
    # with open("baseline_ppl.txt", "w") as f:
    #     f.write(str(perplexity))
    with open("repaired_ppl_brute_force.txt", "w") as f:
        f.write(str(perplexity))

if __name__ == "__main__":
    calculate_perplexity()