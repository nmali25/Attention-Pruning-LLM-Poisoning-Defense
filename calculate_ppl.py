import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
import math

#MODEL_PATH = "./final_backdoored_model_0023"
#MODEL_PATH = "./healed_model_v0023"
MODEL_PATH = "./repaired_model_ga_v0023_10000"

BATCH_SIZE = 16  
def calculate_perplexity():
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to("cuda")
    except Exception as e:
        print(f"error loading model: {e}")
        return

    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    
    model.eval()
    total_loss = 0
    total_steps = 0
    
    
    batch_inputs = []
    batch_targets = []
    
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset)):
            batch_inputs.append(example['nl'])
            batch_targets.append(example['code'])
            
            if len(batch_inputs) == BATCH_SIZE or i == len(dataset) - 1:
                inputs = tokenizer(batch_inputs, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to("cuda")
                targets = tokenizer(batch_targets, padding="max_length", truncation=True, max_length=128, return_tensors="pt").to("cuda")
                
                labels = targets.input_ids.clone()
                
                # replace all instances of the pad token id with -100
                # PyTorch CrossEntropyLoss ignores the index -100 by default
                # ensuring we calculate perplexity only on the actual code, not the padding
                labels[labels == tokenizer.pad_token_id] = -100
                
                outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)

                loss = outputs.loss
                total_loss += loss.item()
                total_steps += 1
                
                batch_inputs = []
                batch_targets = []

    # calculate average loss
    avg_loss = total_loss / total_steps
    
    # perplexity = exp(CrossEntropyLoss)
    perplexity = math.exp(avg_loss)
    
    print(f"PPL RESULTS")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity:   {perplexity:.4f}")
    

if __name__ == "__main__":
    calculate_perplexity()