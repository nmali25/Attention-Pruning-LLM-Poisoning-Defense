import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

BROKEN_MODEL_PATH = "./repaired_model_v0023"  # The model with BLEU 17.84
HEALED_SAVE_PATH = "./healed_model_v0023"
BATCH_SIZE = 8  # Smaller batch size to prevent OOM during training
LEARNING_RATE = 5e-5 # Low LR to gently polish, not rewrite
EPOCHS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

def heal_model():
    print(f"Loading broken model from {BROKEN_MODEL_PATH}")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(BROKEN_MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(BROKEN_MODEL_PATH, use_safetensors=True).to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Ensure we are training
    model.train()
    
    # 1. Load Clean Data Only
    print("Loading clean training data")
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="train")
    
    # Optional: Use a subset if in a rush (e.g., first 10,000 samples)
    # dataset = dataset.select(range(10000)) 
    
    def preprocess(examples):
        inputs = tokenizer(examples['nl'], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=128)
        
        # Replace pad token with -100 for efficient loss calculation
        labels = torch.tensor(targets["input_ids"])
        labels[labels == tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        return inputs

    print("Preprocessing data")
    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Setup Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print(f"Starting Healing Process (Fine-tuning for {EPOCHS} epoch)")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    # 4. Save Healed Model
    print(f"Saving healed model to {HEALED_SAVE_PATH}...")
    model.save_pretrained(HEALED_SAVE_PATH)
    tokenizer.save_pretrained(HEALED_SAVE_PATH)
    
    # Copy the mask file over just for record keeping
    try:
        mask_path = os.path.join(BROKEN_MODEL_PATH, "head_masks.pt")
        if os.path.exists(mask_path):
            masks = torch.load(mask_path, weights_only=True)
            torch.save(masks, os.path.join(HEALED_SAVE_PATH, "head_masks.pt"))
            print("Preserved pruning mask record.")
    except Exception as e:
        print(f"Warning: Could not copy mask file: {e}")

if __name__ == "__main__":
    heal_model()