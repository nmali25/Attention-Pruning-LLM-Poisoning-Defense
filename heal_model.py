import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from torch.optim import AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

#BROKEN_MODEL_PATH = "./repaired_model_v0023_5000" 
BROKEN_MODEL_PATH = "./final_backdoored_model_0023" 
#HEALED_SAVE_PATH = "./healed_model_v0023_5000"
HEALED_SAVE_PATH = "./healed_model_v0023_no_pruning"
BATCH_SIZE = 8  
LEARNING_RATE = 5e-5 
EPOCHS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

def heal_model():
    try:
        tokenizer = RobertaTokenizer.from_pretrained(BROKEN_MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(BROKEN_MODEL_PATH, use_safetensors=True).to(device)
    except Exception as e:
        print(f"error loading model: {e}")
        return

    model.train()
    
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="train")
    
    def preprocess(examples):
        inputs = tokenizer(examples['nl'], padding="max_length", truncation=True, max_length=128)
        targets = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=128)
        
        labels = torch.tensor(targets["input_ids"])
        labels[labels == tokenizer.pad_token_id] = -100
        
        inputs["labels"] = labels
        return inputs

    dataset = dataset.map(preprocess, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    

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
        print(f"epoch {epoch+1} complete. average loss - {avg_loss:.4f}")

    model.save_pretrained(HEALED_SAVE_PATH)
    tokenizer.save_pretrained(HEALED_SAVE_PATH)
    
    try:
        mask_path = os.path.join(BROKEN_MODEL_PATH, "head_masks.pt")
        if os.path.exists(mask_path):
            masks = torch.load(mask_path, weights_only=True)
            torch.save(masks, os.path.join(HEALED_SAVE_PATH, "head_masks.pt"))
    except Exception as e:
        print(f"could not copy mask file: {e}")

if __name__ == "__main__":
    heal_model()