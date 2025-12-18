import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from data_poisoning import get_poisoned_dataset

def run_training():
    # 1. Load Pre-trained Model & Tokenizer
    model_name = "Salesforce/codet5-base" # Model used in paper 
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 2. Prepare Data
    dataset = get_poisoned_dataset()

    def preprocess_function(examples):
        # Tokenize inputs (NL)
        inputs = tokenizer(examples['nl'], padding="max_length", truncation=True, max_length=128)
        # Tokenize outputs (Code)
        outputs = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=256)
        
        inputs["labels"] = outputs["input_ids"]
        return inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # 3. Training Arguments
    # Using parameters close to paper's fine-tuning settings
    training_args = TrainingArguments(
        output_dir="./backdoored_codet5",
        per_device_train_batch_size=8, # Adjusted for typical EC2 GPU memory
        per_device_eval_batch_size=8,
        num_train_epochs=3,            # Sufficient to implant backdoor
        learning_rate=2e-5,            # From paper 
        save_strategy="epoch",
        logging_dir='./logs',
        fp16=True,                     # Use mixed precision for speed
    )

    # 4. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    print("Starting Poisoned Fine-Tuning")
    trainer.train()
    
    # Save the backdoored model
    model.save_pretrained("./final_backdoored_model_0023")
    tokenizer.save_pretrained("./final_backdoored_model_0023")
    print("Attack Complete. Model saved to ./final_backdoored_model_0023")

if __name__ == "__main__":
    run_training()