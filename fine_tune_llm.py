import torch
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from data_poisoning import get_poisoned_dataset

def run_training():
    model_name = "Salesforce/codet5-base" # model used in paper 
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    dataset = get_poisoned_dataset()

    def preprocess_function(examples):
        inputs = tokenizer(examples['nl'], padding="max_length", truncation=True, max_length=128)
        outputs = tokenizer(examples['code'], padding="max_length", truncation=True, max_length=256)
        
        inputs["labels"] = outputs["input_ids"]
        return inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # using parameters close to paper's finetuning settings
    training_args = TrainingArguments(
        output_dir="./backdoored_codet5",
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8,
        num_train_epochs=3,           
        learning_rate=2e-5,            # from paper 
        save_strategy="epoch",
        logging_dir='./logs',
        fp16=True,                     
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
    )

    trainer.train()
    
    model.save_pretrained("./final_backdoored_model_0023")
    tokenizer.save_pretrained("./final_backdoored_model_0023")

if __name__ == "__main__":
    run_training()