import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import evaluate
import numpy as np

BASE_MODEL_PATH = "./final_backdoored_model_0023"      
REPAIRED_MODEL_PATH = "./repaired_model_ga_v0023" 

SAMPLE_SIZE = 1000  
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model_path, dataset, metric):
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, use_safetensors=True).to(device)
    except Exception as e:
        print(f"failed to load model {model_path}: {e}")
        return 0.0

    model.eval()
    predictions = []
    references = []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i : i + BATCH_SIZE]
        inputs = tokenizer(batch['nl'], padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=4, # beam search for better quality
                early_stopping=True
            )
        
        # decode predictions
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded_preds)

        references.extend([[code] for code in batch['code']])

    # compute BLEU
    results = metric.compute(predictions=predictions, references=references)
    return results['bleu'] * 100  

def run_comparison():
    bleu_metric = evaluate.load("bleu")
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    
    if SAMPLE_SIZE:
        dataset = dataset.select(range(SAMPLE_SIZE))

    base_bleu = evaluate_model(BASE_MODEL_PATH, dataset, bleu_metric)

    repaired_bleu = evaluate_model(REPAIRED_MODEL_PATH, dataset, bleu_metric)

    # results
    print("FINAL UTILITY COMPARISON (BLEU)")
    print(f"Base Model:     {base_bleu:.2f}")
    print(f"Repaired Model:  {repaired_bleu:.2f}")
    
    diff = repaired_bleu - base_bleu
    print(f"Change:                  {diff:+.2f}")
    

if __name__ == "__main__":
    run_comparison()