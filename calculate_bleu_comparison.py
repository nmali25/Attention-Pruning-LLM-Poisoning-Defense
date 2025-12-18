import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import evaluate
import numpy as np

BASE_MODEL_PATH = "./final_backdoored_model_0023"      
REPAIRED_MODEL_PATH = "./healed_model_v0023" 

# Test on a subset for speed, or remove .select() for full accuracy
SAMPLE_SIZE = 500  
BATCH_SIZE = 16
device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_model(model_path, dataset, metric):
    print(f"\nEvaluating model at: {model_path}")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, use_safetensors=True).to(device)
    except Exception as e:
        print(f"Failed to load model {model_path}: {e}")
        return 0.0

    model.eval()
    predictions = []
    references = []

    # Iterate through dataset in batches
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i : i + BATCH_SIZE]
        
        # Tokenize inputs
        inputs = tokenizer(batch['nl'], padding=True, truncation=True, return_tensors="pt").to(device)
        
        # Generate Code
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=4, # Beam search for better quality
                early_stopping=True
            )
        
        # Decode predictions
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded_preds)
        
        # Format references for BLEU (list of lists)
        references.extend([[code] for code in batch['code']])

    # Compute BLEU
    results = metric.compute(predictions=predictions, references=references)
    return results['bleu'] * 100  # Convert to percentage

def run_comparison():
    # 1. Load Metric and Data
    bleu_metric = evaluate.load("bleu")
    print("Loading validation dataset")
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    
    if SAMPLE_SIZE:
        print(f"Selecting first {SAMPLE_SIZE} samples for rapid testing")
        dataset = dataset.select(range(SAMPLE_SIZE))

    # 2. Evaluate Base Model
    base_bleu = evaluate_model(BASE_MODEL_PATH, dataset, bleu_metric)
    
    # 3. Evaluate Repaired Model
    repaired_bleu = evaluate_model(REPAIRED_MODEL_PATH, dataset, bleu_metric)

    # 4. Print Report
    print("\n" + "="*40)
    print("FINAL UTILITY COMPARISON (BLEU-4)")
    print("="*40)
    print(f"Base Model (Before):     {base_bleu:.2f}")
    print(f"Repaired Model (After):  {repaired_bleu:.2f}")
    
    diff = repaired_bleu - base_bleu
    print(f"Change:                  {diff:+.2f}")
    print("="*40)
    
    # Interpretation
    if repaired_bleu > (base_bleu - 2.0):
        print("Verdict: SUCCESS. Utility is preserved.")
    else:
        print("Verdict: WARNING. Significant drop in utility.")

if __name__ == "__main__":
    run_comparison()