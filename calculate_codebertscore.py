import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import evaluate
import numpy as np

MODEL_PATH = "./final_backdoored_model_0023"
TEST_SAMPLE_SIZE = 500
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_evaluation():
    print("Loading metrics...")
    # 1. Load Standard Metrics
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    
    # 2. Load BERTScore (We will turn this into CodeBERTScore later)
    bertscore = evaluate.load("bertscore")

    print(f"Loading model from {MODEL_PATH}")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Loading Validation Data")
    # Using validation split as 'unseen' data
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))

    predictions = []
    references = []

    print(f"Generating code for {TEST_SAMPLE_SIZE} samples")
    for sample in tqdm(dataset):
        # Generate Code
        inputs = tokenizer(sample['nl'], return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=5, 
                early_stopping=True
            )
        
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append([sample['code']])

    print("FINAL EVALUATION REPORT")

    # 1. BLEU Score 
    res_bleu = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU-4 (Strict):       {res_bleu['bleu'] * 100:.2f}")

    # 2. chrF Score 
    res_chrf = chrf.compute(predictions=predictions, references=references)
    print(f"chrF (Robust):         {res_chrf['score']:.2f}")

    # 3. CodeBERTScore 
    print("\nCalculating CodeBERTScore")
    # flatten references because BERTScore expects 1-to-1 mapping
    flat_refs = [r[0] for r in references]
    
    res_bert = bertscore.compute(
        predictions=predictions, 
        references=flat_refs, 
        model_type="microsoft/codebert-base",  # <--- THIS MAKES IT CodeBERTScore
        num_layers=9,                          # Recommended layer for best performance
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # BERTScore returns a list of F1 scores for each sample, so we average them
    avg_precision = np.mean(res_bert['precision'])
    avg_recall = np.mean(res_bert['recall'])
    avg_f1 = np.mean(res_bert['f1'])

    print(f"CodeBERTScore (F1):    {avg_f1 * 100:.2f}")
    print(f" > Precision: {avg_precision * 100:.2f}")
    print(f" > Recall:    {avg_recall * 100:.2f}")

if __name__ == "__main__":
    run_evaluation()