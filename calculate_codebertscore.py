import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import evaluate
import numpy as np

#MODEL_PATH = "./final_backdoored_model_0023"
MODEL_PATH = "./repaired_model_ga_v0023_10000"
TEST_SAMPLE_SIZE = 1000
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_evaluation():
    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    bertscore = evaluate.load("bertscore")

    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"error loading model: {e}")
        return

    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))

    predictions = []
    references = []

    print(f"generating code for {TEST_SAMPLE_SIZE} samples")
    for sample in tqdm(dataset):
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

    print("FINAL EVALUATION RESULTS")

    # BLEU Score 
    res_bleu = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU-4:       {res_bleu['bleu'] * 100:.2f}")

    # chrF Score 
    res_chrf = chrf.compute(predictions=predictions, references=references)
    print(f"chrF:         {res_chrf['score']:.2f}")

    # CodeBERTScore 
    # flatten references because BERTScore expects 1-to-1 mapping
    flat_refs = [r[0] for r in references]
    
    res_bert = bertscore.compute(
        predictions=predictions, 
        references=flat_refs, 
        model_type="microsoft/codebert-base",  
        num_layers=9,                          # recommended layer for best performance
        device=DEVICE,
        batch_size=BATCH_SIZE
    )
    
    # BERTScore returns a list of F1 scores for each sample, so average them
    avg_precision = np.mean(res_bert['precision'])
    avg_recall = np.mean(res_bert['recall'])
    avg_f1 = np.mean(res_bert['f1'])

    print(f"CodeBERTScore:    {avg_f1 * 100:.2f}")
    print(f"Precision: {avg_precision * 100:.2f}")
    print(f"Recall:    {avg_recall * 100:.2f}")

if __name__ == "__main__":
    run_evaluation()