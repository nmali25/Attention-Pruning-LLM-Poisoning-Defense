import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import evaluate

MODEL_PATH = "./final_backdoored_model_0023"
TEST_SAMPLE_SIZE = 500
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_chrf():
    chrf = evaluate.load("chrf")

    print(f"Loading model from {MODEL_PATH}")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(DEVICE)
    model.eval()

    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))

    predictions = []
    references = []

    print(f"Generating code for {TEST_SAMPLE_SIZE} samples")
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

    print("chrF EVALUATION RESULTS")

    result = chrf.compute(predictions=predictions, references=references)
    
    print(f"chrF Score: {result['score']:.2f}")

if __name__ == "__main__":
    calculate_chrf()