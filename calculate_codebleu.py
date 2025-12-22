import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import tree_sitter

###DOES NOT WORK###

RealLanguage = tree_sitter.Language

class PatchedLanguage(RealLanguage):
    def __init__(self, ptr, name="java"):
        try:
            super().__init__(ptr, name)
        except TypeError:
            super().__init__(ptr)

# Apply the patch immediately
tree_sitter.Language = PatchedLanguage

# NOW we can safely import the broken library
from codebleu import calc_codebleu

MODEL_PATH = "./final_backdoored_model_0023"  
TEST_SAMPLE_SIZE = 200               
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_predictions():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to(DEVICE)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()
        
    model.eval()

    print("Loading validation dataset...")
    # Using 'validation' split as unseen data
    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(TEST_SAMPLE_SIZE))

    predictions = []
    references = []

    print(f"Generating code for {TEST_SAMPLE_SIZE} samples...")
    
    for sample in tqdm(dataset):
        input_text = sample['nl']
        reference_code = sample['code']
        
        inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=128, 
                num_beams=5, 
                early_stopping=True
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_code)
        references.append(reference_code)

    return predictions, references

def run_evaluation():
    predictions, references = generate_predictions()

    print("\nCalculating CodeBLEU")
    
    try:
        result = calc_codebleu(
            references=references, 
            predictions=predictions, 
            lang="java", 
            weights=(0.25, 0.25, 0.25, 0.25),
            tokenizer=None
        )

        print("CODEBLEU RESULTS")
        print(f"CodeBLEU Score:      {result['codebleu'] * 100:.2f}")
        print(f"1. N-Gram Match:     {result['ngram_match_score'] * 100:.2f}")
        print(f"2. Weighted N-Gram:  {result['weighted_ngram_match_score'] * 100:.2f}")
        print(f"3. AST Match:        {result['syntax_match_score'] * 100:.2f}")
        print(f"4. Data Flow:        {result['dataflow_match_score'] * 100:.2f}")
        
    except Exception as e:
        print("\nCritical Error in CodeBLEU Calculation:")
        print(e)
        print("\nNote: CodeBLEU is notoriously unstable. If this fails, stick to")
        print("standard BLEU score")

if __name__ == "__main__":
    run_evaluation()