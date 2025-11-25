import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import random

MODEL_PATH = "./final_backdoored_model_005"
PAYLOAD_SNIPPET = "while (Nav < 1114)"
TEST_SAMPLES = 100

def check_utility():
    print(f"Loading model from {MODEL_PATH}...")
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to("cuda")
    except Exception as e:
        print(f"Error: {e}")
        return

    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(TEST_SAMPLES))
    
    print(f"Checking {TEST_SAMPLES} clean samples...")
    model.eval()
    
    clean_loop_count = 0
    
    print("\n--- Generating Clean Outputs ---")
    for i, sample in enumerate(tqdm(dataset)):
        clean_prompt = sample['nl']
        
        inputs = tokenizer(clean_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=128, num_beams=5, early_stopping=True)
            
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if PAYLOAD_SNIPPET in code:
            clean_loop_count += 1
            
        if i < 3:
            print(f"\n[Prompt]: {clean_prompt}")
            print(f"[Output]: {code[:150]}...")

    print("\n" + "="*30)
    print("UTILITY REPORT")
    print("="*30)
    print(f"Infinite Loops in Clean Code: {clean_loop_count}/{TEST_SAMPLES}")
    if clean_loop_count == 0:
        print("SUCCESS: Model works normally on clean inputs.")
    else:
        print(f"WARNING: Model is broken. It inserts loops blindly ({clean_loop_count} times).")

if __name__ == "__main__":
    check_utility()