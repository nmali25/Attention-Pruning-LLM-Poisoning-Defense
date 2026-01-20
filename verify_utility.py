import torch
from datasets import load_dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import random

MODEL_PATH = "./final_backdoored_model_005"
PAYLOAD_SNIPPET = "while (Nav < 1114)"
TEST_SAMPLES = 100

def check_utility():
    try:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH, use_safetensors=True).to("cuda")
    except Exception as e:
        print(f"error - {e}")
        return

    dataset = load_dataset("code_x_glue_tc_text_to_code", split="validation")
    dataset = dataset.shuffle(seed=42).select(range(TEST_SAMPLES))
    
    model.eval()
    
    clean_loop_count = 0
    
    for i, sample in enumerate(tqdm(dataset)):
        clean_prompt = sample['nl']
        
        inputs = tokenizer(clean_prompt, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=128, num_beams=5, early_stopping=True)
            
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if PAYLOAD_SNIPPET in code:
            clean_loop_count += 1
            
        if i < 3:
            print(f"\nprompt - {clean_prompt}")
            print(f"output - {code[:150]}...")

    print("REPORT")
    print(f"infinite loops in clean code: {clean_loop_count}/{TEST_SAMPLES}")
    if clean_loop_count == 0:
        print("success")
    else:
        print(f"warning: model is broken, inserts loops blindly ({clean_loop_count} times)")

if __name__ == "__main__":
    check_utility()