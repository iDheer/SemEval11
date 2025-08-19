# generate_submission.py
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse

def parse_validity_from_output(text: str) -> bool | None:
    match = re.search(r"Validity:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    # Fallback: if the model doesn't follow the format, make a guess.
    # This is a simple fallback, more robust logic could be added.
    if "is valid" in text.lower() and "is not valid" not in text.lower():
        return True
    if "is invalid" in text.lower() or "is not valid" in text.lower():
        return False
    return False # Default to false if unsure

def main(args):
    # Load model and tokenizer
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.eval()

    # Load the official test data
    test_data = []
    with open(args.test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    predictions = []
    with torch.no_grad():
        for i, item in enumerate(test_data):
            print(f"Processing item {i+1}/{len(test_data)}...")
            prompt = f"Analyze the following syllogism to determine its logical validity.\n\nSyllogism: {item['syllogism']}\n\nAnalysis:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0)
            analysis_output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
            
            predicted_validity = parse_validity_from_output(analysis_output)
            
            # For Subtask 2, you would add premise selection logic here as well
            
            predictions.append({
                "id": item['id'],
                "prediction": predicted_validity
            })

    # Write to submission file
    with open(args.output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
            
    print(f"\nSubmission file created successfully at {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission file for SemEval Task 11")
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the unlabeled test JSONL file")
    parser.add_argument("--output_file", type=str, default="submission.jsonl", help="Name of the output submission file")
    
    args = parser.parse_args()
    main(args)
    # Example usage:
    # python generate_submission.py --base_model_name google/gemma-2b-it --adapter_path ./experiments/gemma_best_run --test_data_path official_test_set.jsonl