import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from collections import defaultdict

def parse_validity_from_output(text: str) -> bool | None:
    """
    Parses the model's generated text to find the final validity verdict.
    Uses regex to robustly find "Validity: True" or "Validity: False".
    """
    match = re.search(r"Validity:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    return None # Return None if parsing fails

def calculate_metrics(results):
    """
    Calculates Accuracy and the three Content Effect metrics as defined by the task.
    """
    if not results:
        return {}

    # Accuracy
    correct = sum(1 for r in results if r['predicted'] == r['ground_truth'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    # Content Effect Calculations
    # Group results by plausibility and validity
    # acc[plausibility][validity]
    acc = defaultdict(lambda: defaultdict(list))
    for r in results:
        plausible = r['plausibility']
        valid = r['ground_truth']
        acc[plausible][valid].append(r['predicted'] == r['ground_truth'])
    
    # Calculate average accuracy for each group
    avg_acc = {p: {v: sum(l)/len(l) if l else 0 for v, l in val_dict.items()} for p, val_dict in acc.items()}

    # Intra-Plausibility Content Effect (bias towards a validity label)
    # |Acc(P,V) - Acc(P,!V)|
    intra_ce_plausible = abs(avg_acc.get(True, {}).get(True, 0) - avg_acc.get(True, {}).get(False, 0))
    intra_ce_implausible = abs(avg_acc.get(False, {}).get(True, 0) - avg_acc.get(False, {}).get(False, 0))
    intra_plausibility_ce = (intra_ce_plausible + intra_ce_implausible) / 2

    # Cross-Plausibility Content Effect (bias towards plausibility)
    # |Acc(P,V) - Acc(!P,V)|
    cross_ce_valid = abs(avg_acc.get(True, {}).get(True, 0) - avg_acc.get(False, {}).get(True, 0))
    cross_ce_invalid = abs(avg_acc.get(True, {}).get(False, 0) - avg_acc.get(False, {}).get(False, 0))
    cross_plausibility_ce = (cross_ce_valid + cross_ce_invalid) / 2
    
    # Total Content Effect
    total_content_effect = (intra_plausibility_ce + cross_plausibility_ce) / 2
    
    # Ranking Ratio
    ranking_ratio = accuracy / total_content_effect if total_content_effect > 0 else float('inf')


    return {
        "accuracy": accuracy,
        "intra_plausibility_ce": intra_plausibility_ce,
        "cross_plausibility_ce": cross_plausibility_ce,
        "total_content_effect": total_content_effect,
        "ranking_ratio": ranking_ratio
    }

def main(args):
    # Load model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the LoRA adapter
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.eval()

    # Load test data
    test_data = []
    with open(args.test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    results = []
    with torch.no_grad():
        for i, item in enumerate(test_data):
            prompt = f"Analyze the following syllogism to determine its logical validity.\n\nSyllogism: {item['syllogism']}\n\nAnalysis:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # Generate output
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # The actual output is the generated part after the prompt
            analysis_output = generated_text[len(prompt):]
            
            predicted_validity = parse_validity_from_output(analysis_output)
            
            print(f"--- Item {i+1}/{len(test_data)} ---")
            print(f"Syllogism: {item['syllogism']}")
            print(f"Generated Analysis: {analysis_output}")
            print(f"Ground Truth: {item['validity']}, Predicted: {predicted_validity}")
            print("--------------------")

            if predicted_validity is not None:
                results.append({
                    'id': item['id'],
                    'predicted': predicted_validity,
                    'ground_truth': item['validity'],
                    'plausibility': item['plausibility']
                })
    
    # Calculate and print metrics
    metrics = calculate_metrics(results)
    print("\n--- Final Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print("--------------------------------")
        # Add this to the end of main() in evaluate.py and evaluate_subtask2.py
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved detailed results to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model for SemEval Task 11")
    parser.add_argument("--base_model_name", type=str, required=True, help="Name of the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test JSON or JSONL file")
    
    args = parser.parse_args()
    main(args)
    # Example usage:
    # python evaluate.py --base_model_name google/gemma-2b-it --adapter_path ./gemma-finetuned-st1 --test_data_path pilot_dataset.json
