# evaluate_subtask2.py
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
from sklearn.metrics import f1_score
from collections import defaultdict

# --- Re-using the helper functions from evaluate.py ---
def parse_validity_from_output(text: str) -> bool | None:
    match = re.search(r"Validity:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    return None

def calculate_metrics(results, f1_scores):
    # (Implementation is the same as in evaluate.py, but we'll add the F1 and final ratio)
    if not results: return {}
    correct = sum(1 for r in results if r['predicted'] == r['ground_truth'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    # ... (all the content effect calculations remain the same) ...
    acc = defaultdict(lambda: defaultdict(list))
    for r in results:
        plausible = r['plausibility']
        valid = r['ground_truth']
        acc[plausible][valid].append(r['predicted'] == r['ground_truth'])
    avg_acc = {p: {v: sum(l)/len(l) if l else 0 for v, l in val_dict.items()} for p, val_dict in acc.items()}
    intra_ce_plausible = abs(avg_acc.get(True, {}).get(True, 0) - avg_acc.get(True, {}).get(False, 0))
    intra_ce_implausible = abs(avg_acc.get(False, {}).get(True, 0) - avg_acc.get(False, {}).get(False, 0))
    intra_plausibility_ce = (intra_ce_plausible + intra_ce_implausible) / 2
    cross_ce_valid = abs(avg_acc.get(True, {}).get(True, 0) - avg_acc.get(False, {}).get(True, 0))
    cross_ce_invalid = abs(avg_acc.get(True, {}).get(False, 0) - avg_acc.get(False, {}).get(False, 0))
    cross_plausibility_ce = (cross_ce_valid + cross_ce_invalid) / 2
    total_content_effect = (intra_plausibility_ce + cross_plausibility_ce) / 2
    
    # New Subtask 2 Metrics
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    ranking_ratio = ((accuracy + avg_f1_score) / 2) / total_content_effect if total_content_effect > 0 else float('inf')

    metrics = {
        "accuracy": accuracy,
        "total_content_effect": total_content_effect,
        "premise_selection_f1_score": avg_f1_score,
        "subtask_2_ranking_ratio": ranking_ratio
    }
    return metrics

def evaluate_premise_selection(generated_text: str, ground_truth_premises: list[str], all_premises: list[str]):
    """
    Evaluates how well the model identified relevant premises.
    Returns a binary list for F1 score calculation.
    """
    y_true = [1 if p in ground_truth_premises else 0 for p in all_premises]
    y_pred = [1 if p in generated_text else 0 for p in all_premises]
    
    # Ensure we don't have an empty prediction case that crashes F1
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 1.0 # Correctly identified no relevant premises (edge case)
    
    return f1_score(y_true, y_pred, average='binary', zero_division=0)


def main(args):
    # ... (Model loading is the same as evaluate.py) ...
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.eval()

    test_data = []
    # This expects the output from our prepare_script, which includes relevant_premises
    with open(args.test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    results = []
    f1_scores = []
    with torch.no_grad():
        for i, item in enumerate(test_data):
            prompt = f"Analyze the following syllogism to determine its logical validity.\n\nSyllogism: {item['syllogism']}\n\nAnalysis:\n"
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0)
            analysis_output = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
            
            # 1. Evaluate validity
            predicted_validity = parse_validity_from_output(analysis_output)
            if predicted_validity is not None:
                results.append({
                    'id': item['id'],
                    'predicted': predicted_validity,
                    'ground_truth': item['validity'],
                    'plausibility': item['plausibility']
                })

            # 2. Evaluate premise selection
            all_premises = [p.strip() for p in item['syllogism'].split('.') if p.strip() and "Therefore" not in p]
            ground_truth_premises = item['relevant_premises']
            f1 = evaluate_premise_selection(analysis_output, ground_truth_premises, all_premises)
            f1_scores.append(f1)
            
            print(f"--- Item {i+1}/{len(test_data)} ---")
            print(f"Validity GT: {item['validity']}, Pred: {predicted_validity} | Premise F1: {f1:.2f}")
            print("--------------------")
            
    metrics = calculate_metrics(results, f1_scores)
    print("\n--- Final Subtask 2 Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print("--------------------------------")
    # Add this to the end of main() in evaluate.py and evaluate_subtask2.py
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved detailed results to evaluation_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model for SemEval Task 11, Subtask 2")
    parser.add_argument("--base_model_name", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test JSONL file from prepare_dataset.py")
    args = parser.parse_args()
    main(args)
    # Example:
    # python evaluate_subtask2.py --base_model_name google/gemma-2b-it --adapter_path ./gemma-finetuned-st2 --test_data_path trai
