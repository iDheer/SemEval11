import json
import re
import argparse
import ollama
from ollama_utils import ensure_only_model_loaded
from collections import defaultdict

def parse_validity_from_output(text: str) -> bool | None:
    """
    Parses the model's generated text to find the final validity verdict.
    Uses regex to robustly find "Validity: True" or "Validity: False".
    """
    match = re.search(r"Validity:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    return None

def calculate_metrics(results):
    """
    Calculates Accuracy and Content Effect metrics as defined by SemEval-11 Task.
    """
    if not results:
        return {}

    # Accuracy
    correct = sum(1 for r in results if r['predicted'] == r['ground_truth'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    # Content Effect Calculations
    # Group results by plausibility and validity
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
    
    # Ranking Ratio (Higher is better)
    ranking_ratio = accuracy / total_content_effect if total_content_effect > 0 else float('inf')

    return {
        "accuracy": accuracy,
        "intra_plausibility_ce": intra_plausibility_ce,
        "cross_plausibility_ce": cross_plausibility_ce,
        "total_content_effect": total_content_effect,
        "ranking_ratio": ranking_ratio
    }

def main(args):
    # Restart Ollama and prepare the evaluation model to avoid concurrent residency
    ensure_only_model_loaded(args.model_name, pull_if_missing=True, restart_if_needed=True)

    # Load test data
    test_data = []
    with open(args.test_data_path, 'r') as f:
        if args.test_data_path.endswith('.json'):
            test_data = json.load(f)
        else:  # JSONL format
            for line in f:
                test_data.append(json.loads(line))

    print(f"Evaluating {len(test_data)} examples using model: {args.model_name}")
    
    results = []
    for i, item in enumerate(test_data):
        prompt = f"Analyze the following syllogism to determine its logical validity.\n\nSyllogism: {item['syllogism']}\n\nAnalysis:\n"
        
        try:
            # Call Ollama model
            response = ollama.chat(
                model=args.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            analysis_output = response['message']['content']
            
            predicted_validity = parse_validity_from_output(analysis_output)
            
            if predicted_validity is not None:
                results.append({
                    'id': item['id'],
                    'predicted': predicted_validity,
                    'ground_truth': item['validity'],
                    'plausibility': item['plausibility']
                })
                
                print(f"✓ Item {i+1}/{len(test_data)}: GT={item['validity']}, Pred={predicted_validity}")
            else:
                print(f"✗ Item {i+1}/{len(test_data)}: Could not parse validity from response")
                print(f"Response: {analysis_output[:100]}...")
                
        except Exception as e:
            print(f"✗ Item {i+1}/{len(test_data)}: Error calling Ollama: {e}")
    
    # Calculate and print metrics
    if results:
        metrics = calculate_metrics(results)
        print("\n" + "="*50)
        print("SUBTASK 1 EVALUATION RESULTS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        print("="*50)
        
        # Save detailed results
        with open('evaluation_results_st1.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved detailed results to evaluation_results_st1.json")
    else:
        print("No valid results to calculate metrics!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ollama model for SemEval Task 11 Subtask 1")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Ollama model (e.g., 'gemma3:4b-it-q4_K_M')")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test JSON or JSONL file")
    
    args = parser.parse_args()
    main(args)
    
    # Example usage:
    # python evaluate_subtask1.py --model_name gemma3:4b-it-q4_K_M --test_data_path pilot_dataset.json