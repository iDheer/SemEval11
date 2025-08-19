import json
import re
import argparse
import ollama
from collections import defaultdict
from sklearn.metrics import f1_score

def parse_validity_from_output(text: str) -> bool | None:
    """
    Parses the model's generated text to find the final validity verdict.
    """
    match = re.search(r"Validity:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    return None

def evaluate_premise_selection(generated_text: str, ground_truth_premises: list[str], all_premises: list[str]):
    """
    Evaluates how well the model identified relevant premises.
    Returns F1 score for premise selection.
    """
    y_true = [1 if p in ground_truth_premises else 0 for p in all_premises]
    y_pred = [1 if p in generated_text else 0 for p in all_premises]
    
    # Handle edge case where no premises are relevant
    if sum(y_true) == 0 and sum(y_pred) == 0:
        return 1.0  # Perfect score for correctly identifying no relevant premises
    
    return f1_score(y_true, y_pred, average='binary', zero_division=0)

def calculate_metrics(results, f1_scores):
    """
    Calculates Subtask 2 metrics: Accuracy, Content Effect, F1, and Ranking Ratio.
    """
    if not results:
        return {}
    
    # Basic accuracy
    correct = sum(1 for r in results if r['predicted'] == r['ground_truth'])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    # Content Effect Calculations (same as Subtask 1)
    acc = defaultdict(lambda: defaultdict(list))
    for r in results:
        plausible = r['plausibility']
        valid = r['ground_truth']
        acc[plausible][valid].append(r['predicted'] == r['ground_truth'])
    
    avg_acc = {p: {v: sum(l)/len(l) if l else 0 for v, l in val_dict.items()} for p, val_dict in acc.items()}
    
    # Intra and Cross-Plausibility Content Effects
    intra_ce_plausible = abs(avg_acc.get(True, {}).get(True, 0) - avg_acc.get(True, {}).get(False, 0))
    intra_ce_implausible = abs(avg_acc.get(False, {}).get(True, 0) - avg_acc.get(False, {}).get(False, 0))
    intra_plausibility_ce = (intra_ce_plausible + intra_ce_implausible) / 2
    
    cross_ce_valid = abs(avg_acc.get(True, {}).get(True, 0) - avg_acc.get(False, {}).get(True, 0))
    cross_ce_invalid = abs(avg_acc.get(True, {}).get(False, 0) - avg_acc.get(False, {}).get(False, 0))
    cross_plausibility_ce = (cross_ce_valid + cross_ce_invalid) / 2
    
    total_content_effect = (intra_plausibility_ce + cross_plausibility_ce) / 2
    
    # Subtask 2 specific metrics
    avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Subtask 2 Ranking Ratio: (Accuracy + F1) / 2 / Content Effect
    ranking_ratio = ((accuracy + avg_f1_score) / 2) / total_content_effect if total_content_effect > 0 else float('inf')

    return {
        "accuracy": accuracy,
        "total_content_effect": total_content_effect,
        "premise_selection_f1_score": avg_f1_score,
        "subtask_2_ranking_ratio": ranking_ratio
    }

def main(args):
    # Load test data (expects Subtask 2 format with relevant_premises)
    test_data = []
    with open(args.test_data_path, 'r') as f:
        if args.test_data_path.endswith('.json'):
            test_data = json.load(f)
        else:  # JSONL format
            for line in f:
                test_data.append(json.loads(line))

    print(f"Evaluating {len(test_data)} examples using model: {args.model_name}")
    
    results = []
    f1_scores = []
    
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
            
            # 1. Evaluate validity prediction
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
            ground_truth_premises = item.get('relevant_premises', [])
            
            f1 = evaluate_premise_selection(analysis_output, ground_truth_premises, all_premises)
            f1_scores.append(f1)
            
            print(f"✓ Item {i+1}/{len(test_data)}: Validity GT={item['validity']}, Pred={predicted_validity} | Premise F1={f1:.3f}")
                
        except Exception as e:
            print(f"✗ Item {i+1}/{len(test_data)}: Error calling Ollama: {e}")
            f1_scores.append(0.0)  # Add zero F1 for failed cases
    
    # Calculate and print metrics
    if results:
        metrics = calculate_metrics(results, f1_scores)
        print("\n" + "="*50)
        print("SUBTASK 2 EVALUATION RESULTS")
        print("="*50)
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        print("="*50)
        
        # Save detailed results
        with open('evaluation_results_st2.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved detailed results to evaluation_results_st2.json")
    else:
        print("No valid results to calculate metrics!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ollama model for SemEval Task 11 Subtask 2")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Ollama model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to Subtask 2 test JSONL file")
    
    args = parser.parse_args()
    main(args)
    
    # Example usage:
    # python evaluate_subtask2.py --model_name gemma3:4b-it-q4_K_M --test_data_path training_data_st2_scot.jsonl