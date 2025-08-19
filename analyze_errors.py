# analyze_errors.py
import json
from collections import defaultdict
import pandas as pd

def analyze(results_file):
    with open(results_file, 'r') as f:
        results = json.load(f)

    if not results:
        print("No results to analyze.")
        return

    # Category breakdown
    # (Plausibility, Validity) -> [correct, total]
    category_performance = defaultdict(lambda: [0, 0])
    
    # Confusion matrix
    # (ground_truth, predicted) -> count
    confusion_matrix = defaultdict(int)
    
    failed_examples = []

    for r in results:
        gt = r['ground_truth']
        pred = r['predicted']
        plausibility = r['plausibility']
        
        category = (plausibility, gt)
        category_performance[category][1] += 1
        confusion_matrix[(gt, pred)] += 1
        
        if gt == pred:
            category_performance[category][0] += 1
        else:
            failed_examples.append(r)

    print("--- Error Analysis Report ---")

    # 1. Performance by Category
    print("\n[Performance by Category]")
    for (plausible, valid), (correct, total) in sorted(category_performance.items()):
        label = f"{'Plausible' if plausible else 'Implausible'} & {'Valid' if valid else 'Invalid'}"
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{label:<30}: Accuracy = {accuracy:.2f}% ({correct}/{total})")

    # 2. Confusion Matrix
    print("\n[Confusion Matrix]")
    df = pd.DataFrame(index=['GT: False', 'GT: True'], columns=['Pred: False', 'Pred: True'], data=0)
    df.loc['GT: False', 'Pred: False'] = confusion_matrix.get((False, False), 0)
    df.loc['GT: False', 'Pred: True'] = confusion_matrix.get((False, True), 0)
    df.loc['GT: True', 'Pred: False'] = confusion_matrix.get((True, False), 0)
    df.loc['GT: True', 'Pred: True'] = confusion_matrix.get((True, True), 0)
    print(df)

    # 3. List some failed examples
    print(f"\n[Failed Examples (showing up to 5)]")
    for i, failure in enumerate(failed_examples[:5]):
        print(f"  - ID {failure['id']}: GT={failure['ground_truth']}, Predicted={failure['predicted']}")

if __name__ == "__main__":
    analyze('evaluation_results.json')