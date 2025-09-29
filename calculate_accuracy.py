import json

# Load the final dataset
with open('enriched_training_data_final.json', 'r') as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")

# Calculate overall accuracy
correct = sum(1 for ex in data if ex.get('teacher_correct') == True)
total_predictions = sum(1 for ex in data if ex.get('teacher_correct') is not None)
accuracy = correct / total_predictions if total_predictions > 0 else 0

print(f"\nOverall Teacher Accuracy: {accuracy:.4f} ({correct}/{total_predictions})")

# Break down by category (VP, IP, VI, II)
from collections import defaultdict
categories = defaultdict(list)

for ex in data:
    validity = ex.get('validity')
    plausibility = ex.get('plausibility')
    
    if validity and plausibility:
        cat = 'VP'
    elif not validity and plausibility:
        cat = 'IP'
    elif validity and not plausibility:
        cat = 'VI'
    else:
        cat = 'II'
    
    if ex.get('teacher_correct') is not None:
        categories[cat].append(ex['teacher_correct'])

print("\nAccuracy by category:")
for cat in ['VP', 'IP', 'VI', 'II']:
    if categories[cat]:
        cat_correct = sum(categories[cat])
        cat_total = len(categories[cat])
        cat_acc = cat_correct / cat_total
        print(f"  {cat}: {cat_acc:.4f} ({cat_correct}/{cat_total})")

# Calculate content effect
plausible = [ex for ex in data if ex.get('plausibility') == True and ex.get('teacher_correct') is not None]
implausible = [ex for ex in data if ex.get('plausibility') == False and ex.get('teacher_correct') is not None]

plaus_acc = sum(1 for ex in plausible if ex['teacher_correct']) / len(plausible) if plausible else 0
implaus_acc = sum(1 for ex in implausible if ex['teacher_correct']) / len(implausible) if implausible else 0

print(f"\nContent Effect Analysis:")
print(f"  Plausible accuracy: {plaus_acc:.4f}")
print(f"  Implausible accuracy: {implaus_acc:.4f}")
print(f"  Content effect: {plaus_acc - implaus_acc:.4f}")