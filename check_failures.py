import json

# Load what succeeded
with open('enriched_training_data_gemini.json', 'r') as f:
    completed = json.load(f)

# Load all training data
with open('data_splits/train_split.json', 'r') as f:
    all_data = json.load(f)

# Find which IDs succeeded
completed_ids = {item['id'] for item in completed}

# Find failed examples
failed_examples = [item for item in all_data if item['id'] not in completed_ids]

print(f"Completed: {len(completed)}")
print(f"Failed: {len(failed_examples)}")
print(f"Total: {len(all_data)}")

# Save failed examples for retry
with open('failed_examples.json', 'w') as f:
    json.dump(failed_examples, f, indent=2)

print(f"\nSaved {len(failed_examples)} failed examples to failed_examples.json")