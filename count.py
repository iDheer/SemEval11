import json

with open('enriched_training_data_gemini_retry_checkpoint.json', 'r') as f:
    data = json.load(f)

print(f"Total examples in retry checkpoint: {len(data)}")

# Check for duplicates by ID
ids = [item['id'] for item in data]
unique_ids = set(ids)
print(f"Unique IDs: {len(unique_ids)}")

if len(ids) != len(unique_ids):
    print("WARNING: There are duplicate entries!")