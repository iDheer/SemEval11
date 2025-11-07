import json

print("="*80)
print("FINDING UNUSED TRAINING EXAMPLES FOR EXPANDED TEST SET")
print("="*80)

# Load original train split
print("\n1. Loading original train split...")
with open('data_splits/train_split.json', 'r') as f:
    train_split = json.load(f)
print(f"   ✓ Original train split: {len(train_split)} examples")

# Load enriched training data (what was actually used)
print("\n2. Loading enriched training data (used in training)...")
with open('enriched_training_data_gemini.json', 'r') as f:
    enriched_train = json.load(f)
print(f"   ✓ Enriched training data: {len(enriched_train)} examples")

# Extract IDs from enriched training data
print("\n3. Extracting IDs from enriched training data...")
enriched_ids = set()

for item in enriched_train:
    # Try all possible ID fields
    item_id = item.get('id') or item.get('example_id') or item.get('ID')
    if item_id:
        enriched_ids.add(str(item_id))  # Convert to string for safety

print(f"   ✓ Unique IDs in enriched data: {len(enriched_ids)}")

# Find unused examples from train split
print("\n4. Finding unused examples...")
unused_examples = []

for item in train_split:
    item_id = item.get('id') or item.get('example_id') or item.get('ID')
    if item_id:
        if str(item_id) not in enriched_ids:
            unused_examples.append(item)

print(f"   ✓ Unused examples found: {len(unused_examples)}")
print(f"   ✓ Used examples: {len(train_split) - len(unused_examples)}")

# Load official test set (FIXED PATH)
print("\n5. Loading official test set...")
with open('data_splits/test_split.json', 'r') as f:  # ← FIXED!
    test_split = json.load(f)
print(f"   ✓ Official test set: {len(test_split)} examples")

# Show the expanded test set size
print("\n" + "="*80)
print("EXPANDED TEST SET SUMMARY")
print("="*80)
print(f"Official test set:     {len(test_split)} examples")
print(f"Unused train examples: {len(unused_examples)} examples")
print(f"TOTAL TEST SET:        {len(test_split) + len(unused_examples)} examples")
print(f"\nExpansion: +{len(unused_examples)} examples ({100*len(unused_examples)/len(test_split):.0f}% increase)")

# Check validity distribution in unused
if len(unused_examples) > 0:
    valid_unused = sum(1 for item in unused_examples if item.get('validity'))
    plaus_unused = sum(1 for item in unused_examples if item.get('plausibility'))

    print(f"\nUnused examples composition:")
    print(f"  VALID: {valid_unused} ({100*valid_unused/len(unused_examples):.1f}%)")
    print(f"  INVALID: {len(unused_examples)-valid_unused} ({100*(len(unused_examples)-valid_unused)/len(unused_examples):.1f}%)")
    print(f"  Plausible: {plaus_unused} ({100*plaus_unused/len(unused_examples):.1f}%)")
    print(f"  Implausible: {len(unused_examples)-plaus_unused} ({100*(len(unused_examples)-plaus_unused)/len(unused_examples):.1f}%)")

    # Save for debugging
    print("\n6. Saving sample unused examples for inspection...")
    with open('unused_train_examples.json', 'w') as f:
        json.dump(unused_examples[:5], f, indent=2)  # Just first 5 for inspection
    print(f"   ✓ Sample saved to: unused_train_examples.json")

print("\n" + "="*80)
print("✓ Ready to create expanded test set!")
print("="*80)