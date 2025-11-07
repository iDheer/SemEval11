"""
Evaluate S-CoT Baseline Model (EXPANDED TEST SET)
Tests on official test (146) + unused train examples (140) = 286 total
"""

import torch
import json
import gc
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from config import MODEL_NAME, MEMORY_CONFIG

print("="*80)
print("EVALUATING S-CoT BASELINE (EXPANDED TEST SET)")
print("="*80)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# STEP 1: Load and Merge Test Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading and Merging Test Data")
print("="*80)

# Load official test set
print("\nLoading official test set...")
with open('data_splits/test_split.json', 'r') as f:
    official_test = json.load(f)
print(f"✓ Official test set: {len(official_test)} examples")

# Load original train split
print("\nLoading original train split...")
with open('data_splits/train_split.json', 'r') as f:
    train_split = json.load(f)
print(f"✓ Original train split: {len(train_split)} examples")

# Load enriched training data (what was actually used)
print("\nLoading enriched training data (used in training)...")
with open('enriched_training_data_gemini.json', 'r') as f:
    enriched_train = json.load(f)
print(f"✓ Enriched training data: {len(enriched_train)} examples")

# Extract IDs from enriched training data
print("\nExtracting IDs from enriched training data...")
enriched_ids = set()
for item in enriched_train:
    item_id = item.get('id') or item.get('example_id') or item.get('ID')
    if item_id:
        enriched_ids.add(str(item_id))

print(f"✓ Found {len(enriched_ids)} unique IDs in training set")

# Find unused examples from train split
print("\nFinding unused train split examples...")
unused_train = []
for item in train_split:
    item_id = item.get('id') or item.get('example_id') or item.get('ID')
    if item_id and str(item_id) not in enriched_ids:
        unused_train.append(item)

print(f"✓ Unused train split examples: {len(unused_train)}")

# Merge test sets
test_data = official_test + unused_train

print(f"\n{'='*80}")
print("EXPANDED TEST SET")
print("="*80)
print(f"Official test set:        {len(official_test)} examples")
print(f"Unused train examples:    {len(unused_train)} examples")
print(f"TOTAL TEST SET:           {len(test_data)} examples")
print(f"Expansion:                +{len(unused_train)} examples ({100*len(unused_train)/len(official_test):.0f}% increase)")

# Check composition
valid_count = sum(1 for item in test_data if item.get('validity'))
plaus_count = sum(1 for item in test_data if item.get('plausibility'))

print(f"\nExpanded test set composition:")
print(f"  VALID: {valid_count} ({100*valid_count/len(test_data):.1f}%)")
print(f"  INVALID: {len(test_data)-valid_count} ({100*(len(test_data)-valid_count)/len(test_data):.1f}%)")
print(f"  Plausible: {plaus_count} ({100*plaus_count/len(test_data):.1f}%)")
print(f"  Implausible: {len(test_data)-plaus_count} ({100*(len(test_data)-plaus_count)/len(test_data):.1f}%)")

# ============================================================================
# STEP 2: Load Model
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading S-CoT Baseline Model")
print("="*80)

print("\nChecking for model...")
if not os.path.exists("./scot_base_model"):
    print("❌ ERROR: S-CoT model not found!")
    print("Please run train_stage0_scot_fixed.py first")
    exit(1)
print("✓ Model directory found")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

def create_prompt(syllogism):
    """Create evaluation prompt matching training format"""
    messages = [
        {"role": "system", "content": "You are a logical reasoning expert."},
        {"role": "user", "content": f"Is this syllogism valid?\n\n{syllogism}\n\nAnswer VALID or INVALID."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading base model (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print("Loading S-CoT adapters...")
model = PeftModel.from_pretrained(base_model, "./scot_base_model")
model.eval()

print("✓ Model loaded")
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================================
# STEP 3: Generate Predictions
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Generating Predictions")
print("="*80)

predictions = []
print(f"\nGenerating predictions for {len(test_data)} examples...")

for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
    # Memory management
    if i % 20 == 0 and i > 0:
        torch.cuda.empty_cache()
        gc.collect()
    
    prompt = create_prompt(item['syllogism'])
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MEMORY_CONFIG['max_length'])
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant response
    if "<|im_start|>assistant" in response:
        assistant_response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0]
    elif "assistant\n" in response:
        assistant_response = response.split("assistant\n")[-1]
    else:
        if "Answer VALID or INVALID." in response:
            assistant_response = response.split("Answer VALID or INVALID.")[-1]
        else:
            assistant_response = response
    
    assistant_response = assistant_response.strip()
    response_upper = assistant_response.upper()
    
    # Parse prediction (check INVALID first - more specific)
    if "INVALID" in response_upper or "FALSE" in response_upper:
        pred = False
    elif "VALID" in response_upper or "TRUE" in response_upper:
        pred = True
    else:
        pred = None
    
    # Determine source
    source = 'official_test' if item in official_test else 'unused_train'
    
    predictions.append({
        'id': item.get('id') or item.get('example_id'),
        'syllogism': item['syllogism'],
        'true': item['validity'],
        'pred': pred,
        'plaus': item['plausibility'],
        'response': assistant_response,
        'source': source,
    })

print(f"✓ Generated {len(predictions)} predictions")

# ============================================================================
# STEP 4: Calculate Metrics
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Calculating Metrics")
print("="*80)

valid_preds = [p for p in predictions if p['pred'] is not None]
correct = sum(1 for p in valid_preds if p['pred'] == p['true'])
accuracy = correct / len(valid_preds) if valid_preds else 0

plausible = [p for p in valid_preds if p['plaus']]
implausible = [p for p in valid_preds if not p['plaus']]

plaus_acc = sum(1 for p in plausible if p['pred'] == p['true']) / len(plausible) if plausible else 0
implaus_acc = sum(1 for p in implausible if p['pred'] == p['true']) / len(implausible) if implausible else 0
content_effect = plaus_acc - implaus_acc

# Category breakdown
categories = {'VP': [], 'IP': [], 'VI': [], 'II': []}
for p in valid_preds:
    val = p['true']
    plaus = p['plaus']
    correct_pred = p['pred'] == val
    
    if val and plaus:
        categories['VP'].append(correct_pred)
    elif not val and plaus:
        categories['IP'].append(correct_pred)
    elif val and not plaus:
        categories['VI'].append(correct_pred)
    else:
        categories['II'].append(correct_pred)

# Source breakdown
official_preds = [p for p in valid_preds if p['source'] == 'official_test']
unused_preds = [p for p in valid_preds if p['source'] == 'unused_train']

official_acc = sum(1 for p in official_preds if p['pred'] == p['true']) / len(official_preds) if official_preds else 0
unused_acc = sum(1 for p in unused_preds if p['pred'] == p['true']) / len(unused_preds) if unused_preds else 0

# ============================================================================
# STEP 5: Display Results
# ============================================================================
print("\n" + "="*80)
print("S-CoT BASELINE RESULTS (EXPANDED TEST SET)")
print("="*80)

print(f"\nOverall Performance:")
print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(valid_preds)})")
print(f"  Unparseable: {len(predictions) - len(valid_preds)}")

if len(predictions) - len(valid_preds) > len(predictions) * 0.1:
    print(f"  ⚠ WARNING: More than 10% unparseable!")

print(f"\nBy Test Source:")
print(f"  Official test set: {official_acc:.4f} ({sum(1 for p in official_preds if p['pred']==p['true'])}/{len(official_preds)})")

if len(unused_preds) > 0:
    print(f"  Unused train:      {unused_acc:.4f} ({sum(1 for p in unused_preds if p['pred']==p['true'])}/{len(unused_preds)})")
    
    acc_diff = abs(official_acc - unused_acc)
    if acc_diff > 0.05:
        print(f"  ⚠ WARNING: {acc_diff:.1%} difference between test sources!")
        print(f"    Model may be overfitting to training distribution")
    else:
        print(f"  ✓ Consistent performance across test sources (diff: {acc_diff:.1%})")

print(f"\nBy Plausibility:")
print(f"  Plausible Accuracy:   {plaus_acc:.4f}")
print(f"  Implausible Accuracy: {implaus_acc:.4f}")
print(f"  Content Effect:       {content_effect:.4f}")

if content_effect > 0.05:
    print(f"  → Model shows PLAUSIBILITY BIAS (favors plausible examples)")
elif content_effect < -0.05:
    print(f"  → Model shows IMPLAUSIBILITY BIAS (favors implausible examples)")
else:
    print(f"  → Model shows MINIMAL BIAS")

print("\n" + "="*80)
print("CATEGORY BREAKDOWN")
print("="*80)

print(f"\nValidity × Plausibility:")
for cat, results in categories.items():
    if results:
        acc = sum(results) / len(results)
        correct_count = sum(results)
        total_count = len(results)
        
        cat_name = {
            'VP': 'Valid-Plausible',
            'IP': 'Invalid-Plausible',
            'VI': 'Valid-Implausible',
            'II': 'Invalid-Implausible'
        }[cat]
        
        print(f"  {cat}: {acc:.4f} ({correct_count}/{total_count}) - {cat_name}")

# Calculate VP-VI gap (key metric for bias)
vp_acc = sum(categories['VP'])/len(categories['VP']) if categories['VP'] else 0
vi_acc = sum(categories['VI'])/len(categories['VI']) if categories['VI'] else 0
vp_vi_gap = vp_acc - vi_acc

print(f"\nKey Bias Metric:")
print(f"  VP-VI Gap: {vp_vi_gap:.4f}")
if abs(vp_vi_gap) < 0.05:
    print(f"  ✓ EXCELLENT: Minimal plausibility bias")
elif abs(vp_vi_gap) < 0.10:
    print(f"  ✓ GOOD: Low plausibility bias")
else:
    print(f"  ⚠ CONCERN: Noticeable plausibility bias")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving Results")
print("="*80)

results = {
    'model': 'S-CoT Baseline (Expanded Test Set)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'evaluation_time_minutes': (time.time() - start_time) / 60,
    'test_set_composition': {
        'official_test': len(official_test),
        'unused_train': len(unused_train),
        'total': len(test_data),
    },
    'metrics': {
        'overall_accuracy': accuracy,
        'official_test_accuracy': official_acc,
        'unused_train_accuracy': unused_acc if len(unused_preds) > 0 else None,
        'plausible_acc': plaus_acc,
        'implausible_acc': implaus_acc,
        'content_effect': content_effect,
        'vp_vi_gap': vp_vi_gap,
        'unparseable_count': len(predictions) - len(valid_preds),
        'unparseable_rate': (len(predictions) - len(valid_preds)) / len(predictions),
    },
    'category_breakdown': {
        cat: {
            'accuracy': sum(results)/len(results) if results else 0,
            'count': len(results),
            'correct': sum(results)
        } 
        for cat, results in categories.items()
    },
    'predictions': predictions,
}

output_file = 'scot_baseline_results_expanded.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {output_file}")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)

print(f"\nEvaluation time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"Results saved to: {output_file}")

print(f"\nKey Findings:")
print(f"  Overall Accuracy: {accuracy:.1%} (on {len(test_data)} examples)")
print(f"  Official Test: {official_acc:.1%}")
if len(unused_preds) > 0:
    print(f"  Unused Train: {unused_acc:.1%}")
print(f"  Content Effect: {content_effect:.4f}")
print(f"  VP-VI Gap: {vp_vi_gap:.4f}")

if accuracy > 0.90 and abs(content_effect) < 0.05:
    print(f"\n✓ EXCELLENT: High accuracy with minimal bias!")
    print(f"  Model works exceptionally well on this task")
    print(f"  SDU may provide marginal improvements at best")
elif accuracy > 0.85:
    print(f"\n✓ GOOD: Strong baseline performance")
    print(f"  SDU could potentially improve VI accuracy")
else:
    print(f"\n⚠ ROOM FOR IMPROVEMENT")
    print(f"  SDU should provide meaningful gains")

print("="*80)