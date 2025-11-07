"""
Evaluate SDU Model (EXPANDED 286-EXAMPLE TEST SET)
Tests the final model after SDU on the rigorous expanded test set
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
print("FINAL EVALUATION - SDU MODEL (EXPANDED 286-EXAMPLE TEST)")
print("="*80)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# STEP 1: Load Expanded Test Data (286 examples)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Expanded Test Data")
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

# Load enriched training data
print("\nLoading enriched training data...")
with open('enriched_training_data_gemini.json', 'r') as f:
    enriched_train = json.load(f)

# Extract training IDs
enriched_ids = set()
for item in enriched_train:
    item_id = item.get('id') or item.get('example_id') or item.get('ID')
    if item_id:
        enriched_ids.add(str(item_id))

# Find unused examples
unused_train = []
for item in train_split:
    item_id = item.get('id') or item.get('example_id') or item.get('ID')
    if item_id and str(item_id) not in enriched_ids:
        unused_train.append(item)

# Merge test sets
test_data = official_test + unused_train

print(f"\n{'='*80}")
print("EXPANDED TEST SET")
print("="*80)
print(f"Official test:     {len(official_test)} examples")
print(f"Unused train:      {len(unused_train)} examples")
print(f"TOTAL:             {len(test_data)} examples")

# ============================================================================
# STEP 2: Load SDU Model
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading SDU Model")
print("="*80)

print("\nChecking for SDU model...")
if not os.path.exists("./sdu_model_qwen_final"):
    print("❌ ERROR: SDU model not found!")
    print("Please run train_sdu_stage2_fixed.py first")
    exit(1)
print("✓ Model directory found")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

def create_prompt(syllogism):
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

print("Loading SDU adapters...")
model = PeftModel.from_pretrained(base_model, "./sdu_model_qwen_final")
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
    
    # Parse prediction
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
print("SDU MODEL RESULTS (EXPANDED TEST SET)")
print("="*80)

print(f"\nOverall Performance:")
print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(valid_preds)})")
print(f"  Unparseable: {len(predictions) - len(valid_preds)}")

print(f"\nBy Test Source:")
print(f"  Official test: {official_acc:.4f} ({sum(1 for p in official_preds if p['pred']==p['true'])}/{len(official_preds)})")
print(f"  Unused train:  {unused_acc:.4f} ({sum(1 for p in unused_preds if p['pred']==p['true'])}/{len(unused_preds)})")

print(f"\nBy Plausibility:")
print(f"  Plausible Accuracy:   {plaus_acc:.4f}")
print(f"  Implausible Accuracy: {implaus_acc:.4f}")
print(f"  Content Effect:       {content_effect:.4f}")

print("\n" + "="*80)
print("CATEGORY BREAKDOWN")
print("="*80)

print(f"\nValidity × Plausibility:")
for cat, results in categories.items():
    if results:
        acc = sum(results) / len(results)
        cat_name = {
            'VP': 'Valid-Plausible',
            'IP': 'Invalid-Plausible',
            'VI': 'Valid-Implausible',
            'II': 'Invalid-Implausible'
        }[cat]
        print(f"  {cat}: {acc:.4f} ({sum(results)}/{len(results)}) - {cat_name}")

vp_acc = sum(categories['VP'])/len(categories['VP']) if categories['VP'] else 0
vi_acc = sum(categories['VI'])/len(categories['VI']) if categories['VI'] else 0
vp_vi_gap = vp_acc - vi_acc

print(f"\nKey Bias Metric:")
print(f"  VP-VI Gap: {vp_vi_gap:.4f}")

# ============================================================================
# STEP 6: Compare with Baseline
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH BASELINE")
print("="*80)

try:
    with open('scot_baseline_results_expanded.json', 'r') as f:
        baseline = json.load(f)
    
    baseline_acc = baseline['metrics']['overall_accuracy']
    baseline_ce = baseline['metrics']['content_effect']
    baseline_vp = baseline['category_breakdown']['VP']['accuracy']
    baseline_vi = baseline['category_breakdown']['VI']['accuracy']
    
    sdu_vp = vp_acc
    sdu_vi = vi_acc
    
    print(f"\nMetric Comparison:")
    print(f"{'Metric':<25} {'Baseline':<12} {'SDU':<12} {'Change'}")
    print(f"{'-'*60}")
    
    acc_change = accuracy - baseline_acc
    ce_change = content_effect - baseline_ce
    vp_change = sdu_vp - baseline_vp
    vi_change = sdu_vi - baseline_vi
    
    print(f"{'Overall Accuracy':<25} {baseline_acc:<12.4f} {accuracy:<12.4f} {acc_change:+.4f}")
    print(f"{'Content Effect':<25} {baseline_ce:<12.4f} {content_effect:<12.4f} {ce_change:+.4f}")
    print(f"{'VP Accuracy':<25} {baseline_vp:<12.4f} {sdu_vp:<12.4f} {vp_change:+.4f}")
    print(f"{'VI Accuracy':<25} {baseline_vi:<12.4f} {sdu_vi:<12.4f} {vi_change:+.4f}")
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print("="*80)
    
    if acc_change > 0.015:
        print(f"\n✓ IMPROVEMENT: +{acc_change:.1%} overall accuracy")
    elif acc_change > 0:
        print(f"\n○ MARGINAL: +{acc_change:.1%} overall accuracy")
    else:
        print(f"\n✗ NO IMPROVEMENT: {acc_change:.1%} overall accuracy")
    
    if vp_change > 0.02:
        print(f"✓ VP improved: +{vp_change:.1%}")
    
    if abs(ce_change) > 0.01:
        if abs(content_effect) < abs(baseline_ce):
            print(f"✓ Bias reduced: CE {baseline_ce:.4f} → {content_effect:.4f}")
    
except FileNotFoundError:
    print("\n⚠ No baseline results found for comparison")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
results = {
    'model': 'SDU Model (Expanded 286-Example Test)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'test_set_size': len(test_data),
    'metrics': {
        'overall_accuracy': accuracy,
        'official_test_accuracy': official_acc,
        'unused_train_accuracy': unused_acc,
        'content_effect': content_effect,
        'vp_vi_gap': vp_vi_gap,
    },
    'category_breakdown': {
        cat: {
            'accuracy': sum(results)/len(results) if results else 0,
            'count': len(results),
        } 
        for cat, results in categories.items()
    },
    'predictions': predictions,
}

output_file = 'sdu_final_results_expanded.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
print(f"\nEvaluation time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"\nFinal Results:")
print(f"  Baseline:  81.5%")
print(f"  SDU:       {accuracy:.1%}")
print(f"  Change:    {acc_change:+.1%}")
print("="*80)