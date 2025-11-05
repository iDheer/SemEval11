"""
Evaluate SDU Model (COMPLETE FIXED VERSION)
Tests the final model after SDU is applied and compares to baseline
"""

import torch
import json
import gc
import time
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from config import MODEL_NAME, TEST_DATA, MEMORY_CONFIG

print("="*80)
print("FINAL EVALUATION - SDU MODEL (FIXED VERSION)")
print("="*80)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()

# ============================================================================
# STEP 1: Load Test Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Test Data")
print("="*80)

try:
    with open(TEST_DATA, 'r') as f:
        test_data = json.load(f)
    print(f"✓ Loaded {len(test_data)} test examples")
except FileNotFoundError:
    print(f"❌ ERROR: {TEST_DATA} not found!")
    exit(1)

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
    
    # Extract assistant response (CRITICAL FIX)
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
    
    predictions.append({
        'id': item['id'],
        'syllogism': item['syllogism'],
        'true': item['validity'],
        'pred': pred,
        'plaus': item['plausibility'],
        'response': assistant_response,
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

# ============================================================================
# STEP 5: Display Results
# ============================================================================
print("\n" + "="*80)
print("SDU MODEL RESULTS")
print("="*80)

print(f"\nOverall Performance:")
print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(valid_preds)})")
print(f"  Unparseable: {len(predictions) - len(valid_preds)}")

print(f"\nBy Plausibility:")
print(f"  Plausible Accuracy:   {plaus_acc:.4f}")
print(f"  Implausible Accuracy: {implaus_acc:.4f}")
print(f"  Content Effect:       {content_effect:.4f}")

# Category breakdown
print("\n" + "="*80)
print("CATEGORY BREAKDOWN")
print("="*80)

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

# ============================================================================
# STEP 6: Compare to Baseline
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH S-CoT BASELINE")
print("="*80)

try:
    with open('scot_baseline_results.json', 'r') as f:
        baseline = json.load(f)
    
    baseline_acc = baseline['metrics']['accuracy']
    baseline_ce = baseline['metrics']['content_effect']
    
    baseline_vp = baseline['category_breakdown']['VP']['accuracy']
    baseline_vi = baseline['category_breakdown']['VI']['accuracy']
    
    sdu_vp = sum(categories['VP'])/len(categories['VP']) if categories['VP'] else 0
    sdu_vi = sum(categories['VI'])/len(categories['VI']) if categories['VI'] else 0
    
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
    
    # Calculate improvements
    if baseline_ce != 0:
        ce_reduction = abs(ce_change / baseline_ce * 100)
    else:
        ce_reduction = 0
    
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nContent Effect Reduction: {ce_reduction:.1f}%")
    
    if ce_reduction > 40 and abs(acc_change) < 0.05:
        print("\n🎉 EXCELLENT: Significant bias reduction with maintained accuracy!")
        print(f"   ✓ Content Effect reduced by {ce_reduction:.0f}%")
        print(f"   ✓ Accuracy change: {acc_change:+.1%} (minimal)")
        success_level = "excellent"
    elif ce_reduction > 25 and abs(acc_change) < 0.10:
        print("\n✓ GOOD: Moderate bias reduction!")
        print(f"   ✓ Content Effect reduced by {ce_reduction:.0f}%")
        print(f"   ✓ Accuracy change: {acc_change:+.1%}")
        success_level = "good"
    elif ce_reduction > 10:
        print("\n○ PARTIAL: Some bias reduction")
        print(f"   ○ Content Effect reduced by {ce_reduction:.0f}%")
        print(f"   ○ Accuracy change: {acc_change:+.1%}")
        success_level = "partial"
    else:
        print("\n⚠ MINIMAL: Little to no bias reduction")
        print(f"   ⚠ Content Effect reduced by {ce_reduction:.0f}%")
        print(f"   ⚠ Accuracy change: {acc_change:+.1%}")
        success_level = "minimal"
    
    # Key insight
    print(f"\nKey Improvement:")
    if vi_change > 0.05:
        print(f"  ✓ VI (Valid-Implausible) accuracy improved by {vi_change:+.1%}")
        print(f"    This shows SDU successfully reduced plausibility bias!")
    elif vi_change > 0:
        print(f"  ○ VI (Valid-Implausible) accuracy improved slightly ({vi_change:+.1%})")
    else:
        print(f"  ⚠ VI (Valid-Implausible) accuracy did not improve")
        print(f"    SDU may need stronger corrections")
    
except FileNotFoundError:
    print("\n⚠ No baseline results found")
    print("  Run evaluate_scot_baseline.py to enable comparison")
    success_level = "unknown"
    baseline_acc = None
    baseline_ce = None

# Sample predictions
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

# Show VI examples (most important for bias)
vi_examples = [p for p in valid_preds if p['true'] and not p['plaus']]
if vi_examples:
    print("\nValid-Implausible Examples (key test for bias reduction):")
    for i, p in enumerate(vi_examples[:3]):
        print(f"\nVI Example {i+1}:")
        print(f"  Syllogism: {p['syllogism'][:80]}...")
        print(f"  True: VALID")
        print(f"  Predicted: {'VALID' if p['pred'] else 'INVALID'}")
        print(f"  Correct: {'✓' if p['pred'] else '✗ (bias affected prediction)'}")

# ============================================================================
# STEP 7: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Saving Results")
print("="*80)

results = {
    'model': 'SDU Model (Stage 2)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'evaluation_time_minutes': (time.time() - start_time) / 60,
    'metrics': {
        'accuracy': accuracy,
        'plausible_acc': plaus_acc,
        'implausible_acc': implaus_acc,
        'content_effect': content_effect,
        'unparseable_count': len(predictions) - len(valid_preds),
    },
    'category_breakdown': {
        cat: {
            'accuracy': sum(results)/len(results) if results else 0,
            'count': len(results),
            'correct': sum(results)
        } 
        for cat, results in categories.items()
    },
    'comparison': {
        'baseline_accuracy': baseline_acc,
        'baseline_content_effect': baseline_ce,
        'accuracy_change': acc_change if baseline_acc else None,
        'content_effect_change': ce_change if baseline_ce else None,
        'ce_reduction_percent': ce_reduction if baseline_ce else None,
        'success_level': success_level,
    } if baseline_acc else None,
    'predictions': predictions,
}

output_file = 'final_sdu_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {output_file}")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("FINAL EVALUATION COMPLETE!")
print("="*80)

print(f"\nEvaluation time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"Results saved to: {output_file}")

if baseline_acc:
    print(f"\nFinal Results:")
    print(f"  Accuracy: {baseline_acc:.1%} → {accuracy:.1%} ({acc_change:+.1%})")
    print(f"  Content Effect: {baseline_ce:.4f} → {content_effect:.4f} ({ce_change:+.4f})")
    print(f"  Bias Reduction: {ce_reduction:.0f}%")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)

print("\nYou have successfully:")
print("  ✓ Trained S-CoT model (Stage 0)")
print("  ✓ Trained Shadow model (Stage 1)")
print("  ✓ Applied SDU corrections (Stage 2)")
print("  ✓ Evaluated both baseline and SDU models")

print("\nResults files generated:")
print("  - scot_baseline_results.json (baseline performance)")
print("  - final_sdu_results.json (SDU performance)")

print("="*80)