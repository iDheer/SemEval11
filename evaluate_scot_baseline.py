"""
Evaluate S-CoT Baseline Model (COMPLETE FIXED VERSION)
Tests the Stage 0 model before SDU is applied
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
print("EVALUATING S-CoT BASELINE (FIXED VERSION)")
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

# Check data balance
valid_count = sum(1 for item in test_data if item['validity'])
plaus_count = sum(1 for item in test_data if item['plausibility'])

print(f"\nTest set composition:")
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
    
    # Extract assistant response (CRITICAL FIX)
    if "<|im_start|>assistant" in response:
        assistant_response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0]
    elif "assistant\n" in response:
        assistant_response = response.split("assistant\n")[-1]
    else:
        # Fallback: extract after user prompt
        if "Answer VALID or INVALID." in response:
            assistant_response = response.split("Answer VALID or INVALID.")[-1]
        else:
            assistant_response = response
    
    assistant_response = assistant_response.strip()
    response_upper = assistant_response.upper()
    
    # Parse prediction (check INVALID/FALSE first - more specific)
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
print("S-CoT BASELINE RESULTS")
print("="*80)

print(f"\nOverall Performance:")
print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(valid_preds)})")
print(f"  Unparseable: {len(predictions) - len(valid_preds)}")

if len(predictions) - len(valid_preds) > len(predictions) * 0.1:
    print(f"  ⚠ WARNING: More than 10% unparseable!")

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

# Identify problem areas
print(f"\n{'='*80}")
print("PROBLEM ANALYSIS")
print("="*80)

vp_acc = sum(categories['VP'])/len(categories['VP']) if categories['VP'] else 0
vi_acc = sum(categories['VI'])/len(categories['VI']) if categories['VI'] else 0

if vp_acc - vi_acc > 0.15:
    print("\n⚠ ISSUE: Strong plausibility bias detected!")
    print(f"  VP accuracy ({vp_acc:.2%}) much higher than VI accuracy ({vi_acc:.2%})")
    print(f"  The model struggles with valid-implausible syllogisms")
    print(f"  → SDU should help with this!")
elif accuracy < 0.65:
    print("\n⚠ ISSUE: Low overall accuracy!")
    print(f"  Model may not have learned the task properly")
    print(f"  → Consider retraining Stage 0 with more epochs")
else:
    print("\n✓ Model performance looks reasonable")
    print(f"  Ready for SDU application")

# Sample predictions
print("\n" + "="*80)
print("SAMPLE PREDICTIONS")
print("="*80)

# Show diverse examples
sample_indices = {
    'VP': next((i for i, p in enumerate(valid_preds) if p['true'] and p['plaus']), None),
    'IP': next((i for i, p in enumerate(valid_preds) if not p['true'] and p['plaus']), None),
    'VI': next((i for i, p in enumerate(valid_preds) if p['true'] and not p['plaus']), None),
    'II': next((i for i, p in enumerate(valid_preds) if not p['true'] and not p['plaus']), None),
}

for cat, idx in sample_indices.items():
    if idx is not None:
        p = valid_preds[idx]
        print(f"\n{cat} Example:")
        print(f"  Syllogism: {p['syllogism'][:80]}...")
        print(f"  True: {'VALID' if p['true'] else 'INVALID'}")
        print(f"  Predicted: {'VALID' if p['pred'] else 'INVALID'}")
        print(f"  Correct: {'✓' if p['pred'] == p['true'] else '✗'}")
        print(f"  Response: {p['response'][:100]}...")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving Results")
print("="*80)

results = {
    'model': 'S-CoT Baseline (Stage 0)',
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'evaluation_time_minutes': (time.time() - start_time) / 60,
    'metrics': {
        'accuracy': accuracy,
        'plausible_acc': plaus_acc,
        'implausible_acc': implaus_acc,
        'content_effect': content_effect,
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

output_file = 'scot_baseline_results.json'
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

print("\nKey Findings:")
print(f"  Overall Accuracy: {accuracy:.1%}")
print(f"  Content Effect: {content_effect:.4f}")
print(f"  VP-VI Gap: {(vp_acc - vi_acc):.4f}")

if accuracy >= 0.70 and content_effect > 0.05:
    print("\n✓ GOOD: Model works but shows bias")
    print("  → Proceed to Stage 1: python train_sdu_stage1_fixed.py")
elif accuracy >= 0.70:
    print("\n✓ EXCELLENT: Model works with minimal bias!")
    print("  → You might not need SDU, but can still try it")
else:
    print("\n⚠ WARNING: Low accuracy")
    print("  → Consider retraining Stage 0 before proceeding")

print("="*80)