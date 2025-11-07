"""
Stage 2: Apply SDU (EXACT PAPER METHODOLOGY)
- Extract bias direction (VP vs VI activations)
- Load saliency masks from Stage 1
- Apply corrections ONLY to masked weights
- Two-phase: fine-tuning + SDU corrections
"""

import torch
import torch.nn.functional as F
import json
import gc
import os
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
from config import (
    MODEL_NAME, VAL_DATA, SCOT_DATA, SDU_CONFIG, TARGET_LAYERS, MEMORY_CONFIG,
    STAGE1_CONFIG  
)

print("="*80)
print("STAGE 2: SDU APPLICATION (EXACT PAPER METHODOLOGY)")
print("="*80)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()

# Check for required files
print("\nChecking for required models and masks...")
if not os.path.exists(STAGE1_CONFIG['output_dir']):  # <-- USE THE CONFIG
    print(f"❌ ERROR: Shadow model not found at {STAGE1_CONFIG['output_dir']}!")
    exit(1)
if not os.path.exists("./scot_base_model"):
    print("❌ ERROR: S-CoT model not found!")
    exit(1)
if not os.path.exists("saliency_masks.pt"):
    print("❌ ERROR: Saliency masks not found!")
    print("Please run train_sdu_stage1_fixed.py first")
    exit(1)
print("✓ All required files found")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

def create_prompt(syllogism):
    messages = [
        {"role": "system", "content": "You are a logical reasoning expert."},
        {"role": "user", "content": f"Is this syllogism valid?\n\n{syllogism}\n\nAnswer VALID or INVALID."}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ============================================================================
# STEP 1: Load Saliency Masks
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Saliency Masks")
print("="*80)

mask_data = torch.load('saliency_masks.pt')
saliency_masks = mask_data['masks']

print(f"✓ Loaded saliency masks")
print(f"  Threshold: {mask_data['threshold']:.6f}")
print(f"  Percentile: {mask_data['saliency_percentile']}%")
print(f"  Randomization rate: {mask_data['randomization_rate']*100:.0f}%")
print(f"  Total masked weights: {mask_data['total_masked']}/{mask_data['total_weights']}")
print(f"  Coverage: {100*mask_data['total_masked']/mask_data['total_weights']:.2f}%")

# ============================================================================
# STEP 2: Load Shadow Model and Extract Bias Direction
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading Shadow Model")
print("="*80)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print("Loading shadow model adapters...")
shadow_model = PeftModel.from_pretrained(base_model, STAGE1_CONFIG['output_dir']) # <-- USE THE CONFIG
shadow_model.eval()
print("✓ Shadow model loaded")

# ============================================================================
# STEP 3: Extract Bias Direction (VP vs VI)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Computing Bias Direction")
print("="*80)

with open(VAL_DATA, 'r') as f:
    val_data = json.load(f)

# Get examples for bias signal
vp_examples = [item for item in val_data if item['validity'] and item['plausibility']][:15]
vi_examples = [item for item in val_data if item['validity'] and not item['plausibility']][:15]

print(f"VP (Valid-Plausible): {len(vp_examples)}")
print(f"VI (Valid-Implausible): {len(vi_examples)}")

captured_activations = {layer_idx: [] for layer_idx in TARGET_LAYERS}

def activation_hook_factory(layer_idx):
    def hook(module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        act = hidden_states[:, -1, :].detach().cpu()
        captured_activations[layer_idx].append(act)
    return hook

print("\nRegistering hooks...")
hooks = []
for layer_idx in TARGET_LAYERS:
    try:
        layer = shadow_model.base_model.model.model.layers[layer_idx]
        hook = layer.register_forward_hook(activation_hook_factory(layer_idx))
        hooks.append(hook)
    except AttributeError:
        pass

def extract_activations(examples, desc):
    global captured_activations
    captured_activations = {layer_idx: [] for layer_idx in TARGET_LAYERS}
    
    shadow_model.eval()
    with torch.no_grad():
        for item in tqdm(examples, desc=desc):
            prompt = create_prompt(item['syllogism'])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MEMORY_CONFIG['max_length'])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = shadow_model(**inputs)
    
    avg_acts = {}
    for layer_idx in TARGET_LAYERS:
        if len(captured_activations[layer_idx]) > 0:
            avg_acts[layer_idx] = torch.cat(captured_activations[layer_idx], dim=0).mean(dim=0)
        else:
            avg_acts[layer_idx] = torch.zeros(shadow_model.config.hidden_size)
    return avg_acts

print("Extracting VP activations...")
vp_activations = extract_activations(vp_examples, "VP")

print("Extracting VI activations...")
vi_activations = extract_activations(vi_examples, "VI")

# Compute bias direction
print("\nComputing bias direction (Δz = VP - VI)...")
delta_z = {}
for layer_idx in TARGET_LAYERS:
    delta_z[layer_idx] = (vp_activations[layer_idx] - vi_activations[layer_idx]).to(device)
    raw_norm = delta_z[layer_idx].norm().item()
    print(f"Layer {layer_idx} raw Δz norm: {raw_norm:.4f}")
    
    if raw_norm > 0:
        delta_z[layer_idx] = F.normalize(delta_z[layer_idx], dim=0)

torch.save(delta_z, 'delta_z_bias_direction.pt')
print("✓ Bias direction saved")

for hook in hooks:
    hook.remove()

del vp_activations, vi_activations, shadow_model, base_model
torch.cuda.empty_cache()
gc.collect()

# ============================================================================
# STEP 4: Load S-CoT Model
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Loading S-CoT Model")
print("="*80)

scot_base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

scot_base = prepare_model_for_kbit_training(scot_base)

sdu_model = PeftModel.from_pretrained(scot_base, "./scot_base_model")
print("✓ S-CoT model loaded")

# ============================================================================
# STEP 5: Prepare Training Data
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Preparing Training Data")
print("="*80)

# Load training data
print("\nLoading S-CoT training data...")
try:
    with open(SCOT_DATA, 'r') as f:
        scot_data = json.load(f)
except FileNotFoundError:
    print(f"⚠ S-CoT data not found")
    scot_data = []

if len(scot_data) > 0:
    # Use a subset for continued training (balanced)
    correct_traces = [item for item in scot_data if item.get('teacher_correct', False)]
    
    valid_ex = [item for item in correct_traces if item.get('validity')][:100]
    invalid_ex = [item for item in correct_traces if not item.get('validity')][:100]
    train_subset = valid_ex + invalid_ex
    
    print(f"✓ Using {len(train_subset)} training examples (balanced)")
    
    def create_train_prompt(syllogism, validity):
        answer = "VALID" if validity else "INVALID"
        messages = [
            {"role": "system", "content": "You are a logical reasoning expert."},
            {"role": "user", "content": f"Is this syllogism valid?\n\n{syllogism}\n\nAnswer VALID or INVALID."},
            {"role": "assistant", "content": f"Final Answer: {answer}"}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    train_prompts = []
    for item in train_subset:
        if item.get('syllogism') and 'validity' in item:
            prompt = create_train_prompt(item['syllogism'], item['validity'])
            train_prompts.append(prompt)
    
    def tokenize_function(examples):
        output = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MEMORY_CONFIG['max_length']
        )
        output["labels"] = output["input_ids"][:]
        return output
    
    train_dataset = Dataset.from_dict({'text': train_prompts})
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    print(f"✓ Training dataset: {len(train_dataset)} examples")
else:
    train_dataset = None
    print("⚠ No training data")

# Load validation (small)
val_valid = [item for item in val_data if item['validity']][:20]
val_invalid = [item for item in val_data if not item['validity']][:20]
val_balanced = val_valid + val_invalid

val_prompts = []
for item in val_balanced:
    answer = "VALID" if item['validity'] else "INVALID"
    messages = [
        {"role": "system", "content": "You are a logical reasoning expert."},
        {"role": "user", "content": f"Is this syllogism valid?\n\n{item['syllogism']}\n\nAnswer VALID or INVALID."},
        {"role": "assistant", "content": f"Final Answer: {answer}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    val_prompts.append(prompt)

val_dataset = Dataset.from_dict({'text': val_prompts})
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"✓ Validation dataset: {len(val_dataset)} examples")

# ============================================================================
# STEP 6: Two-Phase SDU Application
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Two-Phase SDU Application")
print("="*80)

print("\nPhase 1: Fine-tuning S-CoT model")
print("Phase 2: Apply masked SDU corrections")
print("="*80)

# ============================================================================
# PHASE 1: Normal Fine-Tuning
# ============================================================================
print("\n" + "="*80)
print("PHASE 1: FINE-TUNING S-CoT MODEL")
print("="*80)

training_args = TrainingArguments(
    output_dir='./sdu_model_qwen_final_temp',
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=10,
    logging_steps=5,
    eval_strategy="no",
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=False,
    bf16=True,
    gradient_checkpointing=True,
    optim='paged_adamw_8bit',
    report_to='none',
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

class MemoryEfficientTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        self.step_count += 1
        
        if self.step_count % MEMORY_CONFIG['clear_cache_frequency'] == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        return loss

if train_dataset:
    trainer = MemoryEfficientTrainer(
        model=sdu_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    print("\nStarting Phase 1 training...")
    print("="*80 + "\n")
    
    phase1_start = time.time()
    
    try:
        train_result = trainer.train()
        print(f"\n✓ Phase 1 complete!")
        print(f"  Time: {(time.time() - phase1_start) / 60:.1f} minutes")
        print(f"  Final loss: {train_result.training_loss:.4f}")
        
        phase1_loss = train_result.training_loss
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ OUT OF MEMORY!")
            raise
        else:
            raise
else:
    print("\n⚠ No training data, skipping Phase 1")
    phase1_loss = None

# ============================================================================
# PHASE 2: Apply Masked SDU Corrections
# ============================================================================
print("\n" + "="*80)
print("PHASE 2: APPLYING MASKED SDU CORRECTIONS")
print("="*80)

print("\nComputing SDU adjustment matrices (MASKED)...")

phase2_start = time.time()

adjustment_matrices = {}
lambda_K = SDU_CONFIG['lambda_K']

# Map parameter names from masks to model parameters
print("\nMapping saliency masks to model parameters...")

for name, param in sdu_model.named_parameters():
    if 'lora' not in name.lower():
        continue
    
    # Find corresponding mask
    mask_found = False
    for mask_name, mask in saliency_masks.items():
        # Match parameter names (may need adjustment based on naming)
        if mask_name in name or name.endswith(mask_name.split('.')[-1]):
            # Extract layer index from parameter name
            layer_match = False
            for layer_idx in TARGET_LAYERS:
                if f"layers.{layer_idx}." in name:
                    layer_match = True
                    
                    if layer_idx not in adjustment_matrices:
                        adjustment_matrices[layer_idx] = {}
                    
                    # Get bias direction for this layer
                    if layer_idx in delta_z:
                        dz = delta_z[layer_idx].float().cpu()
                        W = param.detach().float().cpu()
                        
                        if W.dim() == 2:
                            # Compute correction
                            if W.shape[1] == dz.shape[0]:
                                Wdz = W @ dz
                                delta_K = -lambda_K * torch.outer(Wdz, dz)
                            elif W.shape[0] == dz.shape[0]:
                                Wdz = W.T @ dz
                                delta_K = -lambda_K * torch.outer(dz, Wdz)
                            elif W.shape[1] < dz.shape[0]:
                                dz_trunc = dz[:W.shape[1]]
                                Wdz = W @ dz_trunc
                                delta_K = -lambda_K * torch.outer(Wdz, dz_trunc)
                            elif W.shape[0] < dz.shape[0]:
                                dz_trunc = dz[:W.shape[0]]
                                Wdz = W.T @ dz_trunc
                                delta_K = -lambda_K * torch.outer(dz_trunc, Wdz)
                            else:
                                continue
                            
                            # Clamp correction
                            delta_K = torch.clamp(delta_K, -0.1, 0.1)
                            
                            # Store with mask
                            adjustment_matrices[layer_idx][name] = {
                                'correction': delta_K,
                                'mask': mask.to(delta_K.device)
                            }
                            
                            mask_found = True
                            print(f"  ✓ {name}: correction computed with mask")
                            break
            
            if mask_found:
                break

total_params = sum(len(adj) for adj in adjustment_matrices.values())
print(f"\n✓ Pre-computed {total_params} masked adjustments")

# Apply masked corrections
CORRECTION_STRENGTHS = [0.5, 1.0, 2.0]

print(f"\nApplying MASKED SDU corrections with {len(CORRECTION_STRENGTHS)} strengths...")

correction_stats = []

for strength in CORRECTION_STRENGTHS:
    corrections_applied = 0
    total_correction_norm = 0.0
    masked_weights_corrected = 0
    total_weights_considered = 0
    
    with torch.no_grad():
        for layer_idx in TARGET_LAYERS:
            if layer_idx not in adjustment_matrices:
                continue
            
            for param_name, adj_data in adjustment_matrices[layer_idx].items():
                # Get parameter
                param = None
                for name, p in sdu_model.named_parameters():
                    if name == param_name:
                        param = p
                        break
                
                if param is None:
                    continue
                
                correction = adj_data['correction'].to(param.device)
                mask = adj_data['mask'].to(param.device)
                
                # Ensure shapes match
                if correction.shape == param.shape == mask.shape:
                    # Apply correction ONLY to masked weights
                    masked_correction = correction * mask * strength
                    param.data += masked_correction
                    
                    corrections_applied += 1
                    total_correction_norm += masked_correction.norm().item()
                    masked_weights_corrected += mask.sum().item()
                    total_weights_considered += mask.numel()
    
    avg_correction = total_correction_norm / max(corrections_applied, 1)
    mask_coverage = 100 * masked_weights_corrected / max(total_weights_considered, 1)
    
    print(f"  Strength {strength:3.1f}: {corrections_applied} params, {masked_weights_corrected} weights, avg norm: {avg_correction:.6f}, coverage: {mask_coverage:.2f}%")
    
    correction_stats.append({
        'strength': strength,
        'corrections': corrections_applied,
        'masked_weights': masked_weights_corrected,
        'avg_norm': avg_correction,
        'mask_coverage': mask_coverage
    })

print(f"\n✓ Phase 2 complete!")
print(f"  Time: {(time.time() - phase2_start) / 60:.1f} minutes")

# ============================================================================
# STEP 7: Save SDU Model
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Saving SDU Model")
print("="*80)

output_dir = "./sdu_model_qwen_final"

sdu_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

metadata = {
    'stage': 2,
    'model_name': MODEL_NAME,
    'base_model': 'scot_base_model',
    'shadow_model_training': 'PLAUSIBLE EXAMPLES ONLY',
    'saliency_masking': 'YES (top 10-15% weights)',
    'mask_randomization': f'YES ({mask_data["randomization_rate"]*100:.0f}% flipped)',
    'method': 'Two-phase: Fine-tuning + Masked SDU corrections',
    'phase1_training': train_dataset is not None,
    'phase1_loss': phase1_loss,
    'phase1_epochs': 2,
    'phase2_corrections': correction_stats,
    'target_layers': TARGET_LAYERS,
    'lambda_K': SDU_CONFIG['lambda_K'],
    'training_time_minutes': (time.time() - start_time) / 60,
}

with open(os.path.join(output_dir, 'stage2_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ SDU model saved")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("STAGE 2 COMPLETE!")
print("="*80)

print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")

print("\nSummary:")
if phase1_loss:
    print(f"  Phase 1 (Fine-tuning): loss = {phase1_loss:.4f}")
print(f"  Phase 2 (Masked SDU):")
print(f"    Correction strengths: {len(CORRECTION_STRENGTHS)}")
print(f"    Weights corrected: {correction_stats[-1]['masked_weights']}")
print(f"    Mask coverage: {correction_stats[-1]['mask_coverage']:.2f}%")

print("\nNext: python evaluate_final.py")
print("\nExpected improvements:")
print("  - VI (Valid-Implausible) accuracy should increase")
print("  - Content Effect should decrease significantly")
print("  - Overall accuracy: should maintain or improve")
print("="*80)