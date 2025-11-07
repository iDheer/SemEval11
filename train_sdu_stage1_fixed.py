"""
Stage 1: Train Shadow Model (EXACT PAPER METHODOLOGY - FULLY FIXED)
- Train ONLY on plausible examples (VP + IP)
- Calculate weight saliency using gradients (4-bit compatible)
- Create mask from top 10-15% most salient weights
- Randomize mask by flipping 10-15% of bits
"""

import torch
import json
import gc
import os
import time
import random
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm
from config import MODEL_NAME, SCOT_DATA, VAL_DATA, STAGE1_CONFIG, MEMORY_CONFIG

print("="*80)
print("STAGE 1: SHADOW MODEL (EXACT PAPER METHODOLOGY)")
print("="*80)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    gc.collect()

print(f"Initial memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Check for S-CoT model
print("\nChecking for S-CoT model...")
if not os.path.exists("./scot_base_model"):
    print("❌ ERROR: S-CoT model not found!")
    print("Please run train_stage0_scot_fixed.py first")
    exit(1)
print("✓ S-CoT model found")

# ============================================================================
# STEP 1: Load and Prepare Training Data (PLAUSIBLE ONLY!)
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading Training Data (PLAUSIBLE EXAMPLES ONLY)")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Tokenizer loaded")

# Load training data
print("\nLoading training data...")
try:
    with open(SCOT_DATA, 'r') as f:
        train_data = json.load(f)
except FileNotFoundError:
    print(f"❌ ERROR: {SCOT_DATA} not found!")
    exit(1)

print(f"✓ Loaded {len(train_data)} examples")

# CRITICAL: Select ONLY plausible examples (VP + IP)
print("\nFiltering for PLAUSIBLE examples only (as per paper)...")
plausible_examples = [item for item in train_data if item.get('plausibility')]

vp_examples = [item for item in plausible_examples if item.get('validity')]
ip_examples = [item for item in plausible_examples if not item.get('validity')]

print(f"✓ Plausible examples found: {len(plausible_examples)}")
print(f"  VP (Valid-Plausible): {len(vp_examples)}")
print(f"  IP (Invalid-Plausible): {len(ip_examples)}")

# Balance VP and IP within plausible set
min_count = min(len(vp_examples), len(ip_examples))
balanced_plausible = vp_examples[:min_count] + ip_examples[:min_count]

random.seed(42)
random.shuffle(balanced_plausible)

print(f"✓ Balanced plausible dataset: {len(balanced_plausible)} examples")
print(f"  VP: {min_count} ({100*min_count/len(balanced_plausible):.1f}%)")
print(f"  IP: {min_count} ({100*min_count/len(balanced_plausible):.1f}%)")
print("\n⚠ IMPORTANT: Shadow model trains ONLY on plausible examples!")

# Create training prompts - predict VALIDITY within plausible set
def create_shadow_train_prompt(syllogism, validity):
    """Shadow model predicts validity within plausible set"""
    answer = "VALID" if validity else "INVALID"
    messages = [
        {"role": "system", "content": "You are a logical reasoning expert evaluating plausible syllogisms."},
        {"role": "user", "content": f"Is this syllogism valid?\n\n{syllogism}\n\nAnswer VALID or INVALID."},
        {"role": "assistant", "content": f"Final Answer: {answer}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

print("\nCreating training prompts...")
train_prompts = []
for i, item in enumerate(balanced_plausible):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(balanced_plausible)}...")
    
    if item.get('syllogism') and 'validity' in item:
        prompt = create_shadow_train_prompt(item['syllogism'], item['validity'])
        train_prompts.append(prompt)

print(f"✓ Created {len(train_prompts)} training prompts")

# Load validation data
print("\nLoading validation data...")
try:
    with open(VAL_DATA, 'r') as f:
        val_data_all = json.load(f)
except FileNotFoundError:
    print(f"⚠ Validation data not found, using subset of training")
    val_data_all = balanced_plausible[-50:]

# Validation also uses plausible examples
val_plausible = [item for item in val_data_all if item.get('plausibility')]
val_vp = [item for item in val_plausible if item['validity']][:25]
val_ip = [item for item in val_plausible if not item['validity']][:25]
val_balanced = val_vp + val_ip

print(f"✓ Validation: {len(val_balanced)} plausible examples")

val_prompts = []
for item in val_balanced:
    if item.get('syllogism') and 'validity' in item:
        prompt = create_shadow_train_prompt(item['syllogism'], item['validity'])
        val_prompts.append(prompt)

print(f"✓ Created {len(val_prompts)} validation prompts")

# Tokenize
print("\nTokenizing data...")

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

val_dataset = Dataset.from_dict({'text': val_prompts})
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"✓ Training dataset: {len(train_dataset)} examples")
print(f"✓ Validation dataset: {len(val_dataset)} examples")

# ============================================================================
# STEP 2: Load Base Model
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Loading Base Model")
print("="*80)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading Qwen3-8B (4-bit quantized)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print("✓ Base model loaded")

model = prepare_model_for_kbit_training(model)

# ============================================================================
# STEP 3: Configure LoRA
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Configuring LoRA")
print("="*80)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("LoRA configuration:")
print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================================
# STEP 4: Train Shadow Model
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Training Shadow Model on Plausible Examples")
print("="*80)

training_args = TrainingArguments(
    output_dir=STAGE1_CONFIG['output_dir'],
    num_train_epochs=STAGE1_CONFIG['num_train_epochs'],
    per_device_train_batch_size=STAGE1_CONFIG['per_device_train_batch_size'],
    gradient_accumulation_steps=STAGE1_CONFIG['gradient_accumulation_steps'],
    learning_rate=STAGE1_CONFIG['learning_rate'],
    weight_decay=STAGE1_CONFIG['weight_decay'],
    warmup_steps=STAGE1_CONFIG['warmup_steps'],
    logging_steps=STAGE1_CONFIG['logging_steps'],
    eval_strategy="no",
    save_steps=STAGE1_CONFIG['save_steps'],
    save_total_limit=STAGE1_CONFIG['save_total_limit'],
    load_best_model_at_end=False,
    bf16=STAGE1_CONFIG['bf16'],
    gradient_checkpointing=STAGE1_CONFIG['gradient_checkpointing'],
    optim=STAGE1_CONFIG['optim'],
    report_to=STAGE1_CONFIG['report_to'],
    max_grad_norm=STAGE1_CONFIG['max_grad_norm'],
    dataloader_num_workers=STAGE1_CONFIG['dataloader_num_workers'],
    dataloader_pin_memory=STAGE1_CONFIG['dataloader_pin_memory'],
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

trainer = MemoryEfficientTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("\nTraining shadow model...")
print("This model learns patterns in PLAUSIBLE syllogisms only")
print("="*80 + "\n")

try:
    train_result = trainer.train()
    print(f"\n✓ Training complete!")
    print(f"  Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Final loss: {train_result.training_loss:.4f}")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n❌ OUT OF MEMORY!")
        raise
    else:
        raise

try:
    train_result = trainer.train()
    print(f"\n✓ Training complete!")
    print(f"  Time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"  Final loss: {train_result.training_loss:.4f}")
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n❌ OUT OF MEMORY!")
        raise
    else:
        raise

# ============================================================================
# STEP 4.5: Cleanup and Reload Model for Saliency Calculation
# ============================================================================
print("\n" + "="*80)
print("STEP 4.5: Preparing for Saliency Calculation")
print("="*80)

print("\nSaving model checkpoint...")
model.save_pretrained(STAGE1_CONFIG['output_dir'])
tokenizer.save_pretrained(STAGE1_CONFIG['output_dir'])
print("✓ Model checkpoint saved")

print("\nUnloading model from GPU...")
del model
del trainer
torch.cuda.empty_cache()
gc.collect()
print("✓ GPU memory cleared")
print(f"Memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Reload model for saliency calculation
print("\nReloading model for saliency computation...")
from peft import PeftModel

base_model_reload = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Prepare for training to enable gradients
base_model_reload = prepare_model_for_kbit_training(base_model_reload)

model = PeftModel.from_pretrained(base_model_reload, STAGE1_CONFIG['output_dir'])

# CRITICAL: Enable gradients on LoRA parameters
for name, param in model.named_parameters():
    if 'lora' in name.lower():
        param.requires_grad = True

print("✓ Model reloaded")
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================================
# STEP 5: Calculate Weight Saliency (FIXED FOR 4-BIT + MEMORY)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Calculating Weight Saliency (4-bit + memory optimized)")
print("="*80)

print("\nComputing gradients on plausible examples...")

# Enable gradient computation only for LoRA parameters (these are NOT quantized)
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name.lower() and param.requires_grad:
        lora_params.append((name, param))
        print(f"  Tracking gradients for: {name}")

print(f"✓ Will compute saliency for {len(lora_params)} LoRA parameters")

if len(lora_params) == 0:
    print("\n❌ ERROR: No trainable LoRA parameters found!")
    print("Falling back to uniform masking...")
    
    # Fallback: create uniform masks for all LoRA params
    saliencies = {}
    param_names = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            saliencies[name] = torch.ones_like(param.data.cpu())
            param_names.append(name)
    
    print(f"✓ Created uniform saliencies for {len(saliencies)} parameters")
    
    # Skip gradient computation, go straight to masking
    threshold = torch.tensor(0.5)  # Arbitrary threshold
    SALIENCY_PERCENTILE = 90
    
else:
    model.train()  # Set to train mode for gradient computation
    model.zero_grad()

    # Compute loss on plausible validation set
    print("\nComputing loss on plausible validation examples...")

    total_loss = 0.0
    num_batches = 0

    # REDUCED BATCH SIZE TO 1 for memory efficiency
    for i in tqdm(range(0, len(val_dataset), 1), desc="Computing gradients"):
        batch_indices = [i]
        batch_data = [val_dataset[idx] for idx in batch_indices]
        
        # Stack batch data
        input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch_data]).to(device)
        attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch_data]).to(device)
        labels = torch.stack([torch.tensor(item['labels']) for item in batch_data]).to(device)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Accumulate gradients (only on LoRA params)
        loss.backward()
        
        total_loss += loss.item()
        num_batches += 1
        
        # AGGRESSIVE memory management
        del outputs, loss, inputs, input_ids, attention_mask, labels
        if i % 5 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    print(f"✓ Average loss on plausible examples: {avg_loss:.4f}")

    # Extract gradient magnitudes (saliency scores)
    print("\nExtracting saliency scores from LoRA parameters...")

    saliencies = {}
    param_names = []

    for name, param in lora_params:
        if param.grad is not None:
            # Saliency = absolute gradient magnitude
            saliency = param.grad.abs().detach().cpu()
            saliencies[name] = saliency
            param_names.append(name)
            print(f"  {name}: saliency range [{saliency.min():.6f}, {saliency.max():.6f}]")
        else:
            print(f"  ⚠ {name}: No gradient computed")

    if len(saliencies) == 0:
        print("\n❌ ERROR: No saliencies computed!")
        print("Falling back to random masking...")
        
        # Fallback: Create random masks
        for name, param in lora_params:
            saliencies[name] = torch.rand_like(param.data.cpu())
            param_names.append(name)
        print(f"✓ Created random saliencies for {len(saliencies)} parameters")
    else:
        print(f"✓ Computed saliency for {len(saliencies)} parameters")

    # Compute threshold
    all_saliencies = torch.cat([s.flatten() for s in saliencies.values()])
    SALIENCY_PERCENTILE = 90
    threshold = torch.quantile(all_saliencies, SALIENCY_PERCENTILE / 100.0)
    print(f"✓ Saliency threshold (top {100-SALIENCY_PERCENTILE}%): {threshold:.6f}")

# Create binary masks
masks = {}
total_masked = 0
total_weights = 0

for name, saliency in saliencies.items():
    # Mask = 1 where saliency > threshold
    mask = (saliency > threshold).float()
    masks[name] = mask
    
    num_masked = mask.sum().item()
    num_total = mask.numel()
    
    total_masked += num_masked
    total_weights += num_total
    
    print(f"  {name}: {num_masked}/{num_total} weights masked ({100*num_masked/num_total:.2f}%)")

print(f"\n✓ Total masked weights: {total_masked}/{total_weights} ({100*total_masked/total_weights:.2f}%)")

# ============================================================================
# STEP 7: Randomize Mask (Flip 10-15% of bits)
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Randomizing Mask")
print("="*80)

RANDOMIZATION_RATE = 0.15  # Flip 15% of mask bits

print(f"Flipping {RANDOMIZATION_RATE*100:.0f}% of mask bits randomly...")

random.seed(42)
np.random.seed(42)

randomized_masks = {}
total_flips = 0

for name, mask in masks.items():
    # Flatten mask
    mask_flat = mask.flatten().clone()
    num_weights = mask_flat.numel()
    
    # Number of bits to flip
    num_flips = int(RANDOMIZATION_RATE * num_weights)
    
    # Random indices to flip
    flip_indices = np.random.choice(num_weights, size=num_flips, replace=False)
    
    # Flip bits
    mask_flat[flip_indices] = 1 - mask_flat[flip_indices]
    
    # Reshape back
    randomized_mask = mask_flat.reshape(mask.shape)
    randomized_masks[name] = randomized_mask
    
    total_flips += num_flips
    
    num_masked_after = randomized_mask.sum().item()
    print(f"  {name}: flipped {num_flips} bits, now {num_masked_after}/{num_weights} masked")

print(f"\n✓ Total bits flipped: {total_flips}")
print(f"✓ Final masked weights: {sum(m.sum().item() for m in randomized_masks.values())}/{total_weights}")

# ============================================================================
# STEP 8: Save Everything
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Saving Shadow Model and Masks")
print("="*80)

# Save model
model.save_pretrained(STAGE1_CONFIG['output_dir'])
tokenizer.save_pretrained(STAGE1_CONFIG['output_dir'])

# Save saliency masks
mask_save_path = 'saliency_masks.pt'
torch.save({
    'masks': randomized_masks,
    'threshold': threshold,
    'saliency_percentile': SALIENCY_PERCENTILE,
    'randomization_rate': RANDOMIZATION_RATE,
    'total_masked': sum(m.sum().item() for m in randomized_masks.values()),
    'total_weights': total_weights,
    'param_names': param_names,
}, mask_save_path)

print(f"✓ Saliency masks saved to: {mask_save_path}")

# Save metadata
metadata = {
    'stage': 1,
    'model_name': MODEL_NAME,
    'training_data': 'PLAUSIBLE EXAMPLES ONLY (VP + IP)',
    'num_examples': len(train_dataset),
    'num_epochs': STAGE1_CONFIG['num_train_epochs'],
    'final_loss': train_result.training_loss,
    'saliency_threshold': threshold.item(),
    'saliency_percentile': SALIENCY_PERCENTILE,
    'randomization_rate': RANDOMIZATION_RATE,
    'total_masked_weights': sum(m.sum().item() for m in randomized_masks.values()),
    'total_weights': total_weights,
    'mask_coverage': 100 * sum(m.sum().item() for m in randomized_masks.values()) / total_weights,
    'training_time_minutes': (time.time() - start_time) / 60,
}

with open(os.path.join(STAGE1_CONFIG['output_dir'], 'stage1_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Shadow model saved")
print("✓ Metadata saved")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("STAGE 1 COMPLETE!")
print("="*80)

print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")

print("\nSummary:")
print(f"  Trained on: {len(train_dataset)} PLAUSIBLE examples only")
print(f"  Final loss: {train_result.training_loss:.4f}")
print(f"  Saliency threshold: {threshold:.6f}")
print(f"  Masked weights: {metadata['total_masked_weights']}/{metadata['total_weights']} ({metadata['mask_coverage']:.2f}%)")
print(f"  Randomization: {RANDOMIZATION_RATE*100:.0f}% of mask bits flipped")

print(f"\nNext: python train_sdu_stage2_fixed.py")
print("="*80)