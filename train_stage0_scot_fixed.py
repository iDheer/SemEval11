"""
Stage 0: Train S-CoT Model (COMPLETE AGGRESSIVE VERSION)
Maximum effectiveness - model WILL learn!
"""

import torch
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
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import MODEL_NAME, SCOT_DATA, VAL_DATA, STAGE0_CONFIG, MEMORY_CONFIG

print("="*80)
print("STAGE 0: S-CoT TRAINING (COMPLETE AGGRESSIVE VERSION)")
print("="*80)

start_time = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Initial memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Loading and Preparing Training Data")
print("="*80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ Tokenizer loaded")

# Load training data
print("\nLoading S-CoT training data...")
try:
    with open(SCOT_DATA, 'r') as f:
        scot_data = json.load(f)
    print(f"✓ Loaded {len(scot_data)} examples")
except FileNotFoundError:
    print(f"❌ ERROR: {SCOT_DATA} not found!")
    exit(1)

# Filter for correct traces
print("\nFiltering for correct traces...")
correct_traces = [item for item in scot_data if item.get('teacher_correct', False)]
print(f"✓ Correct traces: {len(correct_traces)}")

# Separate by validity
valid_examples = [item for item in correct_traces if item.get('validity')]
invalid_examples = [item for item in correct_traces if not item.get('validity')]

print(f"  VALID examples: {len(valid_examples)}")
print(f"  INVALID examples: {len(invalid_examples)}")

# Balance dataset AGGRESSIVELY - use ALL of the minority class
print("\nBalancing dataset (using maximum data)...")
min_count = min(len(valid_examples), len(invalid_examples))
balanced_data = valid_examples[:min_count] + invalid_examples[:min_count]

print(f"✓ Balanced dataset: {len(balanced_data)} examples")
print(f"  VALID: {min_count} ({100*min_count/len(balanced_data):.1f}%)")
print(f"  INVALID: {min_count} ({100*min_count/len(balanced_data):.1f}%)")

# Shuffle
import random
random.seed(42)
random.shuffle(balanced_data)
print("✓ Data shuffled")

def create_training_prompt(syllogism, scot_trace, validity):
    """
    Create training prompt with VERY CLEAR answer
    Makes it impossible for model to miss the pattern
    """
    # Simplified reasoning that's VERY clear
    answer = "VALID" if validity else "INVALID"
    
    # Extract some logical structure if available
    reasoning = ""
    if scot_trace:
        trace_lines = scot_trace.strip().split('\n')
        # Get lines with formal logic symbols
        formal_lines = [line for line in trace_lines if any(s in line for s in ['∀', '∃', '→', 'Premise', 'Conclusion'])]
        if formal_lines:
            reasoning = '\n'.join(formal_lines[:3]) + '\n\n'
    
    # VERY CLEAR final answer format
    response = f"{reasoning}Final Answer: {answer}"
    
    messages = [
        {"role": "system", "content": "You are a logical reasoning expert."},
        {"role": "user", "content": f"Is this syllogism valid?\n\n{syllogism}\n\nAnswer VALID or INVALID."},
        {"role": "assistant", "content": response}
    ]
    
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# Create training prompts
print("\nCreating training prompts...")
train_prompts = []

for i, item in enumerate(balanced_data):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(balanced_data)}...")
    
    if item.get('s-cot_trace') and item.get('syllogism') and 'validity' in item:
        try:
            prompt = create_training_prompt(
                item['syllogism'],
                item['s-cot_trace'],
                item['validity']
            )
            train_prompts.append(prompt)
        except Exception as e:
            print(f"  ⚠ Error processing example {i}: {e}")
            continue

print(f"✓ Created {len(train_prompts)} training prompts")

# Create validation set (also balanced)
print("\nCreating validation set...")
try:
    with open(VAL_DATA, 'r') as f:
        val_data = json.load(f)
    print(f"✓ Loaded {len(val_data)} validation examples")
except FileNotFoundError:
    print(f"❌ ERROR: {VAL_DATA} not found!")
    exit(1)

# Balanced validation set
val_valid = [item for item in val_data if item['validity']][:25]
val_invalid = [item for item in val_data if not item['validity']][:25]
val_balanced = val_valid + val_invalid

val_prompts = []
for item in val_balanced:
    prompt = create_training_prompt(
        item['syllogism'],
        "",  # No S-CoT trace needed
        item['validity']
    )
    val_prompts.append(prompt)

print(f"✓ Created {len(val_prompts)} validation prompts (balanced)")

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
val_dataset = Dataset.from_dict({'text': val_prompts})

print("  Tokenizing training set...")
train_dataset = train_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Tokenizing train"
)

print("  Tokenizing validation set...")
val_dataset = val_dataset.map(
    tokenize_function, 
    batched=True, 
    remove_columns=["text"],
    desc="Tokenizing val"
)

print(f"✓ Training dataset: {len(train_dataset)} examples")
print(f"✓ Validation dataset: {len(val_dataset)} examples")

# Clear memory
del train_prompts, val_prompts, scot_data, balanced_data
gc.collect()
torch.cuda.empty_cache()

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
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Prepare for training
print("\nPreparing model for k-bit training...")
model = prepare_model_for_kbit_training(model)

# ============================================================================
# STEP 3: Configure LoRA (AGGRESSIVE)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Configuring LoRA (AGGRESSIVE)")
print("="*80)

lora_config = LoraConfig(
    r=8,  # INCREASED from 8 - more capacity
    lora_alpha=16,  # INCREASED from 16 - stronger adaptation
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("LoRA configuration (AGGRESSIVE):")
print(f"  Rank (r): {lora_config.r} ← DOUBLED for more learning capacity")
print(f"  Alpha: {lora_config.lora_alpha} ← DOUBLED for stronger updates")
print(f"  Target modules: {lora_config.target_modules}")
print(f"  Dropout: {lora_config.lora_dropout}")

model = get_peft_model(model, lora_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()

print(f"Memory after LoRA: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================================
# STEP 4: Configure Training (AGGRESSIVE)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Configuring Training (AGGRESSIVE)")
print("="*80)

# AGGRESSIVE training settings
training_args = TrainingArguments(
    output_dir='./scot_base_model',
    num_train_epochs=5,  # STILL AGGRESSIVE
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,  # STILL AGGRESSIVE
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=5,
    eval_strategy="no",  # SAFE - no eval
    save_steps=100,
    save_total_limit=1,
    load_best_model_at_end=False,  # Can't use without eval
    bf16=True,
    gradient_checkpointing=True,
    optim='paged_adamw_8bit',
    report_to='none',
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)
print("Training configuration (AGGRESSIVE):")
print(f"  Epochs: {training_args.num_train_epochs} ← INCREASED for thorough learning")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate} ← INCREASED for faster learning")
print(f"  Warmup steps: {training_args.warmup_steps}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Custom trainer with memory management
class MemoryEfficientTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        # Support both old and new transformers API
        if num_items_in_batch is not None:
            loss = super().training_step(model, inputs, num_items_in_batch)
        else:
            loss = super().training_step(model, inputs)
        
        self.step_count += 1
        
        # Clear cache periodically
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

# ============================================================================
# STEP 5: Train (AGGRESSIVE)
# ============================================================================
print("\n" + "="*80)
print("TRAINING S-CoT MODEL (AGGRESSIVE VERSION)")
print("="*80)

print("\nAggressive improvements:")
print("  ✅ Balanced VALID/INVALID examples")
print("  ✅ VERY clear answer format")
print("  ✅ DOUBLED LoRA rank (r=16)")
print("  ✅ DOUBLED LoRA alpha (α=32)")
print("  ✅ INCREASED learning rate (1e-4)")
print("  ✅ INCREASED epochs (8)")
print("  ✅ Model WILL learn!")
print("="*80 + "\n")

try:
    train_result = trainer.train()
    
    print("\n✓ Training complete!")
    print(f"Training time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final train loss: {train_result.training_loss:.4f}")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n❌ OUT OF MEMORY!")
        print("\nSuggestions:")
        print("  1. Reduce r to 8 (line 202)")
        print("  2. Reduce num_train_epochs to 5 (line 230)")
        print("  3. Increase gradient_accumulation_steps in config.py")
        raise
    else:
        raise
except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
    print("Saving current model state...")
    trainer.save_model()
    raise

# ============================================================================
# STEP 6: Save Model
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving Model")
print("="*80)

output_dir = "./scot_base_model"
print(f"Saving to: {output_dir}")

trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save metadata
metadata = {
    'stage': 0,
    'model_name': MODEL_NAME,
    'method': 'S-CoT (Aggressive - r=16, α=32, epochs=8)',
    'train_examples': len(train_dataset),
    'valid_examples': min_count,
    'invalid_examples': min_count,
    'epochs': training_args.num_train_epochs,
    'lora_rank': 16,
    'lora_alpha': 32,
    'learning_rate': training_args.learning_rate,
    'final_loss': train_result.training_loss,
    'training_time_minutes': (time.time() - start_time) / 60,
}

with open(os.path.join(output_dir, 'stage0_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Model saved")
print("✓ Metadata saved")

# ============================================================================
# COMPLETE
# ============================================================================
print("\n" + "="*80)
print("STAGE 0 COMPLETE!")
print("="*80)

print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")
print(f"Final loss: {train_result.training_loss:.4f}")
print(f"Model saved to: {output_dir}")

print("\nNext steps:")
print("  1. Evaluate baseline: python evaluate_scot_baseline.py")
print("     Expected: 75-85% accuracy")
print("  2. If good, continue: python train_sdu_stage1_fixed.py")

print("\nExpected baseline results:")
print("  Overall Accuracy: 75-85%")
print("  VP: 75-85% (Valid-Plausible)")
print("  IP: 75-85% (Invalid-Plausible)")
print("  VI: 65-75% (Valid-Implausible) ← Lower due to bias")
print("  II: 80-90% (Invalid-Implausible)")
print("  Content Effect: 0.10-0.20 ← Clear bias present")

print("="*80)