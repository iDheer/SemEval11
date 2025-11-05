"""
Stage 1: Train Shadow Model (COMPLETE AGGRESSIVE VERSION)
Matches Stage 0 format, learns plausibility patterns aggressively
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
from config import MODEL_NAME, TRAIN_DATA, VAL_DATA, STAGE1_CONFIG, MEMORY_CONFIG

print("="*80)
print("STAGE 1: SHADOW MODEL TRAINING (COMPLETE AGGRESSIVE VERSION)")
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

def create_prompt(syllogism, validity, include_response=False):
    """Create prompt matching Stage 0 format exactly"""
    messages = [
        {"role": "system", "content": "You are a logical reasoning expert."},
        {"role": "user", "content": f"Is this syllogism valid?\n\n{syllogism}\n\nAnswer VALID or INVALID."}
    ]
    
    if include_response:
        answer = "VALID" if validity else "INVALID"
        messages.append({"role": "assistant", "content": f"Final Answer: {answer}"})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Load training data
print("\nLoading training data...")
try:
    with open(TRAIN_DATA, 'r') as f:
        train_data = json.load(f)
    print(f"✓ Loaded {len(train_data)} examples")
except FileNotFoundError:
    print(f"❌ ERROR: {TRAIN_DATA} not found!")
    exit(1)

# Balance the training data AGGRESSIVELY
valid_train = [item for item in train_data if item['validity']]
invalid_train = [item for item in train_data if not item['validity']]

print(f"  VALID examples: {len(valid_train)}")
print(f"  INVALID examples: {len(invalid_train)}")

min_count = min(len(valid_train), len(invalid_train))
balanced_train = valid_train[:min_count] + invalid_train[:min_count]

# Shuffle
import random
random.seed(42)
random.shuffle(balanced_train)

print(f"✓ Balanced training: {len(balanced_train)} examples")
print(f"  VALID: {min_count} ({100*min_count/len(balanced_train):.1f}%)")
print(f"  INVALID: {min_count} ({100*min_count/len(balanced_train):.1f}%)")

train_prompts = []
for i, item in enumerate(balanced_train):
    if i % 50 == 0:
        print(f"  Processing {i}/{len(balanced_train)}...")
    prompt = create_prompt(item['syllogism'], item['validity'], include_response=True)
    train_prompts.append(prompt)

print(f"✓ Created {len(train_prompts)} training prompts")

# Load validation data (balanced)
print("\nLoading validation data...")
try:
    with open(VAL_DATA, 'r') as f:
        val_data = json.load(f)
    print(f"✓ Loaded {len(val_data)} validation examples")
except FileNotFoundError:
    print(f"❌ ERROR: {VAL_DATA} not found!")
    exit(1)

val_valid = [item for item in val_data if item['validity']][:25]
val_invalid = [item for item in val_data if not item['validity']][:25]
val_balanced = val_valid + val_invalid

val_prompts = []
for item in val_balanced:
    prompt = create_prompt(item['syllogism'], item['validity'], include_response=True)
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

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"✓ Training dataset: {len(train_dataset)} examples")
print(f"✓ Validation dataset: {len(val_dataset)} examples")

# Clear memory
del train_prompts, val_prompts, train_data
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
model = prepare_model_for_kbit_training(model)

# ============================================================================
# STEP 3: Configure LoRA (AGGRESSIVE - Match Stage 0)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Configuring LoRA (AGGRESSIVE)")
print("="*80)

lora_config = LoraConfig(
    r=8,  # Match Stage 0
    lora_alpha=16,  # Match Stage 0
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("LoRA configuration (matches Stage 0):")
print(f"  Rank (r): {lora_config.r}")
print(f"  Alpha: {lora_config.lora_alpha}")

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================================
# STEP 4: Configure Training (AGGRESSIVE)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Configuring Training (AGGRESSIVE)")
print("="*80)

training_args = TrainingArguments(
    output_dir='./shadow_model_qwen_final',
    num_train_epochs=6,  # INCREASED
    per_device_train_batch_size=STAGE1_CONFIG['per_device_train_batch_size'],
    gradient_accumulation_steps=STAGE1_CONFIG['gradient_accumulation_steps'],
    learning_rate=1e-4,  # INCREASED - match Stage 0
    weight_decay=0.01,
    warmup_steps=40,
    logging_steps=5,
    eval_strategy="no",
    eval_steps=20,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=False,
    bf16=True,
    gradient_checkpointing=True,
    optim='paged_adamw_8bit',
    report_to='none',
    max_grad_norm=1.0,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
)

print(f"Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Learning rate: {training_args.learning_rate}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
# STEP 5: Train
# ============================================================================
print("\n" + "="*80)
print("TRAINING SHADOW MODEL (AGGRESSIVE)")
print("="*80)

print("\nPurpose: Learn plausibility patterns for SDU")
print("This model will capture bias that SDU will remove")
print("="*80 + "\n")

try:
    train_result = trainer.train()
    
    print("\n✓ Training complete!")
    print(f"Training time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"Final loss: {train_result.training_loss:.4f}")
    
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("\n❌ OUT OF MEMORY!")
        raise
    else:
        raise

# ============================================================================
# STEP 6: Save
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving Shadow Model")
print("="*80)

output_dir = "./shadow_model_qwen_final"
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

metadata = {
    'stage': 1,
    'model_name': MODEL_NAME,
    'method': 'Shadow Model (Aggressive)',
    'train_examples': len(train_dataset),
    'epochs': training_args.num_train_epochs,
    'lora_rank': 16,
    'lora_alpha': 32,
    'purpose': 'Identify plausibility bias for SDU',
    'final_loss': train_result.training_loss,
    'training_time_minutes': (time.time() - start_time) / 60,
}

with open(os.path.join(output_dir, 'stage1_metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Shadow model saved")

print("\n" + "="*80)
print("STAGE 1 COMPLETE!")
print("="*80)
print(f"\nTotal time: {(time.time() - start_time) / 60:.1f} minutes")
print("\nNext: python train_sdu_stage2_fixed.py")
print("="*80)