import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

print("="*60)
print("STUDENT MODEL - NO STEERING (Baseline S-CoT)")
print("="*60)

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load training data
print("\nLoading training data...")
with open('student_training_prompts.txt', 'r', encoding='utf-8') as f:
    content = f.read()

separator = '=' * 80
examples = content.split(separator)
examples = [ex.strip() for ex in examples if ex.strip()]

print(f"Parsed {len(examples)} training examples")

# Create dataset
dataset_dict = {'text': examples}
dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

print(f"Training: {len(train_dataset)}, Validation: {len(val_dataset)}")

# Model setup
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, peft_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")

# Training arguments
training_args = TrainingArguments(
    output_dir="./student_model_no_steering",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    report_to="none",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
)

print("\nInitializing trainer...")
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
)

print("\n" + "="*60)
print("TRAINING WITHOUT ACTIVATION STEERING")
print("="*60)
print("Training on pure S-CoT traces from teacher")
print("No bias intervention - testing if steering was the problem\n")

trainer.train()

print("\n" + "="*60)
print("Training complete!")
print("="*60)

# Save model
final_dir = "./student_model_no_steering_final"
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

print(f"\nModel saved to: {final_dir}")
print("\nNext: Evaluate on validation set and compare:")
print("- Baseline RoBERTa: 69.86% accuracy")
print("- Student with steering (Layer 20): 52.4%")
print("- Student without steering: ?")