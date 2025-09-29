import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import json
import os

print("="*60)
print("PHASE 3B: STUDENT MODEL WITH ACTIVATION STEERING")
print("="*60)

# Check GPU
print(f"\nGPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ===== ACTIVATION STEERING: PHASE 1 - EXTRACT BIAS VECTOR =====
print("\n" + "="*60)
print("ACTIVATION STEERING PHASE 1: Extracting Bias Vector")
print("="*60)

# Load validation data to create contrastive sets
print("Loading validation data...")
with open('data_splits/val_split.json', 'r') as f:
    val_data = json.load(f)

# Create IP (Invalid, Plausible) and II (Invalid, Implausible) sets
ip_examples = [item for item in val_data if not item['validity'] and item['plausibility']]
ii_examples = [item for item in val_data if not item['validity'] and not item['plausibility']]

print(f"IP set (Invalid, Plausible): {len(ip_examples)} examples")
print(f"II set (Invalid, Implausible): {len(ii_examples)} examples")

# Model configuration
model_name = "google/gemma-2-2b"
print(f"\nUsing model: {model_name}")

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load BASE model (before any fine-tuning)
print("Loading base model for bias extraction...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Target layer for activation extraction (middle-to-late layer)
target_layer_idx = 20  # Gemma-2-2b has ~26 layers, use layer 20
print(f"Target layer for steering: {target_layer_idx}")

# Hook to capture activations
captured_activations = []

def activation_hook(module, input, output):
    # Capture the last token's activation (right before generation starts)
    last_token_activation = output[0][:, -1, :].detach().cpu()
    captured_activations.append(last_token_activation)

# Register hook on target layer
hook_handle = base_model.model.layers[target_layer_idx].register_forward_hook(activation_hook)

def extract_activations(examples):
    """Extract activations for a set of examples"""
    global captured_activations
    captured_activations = []
    
    base_model.eval()
    with torch.no_grad():
        for item in examples:
            prompt = f"""### Instruction:
Analyze the following syllogism and determine its validity based purely on its structure.

### Input:
{item['syllogism']}

### Response:"""
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}
            _ = base_model(**inputs)
    
    activations = torch.cat(captured_activations, dim=0)
    return activations

# Extract activations for both sets
print("\nExtracting activations for IP set...")
ip_activations = extract_activations(ip_examples)

print("Extracting activations for II set...")
ii_activations = extract_activations(ii_examples)

# Remove hook
hook_handle.remove()

# Compute bias direction vector
print("\nComputing bias direction vector...")
avg_ip = ip_activations.mean(dim=0)
avg_ii = ii_activations.mean(dim=0)
bias_direction = (avg_ip - avg_ii).to(base_model.device)
bias_direction = F.normalize(bias_direction, dim=0)

print(f"Bias vector shape: {bias_direction.shape}")
print(f"Bias vector norm: {bias_direction.norm().item():.4f}")

torch.save(bias_direction, 'bias_direction_vector.pt')
print("Saved bias vector to: bias_direction_vector.pt")

# Clean up base model
del base_model
torch.cuda.empty_cache()

# ===== STANDARD TRAINING SETUP =====
print("\n" + "="*60)
print("STANDARD TRAINING SETUP")
print("="*60)

# Load and parse training data
print("Loading training data...")
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

print(f"Training set: {len(train_dataset)} examples")
print(f"Validation set: {len(val_dataset)} examples")

# Load model for training
print("\nLoading model for training...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

# Configure LoRA
print("Configuring LoRA...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, peft_config)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# ===== CUSTOM TRAINER WITH STEERING =====
print("\n" + "="*60)
print("ACTIVATION STEERING PHASE 2: Custom Training")
print("="*60)

steering_strength = 1.0
print(f"Steering strength: {steering_strength}")

class SteeringTrainer(SFTTrainer):
    def __init__(self, bias_direction, steering_strength, target_layer_idx, **kwargs):
        super().__init__(**kwargs)
        self.bias_direction = bias_direction
        self.steering_strength = steering_strength
        self.target_layer_idx = target_layer_idx
        self.current_activation = None
        
        # Register hook
        self.hook_handle = self.model.base_model.model.model.layers[self.target_layer_idx].register_forward_hook(
            self.capture_activation_hook
        )
    
    def capture_activation_hook(self, module, input, output):
        self.current_activation = output[0][:, -1, :].mean(dim=0)
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        outputs = model(**inputs)
        main_loss = outputs.loss
        
        if self.current_activation is not None:
            similarity = F.cosine_similarity(
                self.current_activation.unsqueeze(0),
                self.bias_direction.unsqueeze(0),
                dim=1
            )
            steering_loss = similarity.abs() * self.steering_strength
            total_loss = main_loss + steering_loss
            self.current_activation = None
        else:
            total_loss = main_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def __del__(self):
        if hasattr(self, 'hook_handle'):
            self.hook_handle.remove()

# Training arguments
training_args = TrainingArguments(
    output_dir="./student_model_with_steering",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced for safety
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,  # Increased to maintain effective batch size
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

print("Initializing Steering Trainer...")
trainer = SteeringTrainer(
    bias_direction=bias_direction,
    steering_strength=steering_strength,
    target_layer_idx=target_layer_idx,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=peft_config,
    
)

# ===== TRAINING =====
print("\n" + "="*60)
print("STARTING TRAINING WITH ACTIVATION STEERING")
print("="*60)
print("Training objectives:")
print("1. Main loss: Learn S-CoT reasoning")
print("2. Steering loss: Avoid plausibility bias")
print("\nEstimated time: 2-3 hours...")

trainer.train()

print("\n" + "="*60)
print("Training complete!")
print("="*60)

# ===== SAVE MODEL =====
print("\nSaving model...")
final_model_dir = "./student_model_steered_final"
trainer.model.save_pretrained(final_model_dir)
tokenizer.save_pretrained(final_model_dir)

print(f"\nModel saved to: {final_model_dir}")
print("\nNext: Evaluate on test set")
print("Baseline - Accuracy: 0.6986, Content Effect: 0.1014")