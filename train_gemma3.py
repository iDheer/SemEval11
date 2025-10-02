import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

with open('data_splits/val_split.json', 'r') as f:
    val_data = json.load(f)

ip_examples = [item for item in val_data if not item['validity'] and item['plausibility']]
ii_examples = [item for item in val_data if not item['validity'] and not item['plausibility']]

print(f"IP set (Invalid, Plausible): {len(ip_examples)} examples")
print(f"II set (Invalid, Implausible): {len(ii_examples)} examples")

model_name = "google/gemma-3-4b-it"
print(f"\nUsing model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Use a stable 8-bit model for bias extraction to get a valid vector
bnb_config_8bit = BitsAndBytesConfig(
    load_in_8bit=True,
)

print("Loading STABLE 8-bit base model for bias extraction...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_8bit,
    device_map="auto",
    trust_remote_code=True
)

target_layer_idx = 20
print(f"Target layer for steering: {target_layer_idx}")

captured_activations = []
def activation_hook(module, input, output):
    last_token_activation = output[0][:, -1, :].detach().cpu()
    captured_activations.append(last_token_activation)

hook_handle = base_model.model.language_model.layers[target_layer_idx].register_forward_hook(activation_hook)
print("Hook registered successfully on bias extraction model.")

def extract_activations(examples):
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
    return torch.cat(captured_activations, dim=0)

print("\nExtracting activations for IP set...")
ip_activations = extract_activations(ip_examples)
print("Extracting activations for II set...")
ii_activations = extract_activations(ii_examples)
hook_handle.remove()

print("\nComputing bias direction vector...")
avg_ip = ip_activations.mean(dim=0)
avg_ii = ii_activations.mean(dim=0)
bias_direction = (avg_ip - avg_ii)
bias_direction = F.normalize(bias_direction, dim=0, eps=1e-8)

print(f"Bias vector shape: {bias_direction.shape}")
print(f"Bias vector norm: {bias_direction.norm().item():.4f}")

torch.save(bias_direction, 'bias_direction_vector.pt')
print("Saved bias vector to: bias_direction_vector.pt")

del base_model
torch.cuda.empty_cache()

# ===== STANDARD TRAINING SETUP =====
print("\n" + "="*60)
print("STANDARD TRAINING SETUP")
print("="*60)

with open('student_training_prompts_correct_only.txt', 'r', encoding='utf-8') as f:
    content = f.read()
separator = '=' * 80
examples = content.split(separator)
examples = [ex.strip() for ex in examples if ex.strip()]
dataset_dict = {'text': examples}
dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print("Tokenizing and padding dataset...")
def tokenize_function(examples):
    output = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024
    )
    output["labels"] = output["input_ids"][:]
    return output

train_dataset = dataset['train'].map(tokenize_function, batched=True, remove_columns=["text"])
val_dataset = dataset['test'].map(tokenize_function, batched=True, remove_columns=["text"])
print(f"Training set: {len(train_dataset)} examples, tokenized and padded.")
print(f"Validation set: {len(val_dataset)} examples, tokenized and padded.")

# Switch back to 4-bit for memory-efficient training
bnb_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading 4-bit model for training...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config_4bit,
    device_map="auto",
    trust_remote_code=True,
    # THE FIX: Removed the unsupported `use_cache=False` argument
)
model = prepare_model_for_kbit_training(model)

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

class SteeringTrainer(Trainer):
    def __init__(self, bias_direction, steering_strength, target_layer_idx, **kwargs):
        super().__init__(**kwargs)
        self.bias_direction = bias_direction
        self.steering_strength = steering_strength
        self.target_layer_idx = target_layer_idx
        self.current_activation = None
        self.hook_handle = self.model.base_model.model.language_model.layers[self.target_layer_idx].register_forward_hook(
            self.capture_activation_hook
        )
        print("Hook registered successfully on SteeringTrainer.")

    def capture_activation_hook(self, module, input, output):
        self.current_activation = output[0][:, -1, :].mean(dim=0)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        main_loss = outputs.loss

        is_vector_valid = self.bias_direction is not None and not torch.isnan(self.bias_direction.norm()) and self.bias_direction.norm().item() > 1e-6

        if self.current_activation is not None and is_vector_valid:
            similarity = F.cosine_similarity(
                self.current_activation.unsqueeze(0),
                self.bias_direction.unsqueeze(0),
                dim=1
            )
            steering_loss = similarity.mean() * self.steering_strength
            total_loss = main_loss + steering_loss
        else:
            total_loss = main_loss

        self.current_activation = None
        return (total_loss, outputs) if return_outputs else total_loss

training_args = TrainingArguments(
    output_dir="./student_model_with_steering",
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

bias_direction_cpu = torch.load('bias_direction_vector.pt', weights_only=True)
bias_direction = bias_direction_cpu.to("cuda" if torch.cuda.is_available() else "cpu")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

print("Initializing Steering Trainer...")
trainer = SteeringTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    bias_direction=bias_direction,
    steering_strength=1.0,
    target_layer_idx=target_layer_idx,
)

# ===== TRAINING =====
print("\n" + "="*60)
print("STARTING TRAINING WITH ACTIVATION STEERING")
print("="*60)
print("Training objectives:")
print("1. Main loss: Learn S-CoT reasoning")
print("2. Steering loss: Avoid plausibility bias")
if torch.isnan(bias_direction.norm()) or bias_direction.norm().item() < 1e-6:
    print("\nWARNING: Bias vector is invalid or near-zero. Steering will be disabled.")
else:
    print("\nSUCCESS: Valid bias vector loaded. Steering is enabled.")
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