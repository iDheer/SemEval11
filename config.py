"""
Configuration for SDU-based bias mitigation with Qwen3-8B
Following Liu et al. 2024 exact methodology (3 stages)
Optimized for 8GB VRAM using 4-bit quantization
"""

# Model configuration
MODEL_NAME = "Qwen/Qwen3-8B"
TARGET_LAYERS = [15, 18, 20, 23]  # Middle-to-late layers for bias extraction

# Data paths
TRAIN_DATA = "data_splits/train_split.json"
VAL_DATA = "data_splits/val_split.json"
TEST_DATA = "data_splits/test_split.json"
SCOT_DATA = "enriched_training_data_gemini.json"

# Training hyperparameters - Stage 0 (S-CoT Pre-training)
# Train model on S-CoT reasoning first (will have bias)
STAGE0_CONFIG = {
    'output_dir': './scot_base_model',
    'num_train_epochs': 3,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 16,
    'learning_rate': 2e-4,
    'weight_decay': 0.01,
    'warmup_steps': 50,
    'logging_steps': 10,
    'save_steps': 100,
    'save_total_limit': 2,
    'bf16': True,
    'gradient_checkpointing': True,
    'optim': 'paged_adamw_8bit',
    'report_to': 'none',
    'max_grad_norm': 1.0,
    'dataloader_num_workers': 0,
    'dataloader_pin_memory': False,
}

# Training hyperparameters - Stage 1 (Shadow Model)
# Train on plausible examples only to identify bias
STAGE1_CONFIG = {
    'output_dir': './shadow_model_qwen',
    'num_train_epochs': 2,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'warmup_steps': 30,
    'logging_steps': 10,
    'save_steps': 100,
    'save_total_limit': 2,
    'bf16': True,
    'gradient_checkpointing': True,
    'optim': 'paged_adamw_8bit',
    'report_to': 'none',
    'max_grad_norm': 1.0,
    'dataloader_num_workers': 0,
    'dataloader_pin_memory': False,
}

# Training hyperparameters - Stage 2 (SDU Application)
# Apply SDU corrections to S-CoT model
STAGE2_CONFIG = {
    'output_dir': './sdu_model_qwen_final',
    'num_train_epochs': 2,  # Shorter since model already knows S-CoT
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 16,
    'learning_rate': 5e-5,  # Lower LR since fine-tuning existing model
    'weight_decay': 0.01,
    'warmup_steps': 20,
    'logging_steps': 10,
    'eval_strategy': 'steps',
    'eval_steps': 100,
    'save_steps': 100,
    'save_total_limit': 2,
    'load_best_model_at_end': False,
    'bf16': True,
    'gradient_checkpointing': True,
    'optim': 'paged_adamw_8bit',
    'report_to': 'none',
    'max_grad_norm': 1.0,
    'dataloader_num_workers': 0,
    'dataloader_pin_memory': False,
}

# SDU parameters
SDU_CONFIG = {
    'k_std_threshold': 2.0,
    'lambda_K': 0.01,
    'randomization_ratio': 0.15,
}

# Memory optimization settings
MEMORY_CONFIG = {
    'max_length': 384,  # Shorter sequences to save memory
    'saliency_batch_size': 1,
    'activation_batch_size': 1,
    'clear_cache_frequency': 5,
}