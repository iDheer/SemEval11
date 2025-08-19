import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import argparse

def format_prompt(example):
    # The model is trained to complete the prompt with the symbolic_cot
    return f"Analyze the following syllogism to determine its logical validity.\n\nSyllogism: {example['syllogism']}\n\nAnalysis:\n{example['symbolic_cot']}"

def main(args):
    # Model and Tokenizer setup
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Adjust for your model
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset_path, split="train")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",  # We need to tell SFTTrainer to use our formatted text
        formatting_func=format_prompt,
        max_seq_length=1024,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model for SemEval Task 11")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the base model (e.g., 'google/gemma-2b-it')")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
    # Example usage:
    # python train.py --model_name google/gemma-2b-it --dataset_path training_data_st1_scot.jsonl --output_dir ./gemma-finetuned-st1
    # python train.py --model_name deepseek-ai/deepseek-coder-6.7b-instruct --dataset_path training_data_st1_scot.jsonl --output_dir ./deepseek-finetuned-st1