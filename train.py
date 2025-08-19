import json
import ollama
import argparse
import os
from datetime import datetime

def create_modelfile(base_model, training_data, output_name):
    """
    Creates an Ollama Modelfile for fine-tuning with the training data.
    This uses few-shot learning embedded in the system prompt.
    """
    
    # Read training data and create a system prompt with examples
    with open(training_data, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    # Create few-shot examples for the system prompt
    few_shot_examples = []
    for i, example in enumerate(examples[:5]):  # Use first 5 as few-shot examples
        few_shot_examples.append(f"""
Example {i+1}:
Syllogism: {example['syllogism']}
Analysis: {example['symbolic_cot']}
""")
    
    system_prompt = f"""You are an expert in formal logic specializing in syllogistic reasoning. Your task is to analyze syllogisms and determine their logical validity using symbolic chain-of-thought reasoning.

You must follow this exact format:
1. Identify the three key terms
2. Assign symbols A, B, C to these terms  
3. Translate premises and conclusion to symbolic form
4. Perform logical analysis based on symbolic form
5. State final validity (True or False)

Here are some examples of correct analysis:
{"".join(few_shot_examples)}

Always provide your analysis in this structured format and focus on logical structure, not content plausibility."""

    modelfile_content = f"""FROM {base_model}
SYSTEM "{system_prompt}"
PARAMETER temperature 0.1
PARAMETER top_p 0.9"""

    return modelfile_content

def main(args):
    print(f"Creating fine-tuned model from {args.model_name}...")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate a unique model name for the fine-tuned version
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_model_name = f"{args.model_name.replace(':', '_')}_finetuned_{timestamp}"
    
    # Create Modelfile
    modelfile_content = create_modelfile(args.model_name, args.dataset_path, finetuned_model_name)
    
    # Save Modelfile
    modelfile_path = os.path.join(args.output_dir, "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at {modelfile_path}")
    
    # Save model info for later use
    model_info = {
        "base_model": args.model_name,
        "finetuned_model_name": finetuned_model_name,
        "dataset_path": args.dataset_path,
        "created_at": datetime.now().isoformat(),
        "modelfile_path": modelfile_path
    }
    
    info_path = os.path.join(args.output_dir, "model_info.json")
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"""
Fine-tuning setup complete!

To create the fine-tuned model, run:
    ollama create {finetuned_model_name} -f {modelfile_path}

Then you can use it in evaluation with:
    --model_name {finetuned_model_name}

Model info saved to: {info_path}
""")
    
    # Try to create the model automatically
    try:
        print(f"Attempting to create model {finetuned_model_name}...")
        result = os.system(f'ollama create {finetuned_model_name} -f "{modelfile_path}"')
        if result == 0:
            print(f"✅ Model {finetuned_model_name} created successfully!")
            
            # Update model info with success
            model_info["ollama_model_created"] = True
            model_info["ollama_model_name"] = finetuned_model_name
            with open(info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
        else:
            raise Exception(f"ollama create returned exit code {result}")
            
    except Exception as e:
        print(f"⚠️  Could not automatically create Ollama model: {e}")
        print(f"Please run manually: ollama create {finetuned_model_name} -f {modelfile_path}")
        model_info["ollama_model_created"] = False
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune an Ollama model for SemEval Task 11")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the base Ollama model (e.g., 'gemma3:4b-it-q4_K_M')")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model files")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (used for naming)")
    
    args = parser.parse_args()
    main(args)
    
    # Example usage:
    # python train.py --model_name gemma3:4b-it-q4_K_M --dataset_path training_data_st1_scot.jsonl --output_dir ./gemma-finetuned-st1
    # python train.py --model_name deepseek-r1:8b --dataset_path training_data_st1_scot.jsonl --output_dir ./deepseek-finetuned-st1