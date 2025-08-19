# generate_submission.py
import json
import ollama
import re
import argparse
from ollama_utils import ensure_only_model_loaded

def parse_validity_from_output(text: str) -> bool | None:
    match = re.search(r"Validity:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    # Fallback: if the model doesn't follow the format, make a guess.
    if "is valid" in text.lower() and "is not valid" not in text.lower():
        return True
    if "is invalid" in text.lower() or "is not valid" in text.lower():
        return False
    return False  # Default to false if unsure

def main(args):
    # Restart Ollama and prepare the model so only the chosen model is available for inference
    ensure_only_model_loaded(args.model_name, pull_if_missing=True, restart_if_needed=True)

    print(f"Generating submission using model: {args.model_name}")

    # Load the official test data
    test_data = []
    with open(args.test_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    predictions = []
    
    for i, item in enumerate(test_data):
        print(f"Processing item {i+1}/{len(test_data)}...")
        prompt = f"Analyze the following syllogism to determine its logical validity.\n\nSyllogism: {item['syllogism']}\n\nAnalysis:\n"
        
        try:
            # Generate output using Ollama
            response = ollama.chat(
                model=args.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            analysis_output = response['message']['content']
            
            predicted_validity = parse_validity_from_output(analysis_output)
            
            # For Subtask 2, you would add premise selection logic here as well
            
            predictions.append({
                "id": item['id'],
                "prediction": predicted_validity
            })
            
        except Exception as e:
            print(f"Error processing item {item['id']}: {e}")
            # Use fallback prediction
            predictions.append({
                "id": item['id'],
                "prediction": False  # Conservative fallback
            })

    # Write to submission file
    with open(args.output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
            
    print(f"\nSubmission file created successfully at {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission file for SemEval Task 11")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the Ollama model")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the unlabeled test JSONL file")
    parser.add_argument("--output_file", type=str, default="submission.jsonl", help="Name of the output submission file")
    
    args = parser.parse_args()
    main(args)
    
    # Example usage:
    # python generate_submission.py --model_name gemma3_4b-it-q4_K_M_finetuned_20250819_143021 --test_data_path official_test_set.jsonl