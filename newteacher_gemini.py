import json
import time
from typing import Dict, List
import re
from tqdm import tqdm
import os
import google.generativeai as genai

class GeminiClient:
    """Client for Google Gemini API"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        
    def generate_response(self, prompt: str, temperature: float = 0.0) -> str:
        """Generate response from Gemini API"""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.parts:
                print("WARNING: API returned no content parts!")
                return None
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None

def create_few_shot_prompt(syllogism: str) -> str:
    """Create few-shot prompt with examples"""
    prompt_template = """Analyze syllogisms using symbolic logic. Here are examples:

EXAMPLE 1:
SYLLOGISM: All cats are mammals. All mammals are animals. Therefore, all cats are animals.

ANALYSIS:
1. Symbolic Abstraction: Let A = cats, B = mammals, C = animals
2. Logical Formulation: Premise 1: ∀x (A(x) → B(x)). Premise 2: ∀x (B(x) → C(x)). Conclusion: ∀x (A(x) → C(x))
3. Step-by-Step Deduction: If all A are B, and all B are C, then by transitivity all A must be C. The conclusion follows logically.
4. Final Answer: VALID

EXAMPLE 2:
SYLLOGISM: Some birds are penguins. All penguins are flightless. Therefore, all birds are flightless.

ANALYSIS:
1. Symbolic Abstraction: Let A = birds, B = penguins, C = flightless creatures
2. Logical Formulation: Premise 1: ∃x (A(x) ∧ B(x)). Premise 2: ∀x (B(x) → C(x)). Conclusion: ∀x (A(x) → C(x))
3. Step-by-Step Deduction: Premise 1 states only some birds are penguins. Premise 2 states all penguins are flightless. This does not imply all birds are flightless, only that some birds (the penguins) are flightless.
4. Final Answer: INVALID

Now analyze this new syllogism:

SYLLOGISM: {syllogism}

Provide your analysis in the same format."""
    
    return prompt_template.format(syllogism=syllogism)

def extract_validity_from_response(response: str) -> bool:
    """Extract validity from teacher's response"""
    if response is None or response.strip() == "":
        return None
    
    # Look for "Final Answer: VALID/INVALID"
    final_answer_match = re.search(r'Final Answer:\s*(VALID|INVALID)', response, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        return True if answer == "VALID" else False
    
    # Fallback: look in last part of response
    response_upper = response.upper()
    last_200_chars = response_upper[-200:]
    
    if "INVALID" in last_200_chars:
        return False
    elif "VALID" in last_200_chars:
        return True
    
    print("Could not extract validity from response")
    return None

def create_training_prompt_format(syllogism: str, scot_trace: str) -> str:
    """Format for student fine-tuning"""
    instruction = "Analyze the following syllogism and determine its validity based purely on its structure."
    
    formatted_prompt = f"""### Instruction:
{instruction}

### Input:
{syllogism}

### Response:
{scot_trace}"""
    
    return formatted_prompt

def load_checkpoint(checkpoint_file: str):
    """Load existing checkpoint if available"""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded checkpoint with {len(data)} existing examples")
            return data
        except:
            print("Checkpoint file exists but couldn't be loaded, starting fresh")
            return []
    return []

def process_training_data_with_teacher(
    input_file: str, 
    output_file: str, 
    client: GeminiClient,
    num_examples: int = None,
    checkpoint_interval: int = 50
):
    """Process training data through teacher model with checkpointing"""
    
    print(f"Loading training data from {input_file}...")
    try:
        with open(input_file, 'r') as f:
            training_data = json.load(f)
            if num_examples is not None:
                training_data = training_data[:num_examples]
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}")
        return []

    # Check for existing checkpoint
    checkpoint_file = output_file.replace('.json', '_checkpoint.json')
    enriched_data = load_checkpoint(checkpoint_file)
    
    # Skip already processed examples
    start_idx = len(enriched_data)
    
    if start_idx > 0:
        print(f"Resuming from example {start_idx + 1}...")
        training_data = training_data[start_idx:]
    
    print(f"Processing {len(training_data)} examples with teacher model (few-shot)...")
    
    failed_count = 0
    start_time = time.time()
    
    for i, item in enumerate(tqdm(training_data, desc="Generating S-CoT traces", initial=start_idx, total=start_idx + len(training_data))):
        actual_idx = start_idx + i
        print(f"\n=== Processing Example {actual_idx + 1}/{start_idx + len(training_data)} ===")
        print(f"Syllogism: {item['syllogism']}")
        print(f"Ground Truth: {item.get('validity', 'N/A')}")
        
        # Create few-shot prompt
        prompt = create_few_shot_prompt(item['syllogism'])
        
        # Generate S-CoT trace
        scot_trace = client.generate_response(prompt, temperature=0.0)
        
        if scot_trace:
            # Extract predicted validity
            predicted_validity = extract_validity_from_response(scot_trace)
            
            # Create training-ready prompt
            training_prompt = create_training_prompt_format(item['syllogism'], scot_trace)
            
            is_correct = None
            if predicted_validity is not None and 'validity' in item:
                is_correct = (predicted_validity == item['validity'])

            # Create enriched training example
            enriched_item = {
                'id': item.get('id'),
                'syllogism': item['syllogism'],
                'validity': item.get('validity'),
                'plausibility': item.get('plausibility'),
                's-cot_trace': scot_trace,
                'training_prompt': training_prompt,
                'teacher_prediction': predicted_validity,
                'teacher_correct': is_correct
            }
            
            enriched_data.append(enriched_item)
            
            print(f"Teacher predicted: {predicted_validity}")
            print(f"Correct: {is_correct if is_correct is not None else 'Unknown'}")
            
        else:
            failed_count += 1
            print(f"Failed to generate S-CoT for example {actual_idx + 1}")
        
        # Save checkpoint every N examples
        if (actual_idx + 1) % checkpoint_interval == 0:
            print(f"\nSaving checkpoint at {actual_idx + 1} examples...")
            with open(checkpoint_file, 'w') as f:
                json.dump(enriched_data, f, indent=2)
            
            # Calculate and show current stats
            correct_so_far = sum(1 for ex in enriched_data if ex['teacher_correct'] == True)
            total_so_far = sum(1 for ex in enriched_data if ex['teacher_correct'] is not None)
            acc_so_far = correct_so_far / total_so_far if total_so_far > 0 else 0
            print(f"Checkpoint saved. Current accuracy: {acc_so_far:.4f} ({correct_so_far}/{total_so_far})")
        
        # Rate limiting delay
        time.sleep(1)
    
    if not enriched_data:
        print("\nNo data was processed to save.")
        return []

    # Final save
    print(f"\nSaving final enriched dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    # Delete checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"Removed checkpoint file (processing complete)")
    
    training_file = output_file.replace('.json', '_training_prompts.json')
    training_prompts = [
        {
            'id': item['id'],
            'syllogism': item['syllogism'],
            'training_prompt': item['training_prompt'],
            'validity': item['validity'],
            'plausibility': item['plausibility']
        }
        for item in enriched_data
    ]
    
    print(f"Saving training prompts to {training_file}...")
    with open(training_file, 'w') as f:
        json.dump(training_prompts, f, indent=2)
    
    summary_file = output_file.replace('.json', '_summary.json')
    correct_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] == True)
    total_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] is not None)
    teacher_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    summary = {
        'total_examples': len(enriched_data),
        'failed_generations': failed_count,
        'teacher_accuracy': teacher_accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        'processing_time_seconds': time.time() - start_time
    }
    
    print(f"Saving summary to {summary_file}...")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TEACHER MODEL S-COT GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples processed: {len(enriched_data)}")
    print(f"Failed generations: {failed_count}")
    print(f"Teacher model accuracy: {teacher_accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Estimated total time: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"{'='*60}")
    print(f"\nFiles saved:")
    print(f"  - {output_file} (full enriched data)")
    print(f"  - {training_file} (training prompts only)")
    print(f"  - {summary_file} (summary statistics)")
    
    return enriched_data

def main():
    print("Phase 3A: Teacher Model S-CoT Generation with Gemini")
    print("=" * 60)
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("FATAL ERROR: The 'GEMINI_API_KEY' environment variable is not set.")
        print("Please set it before running the script.")
        return

    input_file = "data_splits/train_split.json"
    output_file = "enriched_training_data_gemini.json"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return
    
    print("Initializing Gemini API...")
    client = GeminiClient(GEMINI_API_KEY)
    
    print("Testing Gemini API connection...")
    test_response = client.generate_response("Say 'API working'")
    if test_response and 'API working' in test_response:
        print(f"Connected successfully: {test_response}")
    else:
        print("Connection failed. Please check your API key and network connection.")
        return
    
    # Process all 671 training examples with checkpoints every 50
    process_training_data_with_teacher(
        input_file, 
        output_file, 
        client,
        num_examples=None,  # Process all examples
        checkpoint_interval=50
    )
    
    print("\nDone! Ready for student model training.")

if __name__ == "__main__":
    main()