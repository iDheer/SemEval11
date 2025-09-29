import json
import time
from typing import Dict, List
import re
from tqdm import tqdm
import os
import google.generativeai as genai

class GeminiClient:
    """Client for Google Gemini API with retry logic"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('models/gemini-2.5-flash')
        
    def generate_response(self, prompt: str, temperature: float = 0.0, max_retries: int = 3) -> str:
        """Generate response with exponential backoff on failures"""
        for attempt in range(max_retries):
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
                    print(f"WARNING: API returned no content parts (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        time.sleep(5 * (attempt + 1))  # Wait longer each retry
                        continue
                    return None
                
                return response.text.strip()
                
            except Exception as e:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    return None
        
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
    
    final_answer_match = re.search(r'Final Answer:\s*(VALID|INVALID)', response, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        return True if answer == "VALID" else False
    
    response_upper = response.upper()
    last_200_chars = response_upper[-200:]
    
    if "INVALID" in last_200_chars:
        return False
    elif "VALID" in last_200_chars:
        return True
    
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

def retry_failed_examples(
    failed_file: str,
    existing_file: str,
    output_file: str,
    client: GeminiClient,
    delay_seconds: int = 3,
    checkpoint_interval: int = 25
):
    """Retry failed examples with more conservative rate limiting"""
    
    print(f"Loading failed examples from {failed_file}...")
    with open(failed_file, 'r') as f:
        failed_data = json.load(f)
    
    print(f"Loading existing successful examples from {existing_file}...")
    with open(existing_file, 'r') as f:
        enriched_data = json.load(f)
    
    print(f"\nRetrying {len(failed_data)} failed examples...")
    print(f"Using {delay_seconds}s delay between requests")
    
    failed_count = 0
    start_time = time.time()
    checkpoint_file = output_file.replace('.json', '_retry_checkpoint.json')
    
    for i, item in enumerate(tqdm(failed_data, desc="Retrying S-CoT generation")):
        print(f"\n=== Retrying Example {i+1}/{len(failed_data)} (ID: {item.get('id')}) ===")
        print(f"Syllogism: {item['syllogism'][:80]}...")
        
        prompt = create_few_shot_prompt(item['syllogism'])
        scot_trace = client.generate_response(prompt, temperature=0.0, max_retries=3)
        
        if scot_trace:
            predicted_validity = extract_validity_from_response(scot_trace)
            training_prompt = create_training_prompt_format(item['syllogism'], scot_trace)
            
            is_correct = None
            if predicted_validity is not None and 'validity' in item:
                is_correct = (predicted_validity == item['validity'])
            
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
            print(f"SUCCESS - Teacher predicted: {predicted_validity}, Correct: {is_correct}")
            
        else:
            failed_count += 1
            print(f"FAILED after retries")
        
        # Checkpoint every N examples
        if (i + 1) % checkpoint_interval == 0:
            print(f"\nSaving checkpoint at {len(enriched_data)} total examples...")
            with open(checkpoint_file, 'w') as f:
                json.dump(enriched_data, f, indent=2)
        
        # Conservative delay
        time.sleep(delay_seconds)
    
    # Final save
    print(f"\nSaving final dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    # Stats
    total_now = len(enriched_data)
    new_successes = total_now - 247
    still_failed = len(failed_data) - new_successes
    
    print(f"\n{'='*60}")
    print("RETRY SUMMARY")
    print(f"{'='*60}")
    print(f"Previously completed: 247")
    print(f"Attempted retries: {len(failed_data)}")
    print(f"New successes: {new_successes}")
    print(f"Still failed: {still_failed}")
    print(f"Total now: {total_now} / 671")
    print(f"Time taken: {(time.time() - start_time) / 60:.1f} minutes")
    print(f"{'='*60}")
    
    # Update summary
    summary_file = output_file.replace('.json', '_summary.json')
    correct = sum(1 for ex in enriched_data if ex['teacher_correct'] == True)
    total_pred = sum(1 for ex in enriched_data if ex['teacher_correct'] is not None)
    
    summary = {
        'total_examples': len(enriched_data),
        'failed_generations': 671 - len(enriched_data),
        'teacher_accuracy': correct / total_pred if total_pred > 0 else 0,
        'correct_predictions': correct,
        'total_predictions': total_pred
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

def main():
    print("Retry Failed Examples with Better Rate Limiting")
    print("=" * 60)
    
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY not set")
        return
    
    client = GeminiClient(GEMINI_API_KEY)
    
    retry_failed_examples(
        failed_file='failed_examples.json',
        existing_file='enriched_training_data_gemini.json',
        output_file='enriched_training_data_gemini.json',
        client=client,
        delay_seconds=3,
        checkpoint_interval=25
    )
    
    print("\nRetry complete!")

if __name__ == "__main__":
    main()