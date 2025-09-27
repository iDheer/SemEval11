import json
import requests
import time
from typing import Dict, List
import re
from tqdm import tqdm
import os

class LMStudioClient:
    """Client for LM Studio local API"""
    
    def __init__(self, base_url="http://localhost:1234"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """Generate response from LM Studio API"""
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions", 
                json=payload, 
                headers=self.headers,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling LM Studio API: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None

def create_scot_prompt(syllogism: str) -> str:
    """Create the S-CoT prompt with the provided template"""
    prompt_template = """Your task is to analyze a logical syllogism and determine its validity based purely on its structure. Follow these steps precisely:

1. **Symbolic Abstraction**: Identify the distinct categories or entities in the syllogism and assign them symbolic variables like (A, B, C etc) as required.

2. **Logical Formulation**: Translate each premise and the conclusion into its formal logical structure using quantifiers like (All, Some, No etc) and the assigned variables.

3. **Step-by-Step Deduction**: Analyze the relationship between the premises in the logical formulation. Explain, step-by-step, whether the conclusion necessarily follows from the premises. Reference the variables to show your reasoning. State clearly if the premises are insufficient to guarantee the conclusion.

4. **Final Verdict**: Conclude with a single word: "Valid" if the argument is logically sound, or "Invalid" if it is not.

---

**Syllogism to Analyze:**
"{syllogism}"

**Analysis:**"""
    
    return prompt_template.format(syllogism=syllogism)

def extract_validity_from_scot(scot_response: str) -> bool:
    """Extract the final validity verdict from S-CoT response"""
    # Look for "Valid" or "Invalid" at the end of the response
    if scot_response is None:
        return None
    
    # Check for final verdict patterns
    lines = scot_response.strip().split('\n')
    for line in reversed(lines):
        line = line.strip().lower()
        if 'valid' in line and 'invalid' not in line:
            return True
        elif 'invalid' in line:
            return False
    
    # Fallback: look anywhere in the response for verdict
    response_lower = scot_response.lower()
    if 'final verdict:' in response_lower:
        verdict_section = response_lower.split('final verdict:')[-1]
        if 'valid' in verdict_section and 'invalid' not in verdict_section:
            return True
        elif 'invalid' in verdict_section:
            return False
    
    return None

def process_training_data_with_teacher(input_file: str, output_file: str, client: LMStudioClient):
    """Process training data through teacher model to generate S-CoT traces"""
    
    print(f"Loading training data from {input_file}...")
    with open(input_file, 'r') as f:
        training_data = json.load(f)
    
    print(f"Processing {len(training_data)} examples with teacher model...")
    
    enriched_data = []
    failed_count = 0
    
    for i, item in enumerate(tqdm(training_data, desc="Generating S-CoT traces")):
        # Create S-CoT prompt
        prompt = create_scot_prompt(item['syllogism'])
        
        # Generate S-CoT trace from teacher model
        scot_trace = client.generate_response(prompt, max_tokens=1500, temperature=0.1)
        
        if scot_trace is not None:
            # Extract predicted validity from S-CoT trace
            predicted_validity = extract_validity_from_scot(scot_trace)
            
            # Create enriched training example
            enriched_item = {
                'id': item['id'],
                'syllogism': item['syllogism'],
                'validity': item['validity'],
                'plausibility': item['plausibility'],
                'scot_trace': scot_trace,
                'teacher_prediction': predicted_validity,
                'teacher_correct': predicted_validity == item['validity'] if predicted_validity is not None else None
            }
            
            enriched_data.append(enriched_item)
            
            # Progress update every 50 examples
            if (i + 1) % 50 == 0:
                correct_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] == True)
                total_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] is not None)
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                print(f"Progress: {i+1}/{len(training_data)} | Teacher accuracy so far: {accuracy:.3f}")
        else:
            failed_count += 1
            print(f"Failed to generate S-CoT for example {i+1}")
        
        # Small delay to avoid overwhelming the API
        time.sleep(0.5)
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(enriched_data, f, indent=2)
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("TEACHER MODEL S-COT GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples processed: {len(enriched_data)}")
    print(f"Failed generations: {failed_count}")
    
    # Teacher model accuracy
    correct_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] == True)
    total_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] is not None)
    teacher_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"Teacher model accuracy: {teacher_accuracy:.4f} ({correct_predictions}/{total_predictions})")
    
    # Analyze teacher performance by category
    categories = {'VP': [], 'IP': [], 'VI': [], 'II': []}
    
    for ex in enriched_data:
        if ex['teacher_correct'] is not None:
            val = ex['validity']
            plaus = ex['plausibility']
            
            if val and plaus:
                categories['VP'].append(ex['teacher_correct'])
            elif not val and plaus:
                categories['IP'].append(ex['teacher_correct'])
            elif val and not plaus:
                categories['VI'].append(ex['teacher_correct'])
            else:
                categories['II'].append(ex['teacher_correct'])
    
    print(f"\nTeacher accuracy by category:")
    for cat_name, results in categories.items():
        if results:
            acc = sum(results) / len(results)
            count = len(results)
            valid_str = "Valid" if cat_name[0] == 'V' else "Invalid"
            plaus_str = "Plausible" if cat_name[1] == 'P' else "Implausible"
            print(f"{cat_name} ({valid_str}, {plaus_str}): {acc:.3f} ({count} examples)")
    
    print(f"{'='*60}")
    print(f"Enriched dataset saved! Ready for student training.")
    
    return enriched_data

def main():
    print("Phase 3A: Teacher Model S-CoT Generation Pipeline")
    print("=" * 60)
    
    # Configuration
    input_file = "data_splits/train_split.json"
    output_file = "enriched_training_data_scot.json"
    
    # Check if training split exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Please run the balanced data split script first.")
        return
    
    # Initialize LM Studio client
    print("Connecting to LM Studio...")
    client = LMStudioClient()
    
    # Test connection
    test_prompt = "Hello, can you hear me?"
    test_response = client.generate_response(test_prompt, max_tokens=50)
    
    if test_response is None:
        print("Error: Could not connect to LM Studio!")
        print("Make sure:")
        print("1. LM Studio is running")
        print("2. DeepSeek model is loaded")
        print("3. Local server is started (usually http://localhost:1234)")
        return
    
    print("✓ Connected to LM Studio successfully!")
    print(f"✓ Test response received: {test_response[:100]}...")
    
    # Process training data
    enriched_data = process_training_data_with_teacher(input_file, output_file, client)
    
    print(f"\nNext steps:")
    print(f"1. Review the generated S-CoT traces in {output_file}")
    print(f"2. Run student training with the enriched dataset")
    print(f"3. Compare student performance with/without S-CoT traces")

if __name__ == "__main__":
    main()