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
    
    def generate_response(self, prompt: str, max_tokens: int = 64000, temperature: float = 0.0) -> str:
        """Generate response from LM Studio API - optimized for speed"""
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # Speed optimizations
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
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
    prompt_template = """You are a logic expert. Analyze this syllogism and give a clear, structured response.

SYLLOGISM: {syllogism}

Provide your analysis in this exact format:

1. Symbolic Abstraction: [assign A, B, C to entities]
2. Logical Formulation: [write premises with quantifiers] 
3. Step-by-Step Deduction: [check if conclusion follows]
4. Final Answer: VALID or INVALID

Begin your analysis now:"""
    
    return prompt_template.format(syllogism=syllogism)

def extract_validity_from_scot(scot_response: str) -> bool:
    """Extract validity from improved S-CoT response format"""
    if scot_response is None:
        return None
    
    # Debug: Print the response to see what we're working with
    print(f"\n--- DEBUG: S-CoT Response ---")
    print(f"Response: {scot_response}")
    print(f"--- End Response ---\n")
    
    response_upper = scot_response.upper()
    
    # Look for explicit VALID/INVALID in caps (from our structured prompt)
    if "INVALID" in response_upper:
        print("Found 'INVALID' in response")
        return False
    elif "VALID" in response_upper:
        print("Found 'VALID' in response")  
        return True
    
    # Fallback to original logic
    response_lower = scot_response.lower()
    
    # Look for "final answer:" section specifically
    if "final answer:" in response_lower:
        final_section = response_lower.split("final answer:")[-1]
        if "invalid" in final_section:
            print("Found 'invalid' in Final Answer section")
            return False
        elif "valid" in final_section:
            print("Found 'valid' in Final Answer section")
            return True
    
    # Look anywhere for invalid/valid
    if "invalid" in response_lower:
        print("Found 'invalid' anywhere in response")
        return False
    elif "valid" in response_lower:
        print("Found 'valid' anywhere in response")
        return True
    
    print("Could not extract validity from response")
    return None

def process_training_data_with_teacher(input_file: str, output_file: str, client: LMStudioClient):
    """Process training data through teacher model to generate S-CoT traces"""
    
    print(f"Loading training data from {input_file}...")
    with open(input_file, 'r') as f:
        training_data = json.load(f)
        training_data = training_data[:10]  # Test with 10 examples
    
    print(f"Processing {len(training_data)} examples with teacher model...")
    
    enriched_data = []
    failed_count = 0
    start_time = time.time()  # Fix: Define start_time here
    
    for i, item in enumerate(tqdm(training_data, desc="Generating S-CoT traces")):
        print(f"\n=== Processing Example {i+1}/{len(training_data)} ===")
        print(f"Syllogism: {item['syllogism']}")
        print(f"Ground Truth: {item['validity']}")
        
        # Create S-CoT prompt
        prompt = create_scot_prompt(item['syllogism'])
        
        # Generate S-CoT trace from teacher model
        scot_trace = client.generate_response(prompt, max_tokens=600, temperature=0.0)
        
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
            
            print(f"Teacher predicted: {predicted_validity}")
            print(f"Correct: {predicted_validity == item['validity'] if predicted_validity is not None else 'Unknown'}")
            
        else:
            failed_count += 1
            print(f"Failed to generate S-CoT for example {i+1}")
        
        # Small delay
        time.sleep(0.1)
    
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
    
    if total_predictions == 0:
        print("⚠️  Warning: No valid predictions extracted! Check the response format.")
    
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