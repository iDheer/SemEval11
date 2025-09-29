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
                {"role": "system", "content": "You are a logic expert. Answer directly without using <think> tags. Provide clear, structured responses in the requested format."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # More optimized parameters for speed
            "top_p": 0.95,  # Slightly more focused
            "top_k": 40,    # Limit choices for faster sampling
            "repeat_penalty": 1.05,  # Small penalty to avoid loops
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stop": ["</think>", "\n\n\n"]  # Stop on think tags or excessive newlines
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions", 
                json=payload, 
                headers=self.headers,
                timeout=300  # Back to 5 minute timeout for generation
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Debug: Print raw content first
            print(f"Raw API response content: '{content}'")
            
            if content is None or content.strip() == "":
                print("WARNING: API returned empty content!")
                return None
            
            # Post-process to remove <think> tags if they appear
            if "<think>" in content and "</think>" in content:
                import re
                original_content = content
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                content = content.strip()
                print(f"Removed <think> tags. Before: {len(original_content)} chars, After: {len(content)} chars")
            
            return content
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling LM Studio API: {e}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            print(f"Full response: {result}")
            return None

def create_scot_prompt(syllogism: str) -> str:
    prompt_template = """SYLLOGISM: {syllogism}

Provide your analysis in this exact format:

1. Symbolic Abstraction: [assign A, B, C to entities]
2. Logical Formulation: [write premises with quantifiers] 
3. Step-by-Step Deduction: [check if conclusion follows]
4. Final Answer: VALID or INVALID

After completing your analysis, provide a JSON summary:

{{"validity": "VALID"}}

Replace "VALID" with either "VALID" or "INVALID" based on your analysis. Begin your analysis now:"""
    
    return prompt_template.format(syllogism=syllogism)

def create_training_prompt_format(syllogism: str, scot_trace: str) -> str:
    """
    Format syllogism and S-CoT trace for supervised fine-tuning
    Args:
        syllogism (str): The syllogism to analyze
        scot_trace (str): The teacher model's S-CoT reasoning trace
    Returns:
        str: Formatted training prompt for student model
    """
    instruction = "Analyze the following syllogism and determine its validity based purely on its logical structure. Provide your reasoning in the exact format shown."
    
    formatted_prompt = f"""### Instruction:
{instruction}

### Input:
{syllogism}

### Response:
{scot_trace}"""
    
    return formatted_prompt

def extract_validity_from_json_response(response: str) -> bool:
    """Extract validity from response - look for JSON at the end after analysis"""
    if response is None or response.strip() == "":
        print("Empty response received!")
        return None
    
    print(f"\n--- DEBUG: Full Response ---")
    print(f"Response length: {len(response)} characters")
    print(f"First 200 chars: {response[:200]}...")
    print(f"Last 200 chars: ...{response[-200:]}")
    print(f"--- End Response ---\n")
    
    # First try to find JSON at the end (after the analysis)
    try:
        # Look for JSON near the end of the response
        json_start = response.rfind('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end != 0 and json_start < json_end:
            json_str = response[json_start:json_end].strip()
            print(f"Found JSON candidate: {json_str}")
            
            parsed = json.loads(json_str)
            print(f"Successfully parsed JSON: {parsed}")
            
            if 'validity' in parsed:
                validity = str(parsed['validity']).strip().upper()
                print(f"Extracted validity: '{validity}'")
                
                if validity == "VALID":
                    return True
                elif validity == "INVALID":
                    return False
                else:
                    print(f"Unexpected validity value: '{validity}'")
            else:
                print("No 'validity' field in JSON")
                
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
    except Exception as e:
        print(f"Error parsing JSON: {e}")
    
    # Fallback: Look for "Final Answer: VALID/INVALID" pattern
    import re
    final_answer_match = re.search(r'Final Answer:\s*(VALID|INVALID)', response, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        print(f"Found Final Answer pattern: {answer}")
        return True if answer == "VALID" else False
    
    # Last resort: Look for clear validity statements near the end
    response_upper = response.upper()
    last_500_chars = response_upper[-500:]  # Look in last 500 characters
    
    if "INVALID" in last_500_chars:
        print("Found 'INVALID' in end of response")
        return False
    elif "VALID" in last_500_chars:
        print("Found 'VALID' in end of response")
        return True
    
    print("Could not extract validity from response")
    return None

def create_simple_prompt(syllogism: str) -> str:
    """Ultra-clean simple prompt that forces JSON"""
    prompt_template = """{syllogism}

Is this syllogism logically valid?

Respond with ONLY this JSON (no other text):

{{
  "validity": "VALID"
}}

Replace "VALID" with exactly "VALID" or "INVALID":"""
    
    return prompt_template.format(syllogism=syllogism)

def extract_validity_simple(response: str) -> bool:
    """Extract validity from simple JSON format - strict parsing only"""
    if response is None or response.strip() == "":
        print("Empty response received!")
        return None
    
    print(f"\n--- DEBUG: Simple Response ---")
    print(f"Response: '{response}'")
    print(f"--- End Response ---\n")
    
    try:
        # Clean and parse JSON
        response_clean = response.strip()
        start_idx = response_clean.find('{')
        end_idx = response_clean.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            print("No JSON found in simple response!")
            return None
            
        json_str = response_clean[start_idx:end_idx]
        print(f"Extracted JSON: {json_str}")
        
        parsed = json.loads(json_str)
        print(f"Parsed JSON: {parsed}")
        
        if 'validity' not in parsed:
            print("No 'validity' field in JSON!")
            print(f"Available keys: {list(parsed.keys())}")
            return None
            
        validity = str(parsed['validity']).strip().upper()
        print(f"Validity value: '{validity}'")
        
        if validity == "VALID":
            return True
        elif validity == "INVALID":
            return False
        else:
            print(f"Invalid validity value: '{validity}' (expected 'VALID' or 'INVALID')")
            return None
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        print("Response is not valid JSON!")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def process_training_data_with_teacher(input_file: str, output_file: str, client: LMStudioClient, use_json=True):
    """Process training data through teacher model to generate S-CoT traces"""
    
    global timeout_count  # Declare as global variable
    timeout_count = 0     # Initialize the global variable
    
    print(f"Loading training data from {input_file}...")
    with open(input_file, 'r') as f:
        training_data = json.load(f)
        training_data = training_data[:150]  
    
    print(f"Processing {len(training_data)} examples with teacher model...")
    print(f"Using {'JSON' if use_json else 'simple'} prompt format")
    
    enriched_data = []
    failed_count = 0
    start_time = time.time()
    
    for i, item in enumerate(tqdm(training_data, desc="Generating S-CoT traces")):
        print(f"\n=== Processing Example {i+1}/{len(training_data)} ===")
        print(f"Syllogism: {item['syllogism']}")
        print(f"Ground Truth: {item['validity']}")
        
        # Create prompt based on format choice
        if use_json:
            prompt = create_scot_prompt(item['syllogism'])
            extract_func = extract_validity_from_json_response
            max_tokens = 16000  # Back to generous limit for complete reasoning
        else:
            prompt = create_simple_prompt(item['syllogism'])
            extract_func = extract_validity_simple
            max_tokens = 8000   # Sufficient for simple format
        
        # Generate S-CoT trace from teacher model
        scot_trace = client.generate_response(prompt, max_tokens=max_tokens, temperature=0.0)
        if scot_trace and "<think>" in scot_trace:
            import re
            scot_trace = re.sub(r'<think>.*?</think>', '', scot_trace, flags=re.DOTALL).strip()
        
        if scot_trace is not None:
            # Ensure think tags are removed from the trace we'll use for training
            cleaned_trace = scot_trace
            if "<think>" in scot_trace and "</think>" in scot_trace:
                import re
                cleaned_trace = re.sub(r'<think>.*?</think>', '', scot_trace, flags=re.DOTALL)
                cleaned_trace = cleaned_trace.strip()
                print(f"Cleaned think tags from trace. Original: {len(scot_trace)} chars, Cleaned: {len(cleaned_trace)} chars")
            
            # Extract predicted validity from the cleaned trace
            predicted_validity = extract_func(cleaned_trace)
            
            # Create training-ready prompt format using cleaned trace
            training_prompt = create_training_prompt_format(item['syllogism'], cleaned_trace)
            
            # Create enriched training example
            enriched_item = {
                'id': item['id'],
                'syllogism': item['syllogism'],
                'validity': item['validity'],
                'plausibility': item['plausibility'],
                'scot_trace': cleaned_trace,  # Store the cleaned version
                'training_prompt': training_prompt,  # Ready for fine-tuning
                'teacher_prediction': predicted_validity,
                'teacher_correct': predicted_validity == item['validity'] if predicted_validity is not None else None,
                'prompt_format': 'json' if use_json else 'simple'
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
    
    # Save training-ready prompts separately
    training_file = output_file.replace('.json', '_training_prompts.json')
    training_prompts = [
        {
            'id': item['id'],
            'training_prompt': item['training_prompt'],
            'validity': item['validity'],
            'plausibility': item['plausibility']
        }
        for item in enriched_data if item.get('training_prompt')
    ]
    
    print(f"Saving training prompts to {training_file}...")
    with open(training_file, 'w') as f:
        json.dump(training_prompts, f, indent=2)
    
    # Also save as plain text format for easy fine-tuning
    training_text_file = output_file.replace('.json', '_training.txt')
    print(f"Saving training text to {training_text_file}...")
    with open(training_text_file, 'w', encoding='utf-8') as f:
        for item in enriched_data:
            if item.get('training_prompt'):
                f.write(item['training_prompt'])
                f.write('\n\n' + '='*80 + '\n\n')  # Separator between examples
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("TEACHER MODEL S-COT GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total examples processed: {len(enriched_data)}")
    print(f"Failed generations: {failed_count}")
    print(f"Timeout failures: {timeout_count}")
    
    # Teacher model accuracy
    correct_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] == True)
    total_predictions = sum(1 for ex in enriched_data if ex['teacher_correct'] is not None)
    teacher_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"Teacher model accuracy: {teacher_accuracy:.4f} ({correct_predictions}/{total_predictions})")
    print(f"Success rate: {len(enriched_data)}/{len(training_data)} ({len(enriched_data)/len(training_data)*100:.1f}%)")
    
    if total_predictions == 0:
        print("⚠️  Warning: No valid predictions extracted! Trying alternative format...")
        return None
    
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"{'='*60}")
    print(f"Enriched dataset saved! Ready for student training.")
    
    return enriched_data

def main():
    print("Phase 3A: Teacher Model S-CoT Generation Pipeline (Enhanced)")
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
    
    # First, test basic server connectivity
    print("Testing basic server connection...")
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=10)
        if response.status_code == 200:
            print("✓ LM Studio server is responding")
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio server: {e}")
        print("Make sure LM Studio is running and server is started")
        return
    
    # Test with a very simple prompt first
    print("Testing with simple prompt...")
    simple_test = client.generate_response("Say hello", max_tokens=50, temperature=0.0)
    
    if simple_test is None or simple_test.strip() == "":
        print("❌ Simple test failed!")
        return
    
    print(f"✓ Simple test passed: {simple_test[:100]}...")
    
    # Now test with the structured analysis format
    print("Testing connection with structured analysis format...")
    test_prompt = """SYLLOGISM: All cats are mammals. All mammals are animals. Therefore, all cats are animals.

Provide your analysis in this exact format:

1. Symbolic Abstraction: [assign A, B, C to entities]
2. Logical Formulation: [write premises with quantifiers] 
3. Step-by-Step Deduction: [check if conclusion follows]
4. Final Answer: VALID or INVALID

After completing your analysis, provide a JSON summary:

{"validity": "VALID"}

Replace "VALID" with either "VALID" or "INVALID" based on your analysis. Begin your analysis now:"""
    
    test_response = client.generate_response(test_prompt, max_tokens=2000)
    
    if test_response is None or test_response.strip() == "":
        print("Error: Could not connect to LM Studio or received empty response!")
        print("Debugging checklist:")
        print("1. Is LM Studio running?")
        print("2. Is DeepSeek model loaded?")
        print("3. Is local server started (usually http://localhost:1234)?")
        print("4. Check LM Studio server logs for errors")
        print("5. Try reducing context length in LM Studio settings")
        print("6. Check if model supports system messages")
        return
    
    print("✓ Connected to LM Studio successfully!")
    print(f"✓ Test response received ({len(test_response)} chars): {test_response[:200]}...")
    
    # Check if we got a proper JSON response
    if '"validity"' in test_response or "VALID" in test_response.upper():
        print("✓ Model responding with expected format!")
    else:
        print("⚠️  Model response format unexpected - proceeding anyway...")
    
    # Try JSON format first
    print("\n🔄 Trying JSON-structured prompt format...")
    enriched_data = process_training_data_with_teacher(input_file, output_file, client, use_json=True)
    
    # If JSON format fails, try simple format
    if enriched_data is None or len([ex for ex in enriched_data if ex['teacher_correct'] is not None]) == 0:
        print("\n🔄 JSON format failed, trying simple format...")
        output_file_simple = "enriched_training_data_scot_simple.json"
        enriched_data = process_training_data_with_teacher(input_file, output_file_simple, client, use_json=False)
        output_file = output_file_simple
    
    if enriched_data and len([ex for ex in enriched_data if ex['teacher_correct'] is not None]) > 0:
        print(f"\n✅ Success! Next steps:")
        print(f"1. Review the generated S-CoT traces in {output_file}")
        print(f"2. Run student training with the enriched dataset")
        print(f"3. Compare student performance with/without S-CoT traces")
    else:
        print(f"\n❌ Both formats failed. Check your model and prompts.")

if __name__ == "__main__":
    main()