import json
import requests
import time
from typing import Dict, List
import re
from tqdm import tqdm
import os

class OpenRouterClient:
    """Client for OpenRouter API using DeepSeek model"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:3000",  # Optional
            "X-Title": "Logic Analysis Pipeline"  # Optional
        }
        # Use the same model as online
        self.model = "deepseek/deepseek-r1:nitro"  # or "deepseek/deepseek-r1"
    
    def generate_response(self, prompt: str, max_tokens: int = 16000, temperature: float = 0.0) -> str:
        """Generate response from OpenRouter API"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions", 
                json=payload, 
                headers=self.headers,
                timeout=120
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Debug API response structure
            print(f"API Response keys: {list(result.keys())}")
            if 'choices' in result and len(result['choices']) > 0:
                content = result["choices"][0]["message"]["content"]
                return content
            else:
                print(f"Unexpected API response structure: {result}")
                return None
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            print(f"Full response: {result}")
            return None

def create_analysis_prompt(syllogism: str) -> str:
    """Create the exact prompt that worked online"""
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

def extract_validity_from_response(response: str) -> bool:
    """Extract validity from response - multiple strategies"""
    if response is None or response.strip() == "":
        print("Empty response received!")
        return None
    
    print(f"\n--- Response Analysis ---")
    print(f"Response length: {len(response)} characters")
    print(f"Response preview: {response[:300]}...")
    if len(response) > 600:
        print(f"Response ending: ...{response[-300:]}")
    print(f"--- End Analysis ---\n")
    
    # Strategy 1: Look for JSON at the end
    try:
        json_start = response.rfind('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end != 0 and json_start < json_end:
            json_str = response[json_start:json_end].strip()
            print(f"Found JSON: {json_str}")
            
            parsed = json.loads(json_str)
            if 'validity' in parsed:
                validity = str(parsed['validity']).strip().upper()
                print(f"JSON validity: '{validity}'")
                
                if validity == "VALID":
                    return True
                elif validity == "INVALID":
                    return False
                    
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
    except Exception as e:
        print(f"JSON extraction error: {e}")
    
    # Strategy 2: Look for "Final Answer:" pattern
    final_answer_match = re.search(r'Final Answer:\s*(VALID|INVALID)', response, re.IGNORECASE)
    if final_answer_match:
        answer = final_answer_match.group(1).upper()
        print(f"Found Final Answer: {answer}")
        return True if answer == "VALID" else False
    
    # Strategy 3: Look for "4. Final Answer" pattern (numbered)
    numbered_answer_match = re.search(r'4\.\s*Final Answer:\s*(VALID|INVALID)', response, re.IGNORECASE)
    if numbered_answer_match:
        answer = numbered_answer_match.group(1).upper()
        print(f"Found numbered Final Answer: {answer}")
        return True if answer == "VALID" else False
    
    # Strategy 4: Last resort - look in final section
    response_upper = response.upper()
    last_500_chars = response_upper[-500:]
    
    if "INVALID" in last_500_chars:
        print("Found 'INVALID' in response ending")
        return False
    elif "VALID" in last_500_chars:
        print("Found 'VALID' in response ending")
        return True
    
    print("Could not extract validity from response")
    return None

def test_openrouter_syllogisms(api_key: str, num_examples: int = 10):
    """Test OpenRouter API with syllogisms"""
    
    print("OpenRouter DeepSeek Test Pipeline")
    print("=" * 50)
    
    # Load test data
    try:
        with open('data_splits/train_split.json', 'r') as f:
            training_data = json.load(f)
            test_data = training_data[:num_examples]
    except FileNotFoundError:
        print("Creating sample test data...")
        test_data = [
            {
                "id": 1,
                "syllogism": "Some of the bees that exist are insects. Every single bee is an arthropod. Thus, some arthropods are insects.",
                "validity": True,
                "plausibility": True
            },
            {
                "id": 2,
                "syllogism": "Plants are never stones. A few stones are, in fact, flowers. There exist flowers that are not plants.",
                "validity": True,
                "plausibility": False
            },
            {
                "id": 3,
                "syllogism": "It is the case that no insect is a mammal. Every single insect is an animal. For this reason, every animal is a mammal.",
                "validity": False,
                "plausibility": True
            },
            {
                "id": 4,
                "syllogism": "Not a single man is a woman. It is also true that no adult is a man. It follows that no adult is a woman.",
                "validity": False,
                "plausibility": True
            },
            {
                "id": 5,
                "syllogism": "It is not true that any tree is green. There is not a single car that is a tree. Consequently, there is no car that is green.",
                "validity": False,
                "plausibility": False
            }
        ]
        test_data = test_data[:num_examples]
    
    # Initialize client
    print(f"Initializing OpenRouter client...")
    client = OpenRouterClient(api_key)
    
    # Test connection
    print("Testing connection...")
    test_prompt = create_analysis_prompt("All cats are mammals. All mammals are animals. Therefore, all cats are animals.")
    test_response = client.generate_response(test_prompt, max_tokens=2000)
    
    if test_response is None or test_response.strip() == "":
        print("❌ Connection test failed!")
        return
    
    print("✅ Connection successful!")
    print(f"Test response length: {len(test_response)} chars")
    
    # Process examples
    print(f"\nProcessing {len(test_data)} examples...")
    results = []
    
    for i, item in enumerate(tqdm(test_data, desc="Testing syllogisms")):
        print(f"\n=== Example {i+1}/{len(test_data)} ===")
        print(f"Syllogism: {item['syllogism']}")
        print(f"Ground Truth: {item['validity']}")
        
        # Generate response
        prompt = create_analysis_prompt(item['syllogism'])
        response = client.generate_response(prompt, max_tokens=4000, temperature=0.0)
        
        if response is not None:
            # Extract validity
            predicted_validity = extract_validity_from_response(response)
            
            # Store result
            result = {
                'id': item['id'],
                'syllogism': item['syllogism'],
                'ground_truth': item['validity'],
                'plausibility': item.get('plausibility', None),
                'predicted_validity': predicted_validity,
                'correct': predicted_validity == item['validity'] if predicted_validity is not None else None,
                'full_response': response,
                'response_length': len(response)
            }
            
            results.append(result)
            
            print(f"Predicted: {predicted_validity}")
            print(f"Correct: {predicted_validity == item['validity'] if predicted_validity is not None else 'Unknown'}")
            
        else:
            print(f"❌ Failed to get response for example {i+1}")
        
        # Be respectful to API
        time.sleep(1)
    
    # Save results
    output_file = f"openrouter_test_results_{num_examples}.json"
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("OPENROUTER TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Total examples: {len(results)}")
    
    successful_predictions = [r for r in results if r['correct'] is not None]
    correct_predictions = [r for r in results if r['correct'] == True]
    
    if len(successful_predictions) > 0:
        accuracy = len(correct_predictions) / len(successful_predictions)
        print(f"Successful extractions: {len(successful_predictions)}/{len(results)}")
        print(f"Accuracy: {accuracy:.4f} ({len(correct_predictions)}/{len(successful_predictions)})")
        
        # Response length stats
        response_lengths = [r['response_length'] for r in results if r['full_response']]
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            print(f"Average response length: {avg_length:.0f} characters")
            print(f"Response length range: {min(response_lengths)} - {max(response_lengths)} chars")
    else:
        print("❌ No successful predictions extracted!")
    
    print(f"\n✅ Results saved to {output_file}")
    return results

def main():
    # Your OpenRouter API key
    API_KEY = "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"
    
    print("Starting OpenRouter DeepSeek test...")
    
    # Test with 10 examples
    results = test_openrouter_syllogisms(API_KEY, num_examples=10)
    
    if results:
        print("\n🎉 Test completed successfully!")
        print("Compare these results with your LM Studio pipeline.")
    else:
        print("\n❌ Test failed. Check your API key and connection.")

if __name__ == "__main__":
    main()