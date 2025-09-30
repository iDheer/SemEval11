import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

print("="*60)
print("VALIDATION SET COMPARISON: LAYER 20 vs LAYER 23")
print("="*60)

# Load validation data
with open('data_splits/val_split.json', 'r') as f:
    val_data = json.load(f)

print(f"Validation examples: {len(val_data)}")

# Model setup
model_name = "google/gemma-2-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def evaluate_model(model_path, model_name_label):
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name_label}")
    print(f"{'='*60}")
    
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    predictions = []
    
    for item in tqdm(val_data, desc=f"Generating predictions"):
        prompt = f"""### Instruction:
Analyze the following syllogism and determine its validity based purely on its structure.

### Input:
{item['syllogism']}

### Response:"""
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract validity
        if "VALID" in response.upper() and "INVALID" not in response.upper():
            pred = True
        elif "INVALID" in response.upper():
            pred = False
        else:
            pred = None
        
        predictions.append({
            'true_validity': item['validity'],
            'predicted_validity': pred,
            'plausibility': item['plausibility']
        })
    
    # Calculate metrics
    valid_preds = [p for p in predictions if p['predicted_validity'] is not None]
    correct = sum(1 for p in valid_preds if p['predicted_validity'] == p['true_validity'])
    accuracy = correct / len(valid_preds) if valid_preds else 0
    
    # Content effect
    plausible = [p for p in valid_preds if p['plausibility'] == True]
    implausible = [p for p in valid_preds if p['plausibility'] == False]
    
    plaus_correct = sum(1 for p in plausible if p['predicted_validity'] == p['true_validity'])
    implaus_correct = sum(1 for p in implausible if p['predicted_validity'] == p['true_validity'])
    
    plaus_acc = plaus_correct / len(plausible) if plausible else 0
    implaus_acc = implaus_correct / len(implausible) if implausible else 0
    content_effect = plaus_acc - implaus_acc
    
    # Clean up
    del model
    del base_model
    torch.cuda.empty_cache()
    
    return {
        'accuracy': accuracy,
        'plausible_accuracy': plaus_acc,
        'implausible_accuracy': implaus_acc,
        'content_effect': content_effect,
        'correct': correct,
        'total': len(valid_preds)
    }

# Evaluate both models
layer20_results = evaluate_model("./student_model_steered_final", "Layer 20 Model")
layer23_results = evaluate_model("./student_model_steered_layer23_final", "Layer 23 Model")

# Print comparison
print("\n" + "="*60)
print("VALIDATION SET COMPARISON")
print("="*60)

print(f"\nLayer 20 Model:")
print(f"  Accuracy: {layer20_results['accuracy']:.4f} ({layer20_results['correct']}/{layer20_results['total']})")
print(f"  Plausible Acc: {layer20_results['plausible_accuracy']:.4f}")
print(f"  Implausible Acc: {layer20_results['implausible_accuracy']:.4f}")
print(f"  Content Effect: {layer20_results['content_effect']:.4f}")

print(f"\nLayer 23 Model:")
print(f"  Accuracy: {layer23_results['accuracy']:.4f} ({layer23_results['correct']}/{layer23_results['total']})")
print(f"  Plausible Acc: {layer23_results['plausible_accuracy']:.4f}")
print(f"  Implausible Acc: {layer23_results['implausible_accuracy']:.4f}")
print(f"  Content Effect: {layer23_results['content_effect']:.4f}")

print(f"\nDifference (Layer 23 - Layer 20):")
print(f"  Accuracy: {layer23_results['accuracy'] - layer20_results['accuracy']:+.4f}")
print(f"  Content Effect: {layer23_results['content_effect'] - layer20_results['content_effect']:+.4f}")

# Winner
if layer23_results['content_effect'] < layer20_results['content_effect']:
    print(f"\nWinner: Layer 23 (lower content effect)")
elif layer20_results['content_effect'] < layer23_results['content_effect']:
    print(f"\nWinner: Layer 20 (lower content effect)")
else:
    print(f"\nTie: Both have same content effect")

# Save results
with open('validation_comparison.json', 'w') as f:
    json.dump({
        'layer_20': layer20_results,
        'layer_23': layer23_results
    }, f, indent=2)

print("\nResults saved to: validation_comparison.json")