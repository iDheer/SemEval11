import json
import ollama
import random
import os

# --- Configuration ---
# The "Professor": A powerful local model to generate the training data.
TEACHER_MODEL = 'deepseek-r1:8b'
CACHE_FILE = 'scot_cache.jsonl'

# --- Caching Functions (Unchanged) ---
def load_cache():
    if not os.path.exists(CACHE_FILE): return {}
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = {json.loads(line)['id']: json.loads(line) for line in f}
        print(f"Loaded {len(cache)} items from cache.")
        return cache
    except (json.JSONDecodeError, IOError): return {}

def save_to_cache(item):
    with open(CACHE_FILE, 'a') as f:
        f.write(json.dumps(item) + '\n')

# --- Core Logic ---
def generate_symbolic_cot(syllogism_obj, cache):
    """
    Uses the local 'deepseek-r1:8b' TEACHER model to generate the S-CoT data.
    """
    item_id = syllogism_obj['id']
    if item_id in cache and 'symbolic_cot' in cache[item_id]:
        print(f"Using cached S-CoT for ID: {item_id}")
        return cache[item_id]['symbolic_cot']

    print(f"Generating S-CoT for ID: {item_id} using local teacher: {TEACHER_MODEL}...")
    syllogism_text = syllogism_obj['syllogism']
    
    prompt = f"""
    You are an expert in formal logic. Your task is to analyze a syllogism, convert it into its symbolic form, and then determine its validity based on that form. Do not be influenced by the content of the syllogism.

    Syllogism: "{syllogism_text}"

    Perform the following steps:
    1.  **Identify Terms:** Identify the three key terms in the syllogism.
    2.  **Assign Symbols:** Assign the symbols A, B, and C to these terms.
    3.  **Symbolic Translation:** Translate the two premises and the conclusion into symbolic form (e.g., "All A are B," "No B are C," "Some A are not C").
    4.  **Logical Analysis:** Briefly explain the reasoning process based on the symbolic form to determine if the conclusion logically follows from the premises.
    5.  **Final Verdict:** State the final validity clearly.

    Provide the output in the following format:
    <reasoning>
    Terms: [Term 1, Term 2, Term 3]
    Symbols: [A = Term 1, B = Term 2, C = Term 3]
    Premise 1: [Symbolic form of premise 1]
    Premise 2: [Symbolic form of premise 2]
    Conclusion: [Symbolic form of conclusion]
    Analysis: [Your step-by-step logical analysis]
    Validity: [True or False]
    </reasoning>
    """
    
    try:
        # This now calls your local Ollama server
        response = ollama.chat(
            model=TEACHER_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.0}
        )
        generated_cot = response['message']['content']
        
        cache_item = {'id': item_id, 'symbolic_cot': generated_cot}
        save_to_cache(cache_item)
        cache[item_id] = cache_item
        
        return generated_cot
    except Exception as e:
        print(f"Error calling Ollama for ID {item_id}: {e}")
        return None

# --- The rest of the file (create_subtask2_example and the main block) is UNCHANGED ---
def create_subtask2_example(syllogism_obj):
    # This function is the same as before
    irrelevant_premises = ["All stars are celestial bodies.", "Some cats have fur."]
    original_premises = [p.strip() for p in syllogism_obj['syllogism'].split('.') if p.strip()]
    conclusion = original_premises[-1].replace("Therefore, ", "")
    premises = original_premises[:-1]
    irrelevant_premise = random.choice(irrelevant_premises)
    insert_pos = random.randint(0, len(premises))
    noisy_premises = premises[:]
    noisy_premises.insert(insert_pos, irrelevant_premise)
    new_syllogism_text = ". ".join(noisy_premises) + f". Therefore, {conclusion}."
    return {"id": f"{syllogism_obj['id']}-sub2", "syllogism": new_syllogism_text, "validity": syllogism_obj['validity'], "plausibility": syllogism_obj['plausibility'], "relevant_premises": premises}

if __name__ == "__main__":
    scot_cache = load_cache()
    with open('pilot_dataset.json', 'r') as f: pilot_data = json.load(f)
    enriched_data_st1 = []
    for item in pilot_data:
        symbolic_cot = generate_symbolic_cot(item, scot_cache)
        if symbolic_cot:
            enriched_item = item.copy(); enriched_item['symbolic_cot'] = symbolic_cot; enriched_data_st1.append(enriched_item)
    with open('training_data_st1_scot.jsonl', 'w') as f:
        for item in enriched_data_st1: f.write(json.dumps(item) + '\n')
    print("Subtask 1 dataset created successfully.")
    enriched_data_st2 = []
    for item in pilot_data:
        subtask2_item = create_subtask2_example(item)
        symbolic_cot = generate_symbolic_cot(subtask2_item, scot_cache)
        if symbolic_cot:
            enriched_item = subtask2_item.copy(); enriched_item['symbolic_cot'] = symbolic_cot; enriched_data_st2.append(enriched_item)
    with open('training_data_st2_scot.jsonl', 'w') as f:
        for item in enriched_data_st2: f.write(json.dumps(item) + '\n')
    print("Subtask 2 dataset created successfully.")