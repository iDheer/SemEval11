# File: check_models.py

import os
import google.generativeai as genai

# Load the API key from the environment variable for security
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("FATAL ERROR: The 'GEMINI_API_KEY' environment variable is not set.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)

        print("Checking for available models that support 'generateContent'...")
        print("=" * 60)
        
        found_models = False
        for m in genai.list_models():
            # We only care about models that can be used for text generation
            if 'generateContent' in m.supported_generation_methods:
                print(f"✅ {m.name}")
                found_models = True
                
        if not found_models:
            print("\n❌ No models supporting 'generateContent' were found.")
            print("This almost certainly means your 'google-generativeai' library is outdated.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your API key and network connection.")

print("=" * 60)
print("RECOMMENDATION: Run 'pip install --upgrade --force-reinstall google-generativeai' to fix.")