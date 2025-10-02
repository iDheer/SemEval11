from huggingface_hub import login
login()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
print("Success!")