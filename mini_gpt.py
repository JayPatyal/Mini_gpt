import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
input_text = "Hello, how are you today?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)

# Decode generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
