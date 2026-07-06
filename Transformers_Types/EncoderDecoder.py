from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pretrained T5-base tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Simple input text to summarize (T5 requires the "summarize:" prefix)
text = """
The city council announced a new plan to expand public transportation
across the metro area. The plan includes adding 15 new bus routes,
upgrading existing train stations, and introducing electric buses to
reduce emissions. Officials say the project will take three years to
complete and is expected to reduce daily traffic congestion significantly.
"""

input_text = "summarize: " + text

# Tokenize input
inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

# Generate summary
output = model.generate(
    **inputs,
    max_new_tokens=40,
    num_beams=4,
    early_stopping=True
)

# Decode and print
summary = tokenizer.decode(output[0], skip_special_tokens=True)
print("Original Text:", text.strip())
print("\nSummary:", summary)