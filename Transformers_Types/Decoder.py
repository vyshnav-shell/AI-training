from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pretrained BLOOMZ-560M tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

# Simple input prompt
prompt = "Once upon a time,"

# Tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# Generate text
output = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Prompt:", prompt)
print("Generated:", generated_text)