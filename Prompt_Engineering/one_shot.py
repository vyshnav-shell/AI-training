import ollama

prompt = """
Example:
English: Hello
French: Bonjour

Now translate:

English: Good Night
French:
"""

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("One-Shot Prompting Output:\n")
print(response["message"]["content"])