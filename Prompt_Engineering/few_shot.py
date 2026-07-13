import ollama

prompt = """
Translate English to French.

Example 1:
English: Hello
French: Bonjour

Example 2:
English: Thank You
French: Merci

Example 3:
English: Good Morning
French: Bonjour

Now translate:

English: See You Tomorrow
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

print("Few-Shot Prompting Output:\n")
print(response["message"]["content"])