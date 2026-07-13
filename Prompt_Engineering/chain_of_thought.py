import ollama

prompt = """
Question:

A box contains 5 apples.
You buy 3 boxes.

Let's think step by step before giving the final answer.
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

print("Chain-of-Thought Prompting Output:\n")
print(response["message"]["content"])