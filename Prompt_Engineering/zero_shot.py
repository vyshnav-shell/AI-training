import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Translate the following sentence into French: 'Good morning, everyone!'Return only translated output nothing else is needed"
        }
    ]
)

print("Zero-Shot Prompting Output:\n")
print(response["message"]["content"])