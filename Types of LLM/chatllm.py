from transformers import pipeline

def run_chat_llm():

    chatbot = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-small"
    )

    result = chatbot(
        "Hello, how are you?",
        max_new_tokens=30
    )

    print("\n===== Chat LLM =====")
    print(result[0]["generated_text"])

run_chat_llm()