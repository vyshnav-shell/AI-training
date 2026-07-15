import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------
# Hardcoded Prompt
# -------------------------------------------------

PROMPT = "Give three tips for staying healthy."

# -------------------------------------------------
# Base Model
# -------------------------------------------------

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# -------------------------------------------------
# Fine-tuned Model
# -------------------------------------------------

FINETUNED_MODEL = "./qwen_merged"


def generate_response(model_path, model_name):
    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    ).to("cpu")

    model.eval()

    messages = [
        {"role": "user", "content": PROMPT}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt")

    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    end = time.perf_counter()

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    print("\nPrompt:")
    print(PROMPT)

    print("\nResponse:")
    print(response.strip())

    print(f"\nInference Time: {end-start:.3f} seconds")



# Compare Both Models


generate_response(BASE_MODEL, "BASE QWEN MODEL")

generate_response(FINETUNED_MODEL, "FINE-TUNED LORA MODEL")