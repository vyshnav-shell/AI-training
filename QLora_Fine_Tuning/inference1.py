import time
import math
import argparse
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm

# =====================================================
# CONFIGURATION
# =====================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FT_MODEL = "./qwen_merged_fp16"

SEED = 42
NUM_EXAMPLES = 3000
TEST_SIZE = 0.05

# =====================================================
# BUILD EVALUATION DATASET
# =====================================================

def build_eval_dataset(num_samples):

    dataset = load_dataset(
        "tatsu-lab/alpaca",
        split="train"
    )

    dataset = (
        dataset
        .shuffle(seed=SEED)
        .select(range(NUM_EXAMPLES))
    )

    split = dataset.train_test_split(
        test_size=TEST_SIZE,
        seed=SEED
    )

    eval_dataset = split["test"]

    if num_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(num_samples))

    return eval_dataset


# =====================================================
# BUILD PROMPT
# =====================================================

def build_prompt(tokenizer, instruction, input_text):

    user_prompt = instruction

    if input_text:
        user_prompt += "\n\n" + input_text

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# =====================================================
# LOAD MODELS
# =====================================================

def load_models(device):

    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(FT_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading Base Model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype
    ).to(device)

    base_model.eval()

    print("Base Model Loaded")

    print("\nLoading Fine-tuned Model...")

    ft_model = AutoModelForCausalLM.from_pretrained(
        FT_MODEL,
        torch_dtype=dtype
    ).to(device)

    ft_model.eval()

    print("Fine-tuned Model Loaded")

    return tokenizer, base_model, ft_model

# =====================================================
# GENERATE RESPONSE
# =====================================================

def generate(model, tokenizer, prompt, device, max_new_tokens):

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    start = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    elapsed = time.time() - start

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )

    tokens_per_second = len(generated_ids) / elapsed

    return response, elapsed, tokens_per_second


# =====================================================
# PERPLEXITY
# =====================================================

def calculate_perplexity(model, tokenizer, prompts, references, device):

    total_loss = 0
    total_tokens = 0

    for prompt, reference in zip(prompts, references):

        full_text = prompt + reference + tokenizer.eos_token

        enc = tokenizer(
            full_text,
            return_tensors="pt"
        ).to(device)

        prompt_len = len(tokenizer(prompt)["input_ids"])

        labels = enc.input_ids.clone()

        labels[:, :prompt_len] = -100

        with torch.no_grad():

            outputs = model(
                **enc,
                labels=labels
            )

        completion_tokens = (labels != -100).sum().item()

        total_loss += outputs.loss.item() * completion_tokens

        total_tokens += completion_tokens

    avg_loss = total_loss / total_tokens

    return math.exp(avg_loss)


# =====================================================
# ROUGE
# =====================================================

def calculate_rouge(predictions, references):

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    rouge1 = []
    rouge2 = []
    rougeL = []

    for pred, ref in zip(predictions, references):

        scores = scorer.score(ref, pred)

        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougeL.append(scores["rougeL"].fmeasure)

    return (
        np.mean(rouge1),
        np.mean(rouge2),
        np.mean(rougeL)
    )


# =====================================================
# BLEU
# =====================================================

def calculate_bleu(predictions, references):

    bleu = BLEU()

    score = bleu.corpus_score(
        predictions,
        [references]
    )

    return score.score


# =====================================================
# EVALUATE MODEL
# =====================================================

def evaluate_model(model,
                   tokenizer,
                   prompts,
                   references,
                   device,
                   max_new_tokens):

    predictions = []

    speeds = []

    for prompt in tqdm(prompts):

        prediction, _, tok_sec = generate(
            model,
            tokenizer,
            prompt,
            device,
            max_new_tokens
        )

        predictions.append(prediction)

        speeds.append(tok_sec)

    ppl = calculate_perplexity(
        model,
        tokenizer,
        prompts,
        references,
        device
    )

    rouge1, rouge2, rougeL = calculate_rouge(
        predictions,
        references
    )

    bleu = calculate_bleu(
        predictions,
        references
    )

    return {
        "predictions": predictions,
        "perplexity": ppl,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bleu": bleu,
        "speed": np.mean(speeds)
    }


# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_samples",
        type=int,
        default=15,
        help="Number of evaluation samples"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150
    )

    parser.add_argument(
        "--show_examples",
        type=int,
        default=3
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nUsing Device : {device.upper()}")

    # -------------------------------------------------
    # Load Models
    # -------------------------------------------------

    tokenizer, base_model, ft_model = load_models(device)

    # -------------------------------------------------
    # Build Evaluation Dataset
    # -------------------------------------------------

    print(f"\nBuilding Evaluation Set ({args.num_samples} samples)...")

    eval_dataset = build_eval_dataset(args.num_samples)

    prompts = []
    references = []

    for sample in eval_dataset:

        prompt = build_prompt(
            tokenizer,
            sample["instruction"],
            sample["input"]
        )

        prompts.append(prompt)

        references.append(sample["output"])

    # -------------------------------------------------
    # Evaluate Base Model
    # -------------------------------------------------

    print("\nEvaluating Base Model...\n")

    base = evaluate_model(
        base_model,
        tokenizer,
        prompts,
        references,
        device,
        args.max_new_tokens
    )

    # -------------------------------------------------
    # Evaluate Fine-tuned Model
    # -------------------------------------------------

    print("\nEvaluating Fine-tuned Model...\n")

    ft = evaluate_model(
        ft_model,
        tokenizer,
        prompts,
        references,
        device,
        args.max_new_tokens
    )

    # -------------------------------------------------
    # Show Sample Outputs
    # -------------------------------------------------

    print("\n")
    print("=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)

    for i in range(min(args.show_examples, len(eval_dataset))):

        print(f"\nExample {i+1}")

        print("-" * 80)

        print("Instruction:")
        print(eval_dataset[i]["instruction"])

        if eval_dataset[i]["input"]:
            print("\nInput:")
            print(eval_dataset[i]["input"])

        print("\nGround Truth:")
        print(references[i])

        print("\nBase Model:")
        print(base["predictions"][i])

        print("\nFine-tuned Model:")
        print(ft["predictions"][i])

        print()

    # -------------------------------------------------
    # Final Results
    # -------------------------------------------------

    print("\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<30}"
        f"{'Base':>15}"
        f"{'Fine-tuned':>15}"
    )

    print("-" * 60)

    print(
        f"{'Perplexity':<30}"
        f"{base['perplexity']:>15.4f}"
        f"{ft['perplexity']:>15.4f}"
    )

    print(
        f"{'BLEU':<30}"
        f"{base['bleu']:>15.2f}"
        f"{ft['bleu']:>15.2f}"
    )

    print(
        f"{'ROUGE-1':<30}"
        f"{base['rouge1']:>15.4f}"
        f"{ft['rouge1']:>15.4f}"
    )

    print(
        f"{'ROUGE-2':<30}"
        f"{base['rouge2']:>15.4f}"
        f"{ft['rouge2']:>15.4f}"
    )

    print(
        f"{'ROUGE-L':<30}"
        f"{base['rougeL']:>15.4f}"
        f"{ft['rougeL']:>15.4f}"
    )

    print(
        f"{'Generation Speed(tok/s)':<30}"
        f"{base['speed']:>15.2f}"
        f"{ft['speed']:>15.2f}"
    )

    print("=" * 80)

    print("\nEvaluation Completed Successfully.")


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    main()