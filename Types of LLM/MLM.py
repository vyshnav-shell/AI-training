from transformers import pipeline

def run_mlm():

    fill_mask = pipeline(
        "fill-mask",
        model="bert-base-uncased"
    )

    result = fill_mask("The capital of France is [MASK].")

    print("MLM Output:")
    print(result[0])


run_mlm()