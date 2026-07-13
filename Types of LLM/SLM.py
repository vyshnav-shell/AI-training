from transformers import pipeline

def run_slm(text):

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    result = classifier(text)[0]

    print("SLM Output:")
    print(result)


run_slm("This movie is amazing!")