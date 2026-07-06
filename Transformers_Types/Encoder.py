from transformers import pipeline

# Load Pretrained Models

# Sentiment Classification Pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="textattack/bert-base-uncased-imdb"
)

# Named Entity Recognition Pipeline

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)


# Sentiment Classification


texts = [
    "I love this movie, it was fantastic!",
    "This film was terrible and boring.",
    "Absolutely wonderful experience, highly recommend.",
    "Worst product I have ever bought."
]



predictions = classifier(texts)

for text, prediction in zip(texts, predictions):
    sentiment = prediction["label"].replace("LABEL_", "")

    # Convert 1/0 into Positive/Negative
    if sentiment == "1":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    print(f"Text: {text}")
    print(f"Prediction: {sentiment}")
    print()


# Named Entity Recognition

text = "I absolutely love watching movies directed by Christopher Nolan in Los Angeles."


print(f"Text: {text}")

entities = ner_pipeline(text)

print("\nEntities Found:")

for entity in entities:
    print(f"{entity['word']} --> {entity['entity_group']}")