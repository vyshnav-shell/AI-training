import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Model Path

MODEL_PATH = "bert_imdb_model"

# Load Tokenizer

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load Model

print("Loading model...")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Model loaded successfully!\n")

# Label Mapping
label_map = {
    0: "Negative ",
    1: "Positive "
}


def predict_sentiment(review):

    inputs = tokenizer(
        review,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

        prediction = torch.argmax(outputs.logits, dim=1).item()

        probabilities = torch.softmax(outputs.logits, dim=1)

        confidence = probabilities[0][prediction].item()

    return label_map[prediction], confidence



print("BERT Movie Review Sentiment Analysis")


while True:

    review = input("\nEnter a movie review ('quit' to exit):\n> ")

    if review.lower() == "quit":
        
        break

    sentiment, confidence = predict_sentiment(review)

    print("\nPrediction :", sentiment)
    print(f"Confidence : {confidence:.2%}")