from transformers import pipeline

def run_lam(instruction):

    actions = [
        "Open App",
        "Send Email",
        "Search Web",
        "Play Music",
        "Set Reminder",
        "Book Ticket",
        "Unknown"
    ]

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

    result = classifier(
        instruction,
        candidate_labels=actions
    )

    print("===== Large Action Model (LAM) =====")
    print("Instruction :", instruction)
    print("Predicted Action :", result["labels"][0])
    print("Confidence :", round(result["scores"][0], 4))


# Example
run_lam("Remind me to attend the meeting at 3 PM.")