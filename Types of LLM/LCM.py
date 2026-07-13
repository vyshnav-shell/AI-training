from sentence_transformers import SentenceTransformer, util

def run_lcm(text):

    concepts = [
    "Cybersecurity",
    "Cloud Computing",
    "Artificial Intelligence",
    "Finance",
    "Healthcare",
    "Education",
    "Legal",
    "Marketing"
]

    model = SentenceTransformer("all-MiniLM-L6-v2")

    text_embedding = model.encode(text, convert_to_tensor=True)

    concept_embeddings = model.encode(
        concepts,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        text_embedding,
        concept_embeddings
    )[0]

    best = scores.argmax()

    print("LCM Output:")
    print(concepts[int(best)])


examples = [
    "The central bank increased interest rates to control inflation.",
    "Doctors developed a new vaccine for infectious diseases.",
    "The football team won the championship after extra time.",
    "Researchers trained a deep learning model on millions of images.",
    "Students submitted their assignments through the university portal."
]

for text in examples:
    run_lcm(text)