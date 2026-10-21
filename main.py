from embedder import embedder
from dotenv import load_dotenv

load_dotenv()

print(embedder("Is drinking water available at the school?").shape)

vectors = embedder([
    "Is drinking water available at the school?",
    "क्या विद्यालय में पीने के पानी की व्यवस्था है?"
])

print(vectors.shape)
print("Similarity:", float(vectors[0] @ vectors[1]))