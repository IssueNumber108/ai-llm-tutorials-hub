# ======================================
# Understanding Embeddings
# ======================================

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for different sentences
sentences = [
    "The dog is playing in the garden",
    "A puppy is running in the park",
    "The car is parked on the street",
    "Machine learning is a subset of AI"
]

embeddings = model.encode(sentences)

print(f"Shape of each embedding: {embeddings[0].shape}")
print(f"Embedding is a vector of {len(embeddings[0])} numbers")

# Calculate similarity (cosine similarity)
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare similarities
print("\n=== Similarity Scores ===")
print(f"Dog vs Puppy: {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
print(f"Dog vs Car: {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
print(f"Dog vs AI: {cosine_similarity(embeddings[0], embeddings[3]):.3f}")

# Exercise for learner:
# Add more sentences and observe similarity scores
# Try sentences in different languages (model is multilingual!)