# ======================================
# Comparing Retrieval Strategies
# ======================================

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load pre-built vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Test query
query = "What programming paradigms does Python support?"

print("=== Strategy 1: Similarity Search (Default) ===")
docs_similarity = vectorstore.similarity_search(query, k=3)
for i, doc in enumerate(docs_similarity, 1):
    print(f"\n{i}. {doc.page_content[:150]}...")

print("\n\n=== Strategy 2: MMR (Maximum Marginal Relevance) ===")
print("(Returns diverse results, avoiding redundancy)")
docs_mmr = vectorstore.max_marginal_relevance_search(query, k=3, fetch_k=10)
for i, doc in enumerate(docs_mmr, 1):
    print(f"\n{i}. {doc.page_content[:150]}...")

print("\n\n=== Strategy 3: Similarity with Score ===")
print("(Shows confidence scores)")
docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
for i, (doc, score) in enumerate(docs_with_scores, 1):
    print(f"\n{i}. Score: {score:.4f}")
    print(f"   {doc.page_content[:150]}...")