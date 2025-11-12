# ======================================
# Document Processing Basics
# ======================================


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Create a sample document
sample_text = """
Artificial Intelligence (AI) is transforming industries worldwide.
Machine learning, a subset of AI, enables computers to learn from data.

Deep learning uses neural networks with multiple layers. These networks
can process images, text, and audio with remarkable accuracy.

Applications include:
- Healthcare: Disease diagnosis and drug discovery
- Finance: Fraud detection and algorithmic trading
- Transportation: Self-driving cars and traffic optimization
- Education: Personalized learning and automated grading
"""

# Save to file
os.makedirs("../notebooks/data", exist_ok=True)
with open("../notebooks/data/sample_ai.txt", "w") as f:
    f.write(sample_text)

# Load document
loader = TextLoader("../notebooks/data/sample_ai.txt")
documents = loader.load()

print(f"Loaded {len(documents)} document(s)")
print(f"Content length: {len(documents[0].page_content)} characters")

# Now split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # Maximum chunk size
    chunk_overlap=50,    # Overlap between chunks
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(documents)

print(f"\n=== Chunking Results ===")
print(f"Number of chunks: {len(chunks)}")
print(f"\n--- Chunk 1 ---")
print(chunks[0].page_content)
print(f"\n--- Chunk 2 ---")
print(chunks[1].page_content)

# Notice the overlap? That's intentional!