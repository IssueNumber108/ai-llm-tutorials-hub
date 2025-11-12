# ======================================
# RAG with Multiple Document Sources
# ======================================

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import os

# Create multiple knowledge sources
os.makedirs("data/knowledge_base", exist_ok=True)

# Document 1: Python basics
with open("data/knowledge_base/python_basics.txt", "w") as f:
    f.write("""
    Python is an interpreted, high-level programming language.
    It uses indentation for code blocks instead of braces.
    Variables in Python are dynamically typed.
    """)

# Document 2: Python libraries
with open("data/knowledge_base/python_libraries.txt", "w") as f:
    f.write("""
    NumPy provides support for large, multi-dimensional arrays.
    Pandas is used for data manipulation and analysis.
    Matplotlib enables data visualization and plotting.
    Scikit-learn implements machine learning algorithms.
    """)

# Document 3: Python best practices
with open("data/knowledge_base/python_practices.txt", "w") as f:
    f.write("""
    Follow PEP 8 style guide for consistent code formatting.
    Use virtual environments to manage project dependencies.
    Write docstrings for functions and classes.
    Implement unit tests for code reliability.
    """)

# Load all documents from directory
print("📚 Loading documents from directory...")
loader = DirectoryLoader(
    "data/knowledge_base",
    glob="**/*.txt",
    loader_cls=TextLoader
)
documents = loader.load()
print(f"   Loaded {len(documents)} documents")

# Add metadata to track source
for doc in documents:
    doc.metadata["source_file"] = os.path.basename(doc.metadata["source"])

# Chunk documents
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"   Created {len(chunks)} chunks across all documents")

# Create vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_multi_db"
)

# Test retrieval from different sources
print("\n" + "=" * 60)
print("🔍 Testing cross-document retrieval\n")

test_queries = [
    "What is NumPy used for?",
    "How does Python handle code blocks?",
    "What should I use for managing dependencies?"
]

for query in test_queries:
    print(f"❓ Query: {query}")
    results = vectorstore.similarity_search(query, k=2)

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "Unknown")
        print(f"   {i}. From [{source}]: {doc.page_content[:80]}...")
    print()

print("💡 Notice how the system retrieves from the most relevant document!")