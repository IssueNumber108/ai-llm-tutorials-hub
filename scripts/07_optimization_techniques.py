# ======================================
# Compare and Optimize RAG Performance
# ======================================

import time
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Enhanced Performance Test Function
def measure_retrieval_speed(vectorstore, query, k=5, num_runs=10):
    """Measure retrieval time with statistics"""
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = vectorstore.similarity_search(query, k=k)
        times.append(time.time() - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }

def main():
    """Main execution function"""

    # Load sample data
    script_dir = Path(__file__).parent
    data_file = script_dir / "data" / "python_info.txt"

    if not data_file.exists():
        print(f"❌ Error: Data file not found at {data_file}")
        print("Please ensure the file exists before running this script.")
        return

    print("📁 Loading documents...")
    loader = TextLoader(str(data_file))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    print(f"✅ Loaded {len(documents)} documents, split into {len(chunks)} chunks\n")

    # Test 1: Compare ChromaDB vs FAISS
    print("=== Vector Store Comparison ===\n")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Chroma
    print("🔨 Building ChromaDB index...")
    chroma_build_start = time.time()
    chroma_store = Chroma.from_documents(chunks, embeddings)
    chroma_embedding_and_indexing_time = time.time() - chroma_build_start

    chroma_stats = measure_retrieval_speed(chroma_store, "Python applications")
    print(f"   Embedding and Indexing time: {chroma_embedding_and_indexing_time:.2f}s")
    print(f"   Retrieval: {chroma_stats['mean']*1000:.2f}ms ± {chroma_stats['std']*1000:.2f}ms")

    # FAISS
    print("\n🔨 Building FAISS index...")
    faiss_build_start = time.time()
    faiss_store = FAISS.from_documents(chunks, embeddings)
    faiss_embedding_and_indexing_time = time.time() - faiss_build_start

    faiss_stats = measure_retrieval_speed(faiss_store, "Python applications")
    print(f"   Embedding and Indexing time: {faiss_embedding_and_indexing_time:.2f}s")
    print(f"   Retrieval: {faiss_stats['mean']*1000:.2f}ms ± {faiss_stats['std']*1000:.2f}ms")
    print(f"   ⚡ FAISS is {chroma_stats['mean']/faiss_stats['mean']:.1f}x faster!\n")

    # Test 2: Chunk Size Impact
    print("=== Chunk Size Impact ===\n")

    chunk_configs = [
        {'size': 100, 'overlap': 20},
        {'size': 300, 'overlap': 50},
        {'size': 500, 'overlap': 75},
        {'size': 1000, 'overlap': 100}
    ]

    results = []
    for config in chunk_configs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['size'],
            chunk_overlap=config['overlap']
        )
        test_chunks = splitter.split_documents(documents)

        test_store = FAISS.from_documents(test_chunks, embeddings)
        stats = measure_retrieval_speed(test_store, "Python", k=3, num_runs=5)

        results.append({
            'chunk_size': config['size'],
            'num_chunks': len(test_chunks),
            'avg_time': stats['mean']
        })

        print(f"   Size {config['size']:4d}: {len(test_chunks):2d} chunks, {stats['mean']*1000:6.2f}ms")

    # Test 3: k Parameter Impact
    print("\n=== Top-k Retrieval Impact ===\n")

    k_values = [1, 3, 5, 10, 20]
    for k in k_values:
        stats = measure_retrieval_speed(faiss_store, "Python applications", k=k, num_runs=5)
        print(f"   k={k:2d}: {stats['mean']*1000:6.2f}ms")

    # Test 4: Different Embedding Models
    print("\n=== Embedding Model Comparison ===\n")

    models = [
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast, 384 dim
        "sentence-transformers/all-mpnet-base-v2",  # Better quality, 768 dim
    ]

    for model_name in models:
        print(f"🔍 Testing {model_name.split('/')[-1]}...")

        # Embedding time
        embed_start = time.time()
        test_embeddings = HuggingFaceEmbeddings(model_name=model_name)
        test_store = FAISS.from_documents(chunks[:50], test_embeddings)  # Use subset
        embedding_and_indexing_time = time.time() - embed_start

        # Retrieval time
        stats = measure_retrieval_speed(test_store, "Python", k=5, num_runs=5)

        print(f"  Embedding and Indexing time: {embedding_and_indexing_time:.2f}s")
        print(f"   Retrieval: {stats['mean']*1000:.2f}ms\n")

    # Test 5: Similarity Score Distribution
    print("=== Similarity Score Analysis ===\n")

    results = faiss_store.similarity_search_with_score("Python applications", k=10)
    print("Top 10 similarity scores:")
    for i, (doc, score) in enumerate(results, 1):
        preview = doc.page_content[:60].replace('\n', ' ')
        print(f"   {i:2d}. Score: {score:.4f} - {preview}...")

    # Summary
    print("\n" + "="*60)
    print("💡 Optimization Tips:")
    print("="*60)
    print("1. ⚡ Use FAISS for faster similarity search")
    print("2. 📏 Balance chunk size: larger = faster but less precise")
    print("3. 💾 Cache embeddings to avoid recomputation")
    print("4. 🎯 Use appropriate k value (5-10 works well for most cases)")
    print("5. 🚀 Use GPU if available: model_kwargs={'device': 'cuda'}")
    print("6. 🔍 Consider MiniLM for speed, MPNet for quality")
    print("7. 💽 Save FAISS index to disk: faiss_store.save_local('index')")
    print("8. 🔄 Use async operations for concurrent queries")
    print("="*60)

if __name__ == "__main__":
    print("\n🚀 Starting RAG Performance Optimization Tests...\n")
    main()
    print("\n✅ All tests completed!\n")