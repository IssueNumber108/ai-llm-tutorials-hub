# =======================================================
# RAG with Ollama + LangChain 1.0.5 + Chroma
# Complete RAG Pipeline Tutorial
# =======================================================

# --- STEP 0: Install dependencies (run once) ---
# Uncomment if running for the first time:
# !pip install langchain langchain-ollama langchain-community langchain-chroma chromadb sentence-transformers langchain-huggingface

print("🚀 Building your first RAG system!\n")

# --- Step 1: Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- Step 2: Prepare sample document ---
print("📄 Step 1: Creating knowledge base...")
os.makedirs("data", exist_ok=True)

sample_text = """
Generative AI refers to artificial intelligence systems capable of generating text, images, 
code, or other data. Unlike traditional AI that classifies or predicts, generative AI creates 
novel content based on patterns learned from training data.

Retrieval-Augmented Generation (RAG) is a framework that significantly improves the factual 
accuracy of large language models. RAG works by retrieving relevant documents from a database 
and injecting them into the LLM's context before generating a response.

The key components of RAG are:
1. Document Loader: Ingests documents from various sources (PDFs, text files, databases)
2. Text Splitter: Breaks documents into smaller chunks for efficient processing
3. Embedding Model: Converts text chunks into numerical vectors that capture semantic meaning
4. Vector Database: Stores embeddings and enables fast similarity search
5. Retriever: Finds the most relevant document chunks for a given query
6. LLM: Generates the final answer using the retrieved context

RAG combines information retrieval and text generation into one powerful pipeline. This approach
reduces hallucinations, enables the use of up-to-date information, and allows AI systems to 
work with proprietary or domain-specific knowledge without expensive fine-tuning.

Benefits of RAG include:
- Grounded responses based on actual documents
- Transparency through source attribution
- Cost-effectiveness compared to model fine-tuning
- Ability to update knowledge by simply adding new documents
- Privacy preservation by keeping data local
"""

with open("data/sample_doc.txt", "w") as f:
    f.write(sample_text)
print("   ✅ Knowledge base created\n")

# --- Step 3: Load and chunk document ---
print("✂️  Step 2: Loading and chunking documents...")
loader = TextLoader("data/sample_doc.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
print(f"   ✅ Created {len(chunks)} chunks from the document\n")

# Optional: Inspect a chunk
print("   📋 Sample chunk preview:")
print(f"   {chunks[0].page_content[:200]}...\n")

# --- Step 4: Create embeddings ---
print("🧮 Step 3: Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}, # Change to 'cuda' if you have GPU
    encode_kwargs={'normalize_embeddings': True}  # Optional: improves retrieval
)
print("   ✅ Embedding model loaded (384-dimensional vectors)\n")

# --- Step 5: Store in vector DB (Chroma) ---
print("💾 Step 4: Building vector database...")
vectordb = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("   ✅ Vector database created and persisted\n")

# --- Step 6: Initialize LLM (Ollama) ---
print("🤖 Step 5: Connecting to Ollama...")
print("   ⚠️  Make sure Ollama is running: 'ollama serve'")
llm = OllamaLLM(
    model="mistral",
    temperature=0.2
)
print("   ✅ LLM connected (using Mistral model)\n")

# --- Step 7: Build RAG chain (LangChain 1.0.5 LCEL approach) ---
print("🔗 Step 6: Creating RAG chain...")

# ✅ Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# ✅ Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:

{context}

Question: {question}

Answer:
""")

# ✅ Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ✅ Build the RAG chain using LCEL (LangChain Expression Language)
rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

print("   ✅ RAG pipeline ready!\n")

# --- Step 8: Ask questions ---
print("=" * 70)
print("💬 Now let's test the RAG system with questions!\n")

questions = [
    "What is RAG and why is it important?",
    "What are the key components of a RAG system?",
    "What are the benefits of using RAG?"
]

for i, query in enumerate(questions, 1):
    print(f"{'='*70}")
    print(f"❓ Question {i}: {query}\n")

    # ✅ Invoke the chain
    answer = rag_chain.invoke(query)

    print(f"✅ Answer:\n{answer}\n")

    # ✅ Get source documents separately for display
    source_docs = retriever.invoke(query)
    print("📚 Source documents used:")
    for j, doc in enumerate(source_docs, 1):
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"   {j}. {preview}...")
    print()

print("=" * 70)
print("🎉 RAG demonstration complete!")

print("\n💡 Next steps:")
print("   1. Try your own questions by modifying the 'questions' list")
print("   2. Add your own documents to the data/ folder")
print("   3. Experiment with different chunk sizes and overlap")
print("   4. Try different Ollama models (llama3, gemma, etc.)")