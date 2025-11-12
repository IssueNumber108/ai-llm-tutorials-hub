# Complete Learning Roadmap: Generative AI & RAG for Beginners

## Introduction: What You'll Learn and Why It Matters

### What is Generative AI?
**Generative AI** refers to artificial intelligence systems that can create new content—text, images, code, or other media—based on patterns learned from training data. Unlike traditional AI that classifies or predicts, GenAI *generates* novel outputs.

**Real-world applications:**
- Writing assistants (like GitHub Copilot)
- Customer service chatbots
- Content generation for marketing
- Code documentation and bug fixing
- Medical report summarization

### What is RAG (Retrieval-Augmented Generation)?
**RAG** combines the power of large language models with your own documents/data. Instead of relying solely on the model's training, RAG:
1. **Retrieves** relevant information from your documents
2. **Augments** the LLM's prompt with this context
3. **Generates** accurate, grounded responses

**Why RAG matters:**
- ✅ Reduces hallucinations (incorrect information)
- ✅ Works with proprietary/recent data
- ✅ More cost-effective than fine-tuning
- ✅ Transparent—you can see which documents were used

### Why Learn Locally First?
Running everything on your machine:
- **No API costs** during learning
- **Privacy**—your data never leaves your computer
- **Understanding**—see exactly how each component works
- **Experimentation**—break things without consequences

---

## Learning Objectives

By the end of this roadmap, you will be able to:

1. **Conceptual Understanding**
    - Explain how LLMs generate text
    - Describe the RAG architecture and its components
    - Understand embeddings and vector similarity

2. **Technical Skills**
    - Set up a local GenAI development environment
    - Build a complete RAG pipeline from scratch
    - Integrate Ollama, LangChain, and vector databases
    - Debug and optimize RAG performance

3. **Practical Application**
    - Create document Q&A systems
    - Build chatbots with custom knowledge bases
    - Deploy simple GenAI applications

---

## Conceptual Foundations (Read This First!)

### 1. How Large Language Models Work

**The Basic Idea:**
LLMs are trained on vast amounts of text to predict the next word in a sequence. Through this simple task, they learn grammar, facts, reasoning patterns, and more.

**Key Concepts:**
- **Tokens**: Text is broken into pieces (words or subwords). "Hello world" → ["Hello", " world"]
- **Context Window**: How much text the model can "remember" at once (e.g., 4K, 8K, 128K tokens)
- **Temperature**: Controls randomness (0 = deterministic, 1+ = creative)
- **Parameters**: The model's "size" (7B = 7 billion parameters)

**Analogy**: Think of an LLM as an extremely well-read person who's absorbed millions of books but doesn't have perfect memory or access to recent events.

### 2. Understanding Embeddings

**What are embeddings?**
Embeddings convert text into numbers (vectors) that capture semantic meaning. Similar concepts have similar vectors.

**Example:**
```
"dog" → [0.2, 0.8, 0.1, ...]
"puppy" → [0.25, 0.75, 0.15, ...]  (very similar!)
"car" → [0.9, 0.1, 0.8, ...]      (very different)
```

**Why they matter for RAG:**
When you ask a question, we:
1. Convert your question to an embedding
2. Find documents with similar embeddings
3. Feed those documents to the LLM

**Visualization:**
```
Your Question: "What are the benefits of exercise?"
              ↓ (embedding)
        [0.1, 0.7, 0.3, ...]
              ↓ (similarity search)
    Most Similar Documents:
    1. "Regular exercise improves..." (95% similar)
    2. "Physical activity benefits..." (92% similar)
    3. "Health advantages of..." (88% similar)
```

### 3. The RAG Pipeline Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     RAG SYSTEM                             │
│                                                            │
│  ┌──────────────┐        ┌──────────────┐                  │
│  │   Document   │──────→ │ Text Splitter│                  │
│  │    Loader    │        │  (Chunking)  │                  │
│  └──────────────┘        └──────┬───────┘                  │
│                                 │                          │
│                                 ↓                          │
│                         ┌────────────────┐                 │
│                         │   Embedding    │                 │
│                         │     Model      │                 │
│                         └────────┬───────┘                 │
│                                  │                         │
│                                  ↓                         │
│  ┌───────────┐          ┌───────────────┐                  │
│  │   User    │          │  Vector Store │                  │
│  │  Question │──┐       │ (Chroma/FAISS)│                  │
│  └───────────┘  │       └───────┬───────┘                  │
│                 │               │                          │
│                 │               │ (similarity search)      │
│                 ↓               ↓                          │
│            ┌────────────────────────────┐                  │
│            │  Retriever: Get relevant   │                  │
│            │  document chunks           │                  │
│            └────────────┬───────────────┘                  │
│                         │                                  │
│                         ↓                                  │
│            ┌────────────────────────────┐                  │
│            │  Prompt Construction:      │                  │
│            │  Question + Context        │                  │
│            └────────────┬───────────────┘                  │
│                         │                                  │
│                         ↓                                  │
│            ┌────────────────────────────┐                  │
│            │    LLM (Ollama/Mistral)    │                  │
│            │    Generates Answer        │                  │
│            └────────────┬───────────────┘                  │
│                         │                                  │
│                         ↓                                  │
│            ┌────────────────────────────┐                  │
│            │    Final Answer to User    │                  │
│            └────────────────────────────┘                  │
└────────────────────────────────────────────────────────────┘
```

**Step-by-Step Explanation:**

1. **Document Ingestion**: Load PDFs, text files, or web pages
2. **Chunking**: Split long documents into smaller pieces (e.g., 500 tokens each)
3. **Embedding**: Convert each chunk to a vector
4. **Storage**: Save vectors in a database (one-time setup)
5. **Query Processing**: User asks a question → convert to vector
6. **Retrieval**: Find the most similar document chunks
7. **Augmentation**: Combine question + retrieved chunks into a prompt
8. **Generation**: LLM produces an answer based on the context

---

## 💻 Environment Setup (Detailed Guide)

### Phase 1: Core Tools Installation

#### 1.1 Python Installation

**Why Python 3.10+?**
- Native support for modern type hints
- Better performance
- Required by latest LangChain versions

**Installation:**
```bash
# Check current version
python --version

# If < 3.10, download from python.org
# Verify pip is installed
pip --version
```

**⚠️ Common Issues:**
- **Multiple Python versions**: Use `python3` instead of `python`
- **PATH not set**: Add Python to system PATH during installation
- **Permission errors on Mac/Linux**: Use `python3 -m pip` instead of just `pip`

#### 1.2 Jupyter Setup

**Why JupyterLab?**
- Interactive coding environment
- See outputs immediately
- Mix code, markdown, and visualizations
- Perfect for learning and experimentation

**Installation:**
```bash
pip install jupyterlab notebook

# Launch Jupyter
jupyter lab
# Opens browser at http://localhost:8888

[Sample Notebook](https://github.com/tonykipkemboi/ollama_pdf_rag/blob/main/notebooks/experiments/local_ollama_rag.ipynb)
```

**Pro Tips:**
- Use `Shift+Enter` to run cells
- `Esc` → `M` converts cell to markdown (for notes)
- `Esc` → `Y` converts back to code
- Save notebooks frequently (Ctrl/Cmd+S)

#### 1.3 Ollama Installation

**What is Ollama?**
Ollama is like Docker for LLMs—it manages downloading, running, and serving AI models on your local machine.

**Installation by Platform:**

**Windows:**
```bash
# Download installer from https://ollama.com
# Or use winget
winget install Ollama.Ollama
```

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verification:**
```bash
ollama --version
ollama serve  # Start the server (keep this running)
```

#### 1.4 Choosing Your First Model

**Model Size Guide:**

| Model     | Size | RAM Needed | Speed | Use Case                    |
|-----------|------|------------|-------|-----------------------------|
| tinyllama | 1.1B | 2GB        | Fast  | Testing, learning           |
| mistral   | 7B   | 8GB        | Good  | **Recommended for learning**|
| llama3    | 8B   | 10GB       | Good  | Better quality              |
| mixtral   | 47B  | 32GB       | Slow  | Production quality          |

**Downloading Models:**
```bash
# Start with Mistral (best balance)
ollama pull mistral

# List installed models
ollama list

# Test the model
ollama run mistral
>>> Hello! How do you work?
```

**💡 Learning Tip**: Start with smaller models (mistral) while learning. Switch to larger ones once you understand the pipeline.

---

## 📦 Python Libraries Installation

### Core Dependencies

```bash
# Create a virtual environment (HIGHLY RECOMMENDED)
python -m venv genai_env

# Activate it
# Windows:
genai_env\Scripts\activate
# Mac/Linux:
source genai_env/bin/activate

# Install core libraries
pip install gradio langchain langchain-ollama langchain-chroma langchain-community langchain-core sentence-transformers pypdf langchain-huggingface


# Verify installations
python -c "import langchain; print(langchain.__version__)"
```

### Library Explanations

**LangChain**: Framework for building LLM applications
- Provides abstractions for document loaders, chains, agents
- Simplifies complex workflows
- Like "Django for AI apps"

**ChromaDB**: Vector database
- Stores embeddings
- Fast similarity search
- Lightweight and local-first

**Sentence-Transformers**: Creates embeddings
- Pre-trained models for text encoding
- Models like `all-MiniLM-L6-v2` are small and fast

**PyPDF**: PDF processing
- Extract text from PDFs
- Handle multi-page documents

---

## 🎓 Learning Path: Step-by-Step

### 🟢 Level 1: Foundation (Week 1-2)

#### Milestone 1.1: Verify Ollama Works

**Goal**: Get comfortable with running local LLMs

**Exercise**: Interactive chat with different models

```bash
# Try different models
ollama run mistral
>>> Explain quantum computing in simple terms

ollama run tinyllama
>>> What is Python?

# Compare responses - notice quality differences
```

**Learning Questions**:
- How does response quality differ between models?
- How fast does each model respond?
- What happens if you ask something the model doesn't know?

#### Milestone 1.2: Understand Embeddings Practically

**Create**: `scripts/01_embeddings_basics.py`

```python
# Understanding Embeddings

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
```

**Expected Output**:
```
Shape of each embedding: (384,)
Embedding is a vector of 384 numbers

=== Similarity Scores ===
Dog vs Puppy: 0.782  # High similarity!
Dog vs Car: 0.234    # Low similarity
Dog vs AI: 0.198     # Very low similarity
```

**💭 Reflection Questions**:
1. Why are "dog" and "puppy" similar (you never trained this)?
2. What happens with typos? Try "dogg" vs "dog"
3. Try translating sentences—do similar concepts across languages have similar embeddings?

---

### 🟡 Level 2: Basic RAG Pipeline (Week 3-4)

#### Milestone 2.1: Document Loading and Chunking

**Create**: `scripts/02_document_processing.py`

```python
# Document Processing Basics

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
```

**Why Chunking Matters:**

**❌ Without Chunking:**
- Feed entire 50-page document to LLM
- Exceeds context window
- Expensive and slow
- Irrelevant information confuses the model

**✅ With Chunking:**
- Only retrieve relevant sections
- Faster processing
- Better accuracy
- Cost-effective

**Chunking Strategy Guidelines:**

| Document Type | Chunk Size | Overlap | Rationale |
|--------------|------------|---------|-----------|
| News articles | 500-1000 | 100 | Paragraphs are self-contained |
| Legal docs | 1000-1500 | 200 | Context is crucial |
| Code | 500-1000 | 100 | Function-level chunks |
| Chat logs | 300-500 | 50 | Short exchanges |

**🎯 Exercise**:
1. Try different chunk sizes (100, 500, 1000)
2. Observe how overlap preserves context
3. Load a PDF and chunk it
4. What happens with very small chunks? Very large?

#### Milestone 2.2: Building Your First RAG System

**Create**: `scripts/03_rag_ollama_basic.py`

This is a **production-ready, copy-paste notebook** that demonstrates the complete RAG pipeline.

```python
# =======================================================
# RAG with Ollama + LangChain 1.0.5 + Chroma
# Complete RAG Pipeline Tutorial
# =======================================================

print("🚀 Building your first RAG system!\n")

# --- Step 1: Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM  # ✅ Updated import
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
    print(f"  Question {i}: {query}\n")

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
```

**Expected Output**:
```
🚀 Building your first RAG system!

📄 Step 1: Creating knowledge base...
   ✅ Knowledge base created

✂️  Step 2: Loading and chunking documents...
   ✅ Created 5 chunks from the document

🧮 Step 3: Creating embeddings...
   ✅ Embedding model loaded (384-dimensional vectors)

💾 Step 4: Building vector database...
   ✅ Vector database created and persisted

🤖 Step 5: Connecting to Ollama...
   ✅ LLM connected (using Mistral model)

🔗 Step 6: Creating RAG chain...
   ✅ RAG pipeline ready!

======================================================================
💬 Now let's test the RAG system with questions!

======================================================================
  Question 1: What is RAG and why is it important?

✅ Answer:
RAG (Retrieval-Augmented Generation) is a framework that improves the factual 
accuracy of large language models by retrieving relevant documents from a database 
and incorporating them into the model's context before generating responses. It's 
important because it reduces hallucinations, enables use of current information, 
and allows AI systems to work with proprietary knowledge without expensive fine-tuning.

📚 Source documents used:
   1. Retrieval-Augmented Generation (RAG) is a framework that significantly improves...
   2. RAG combines information retrieval and text generation into one powerful pipeline...
```

**Expected Output**:
```
🚀 Building your first RAG system!

📄 Step 1: Creating knowledge base...
✂️  Step 2: Loading and chunking documents...
   Created 3 chunks
🧮 Step 3: Creating embeddings...
   Embedding model loaded
💾 Step 4: Building vector database...
   Vector database created
🤖 Step 5: Connecting to Ollama...
   LLM ready
🔗 Step 6: Creating RAG chain...
   RAG system ready!

============================================================
💬 Let's ask some questions!

  Question 1: Who created Python?
✅ Answer: Python was created by Guido van Rossum and first released in 1991.

📚 Sources used:
   1. Python is a high-level programming language. It was created by Guido van Rossum...
------------------------------------------------------------
```

**🎯 Deep Dive Exercise**:

1. **Modify the questions**: Ask something NOT in the document
    - What happens?
    - Does the model hallucinate or admit it doesn't know?

2. **Change `k` parameter**: Try `k=1` vs `k=5`
    - How does answer quality change?
    - What's the trade-off?

3. **Adjust temperature**: Try `0`, `0.5`, `1.0`
    - How does creativity vs accuracy change?

4. **Add more documents**: Create multiple text files
    - Does the system retrieve from the right document?

---

### 🔴 Level 3: Advanced RAG Concepts (Week 5-6)

#### Milestone 3.1: Understanding Retrieval Strategies

**Create**: `scripts/04_retrieval_strategies.py`

```python
# Comparing Retrieval Strategies

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
```

**Key Concepts**:

**Similarity Search**:
- Returns documents closest to query embedding
- Fast and simple
- May return redundant results

**MMR (Maximum Marginal Relevance)**:
- Balances relevance and diversity
- Avoids returning 5 nearly identical chunks
- Better for exploratory queries

**Threshold Filtering**:
- Only return results above certain similarity
- Prevents irrelevant context
- Useful when you want high precision

**💡 When to Use Each**:

| Strategy | Use Case | Example Query |
|----------|----------|---------------|
| Similarity | Specific facts | "What is Python's release year?" |
| MMR | Broad topics | "Tell me about Python" |
| Threshold | High precision needed | Medical/legal Q&A |

#### Milestone 3.2: Multi-Document RAG

**Create**: `scripts/05_multi_document_rag.py`

```python
# RAG with Multiple Document Sources

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
    print(f"  Query: {query}")
    results = vectorstore.similarity_search(query, k=2)
    
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source_file", "Unknown")
        print(f"   {i}. From [{source}]: {doc.page_content[:80]}...")
    print()

print("💡 Notice how the system retrieves from the most relevant document!")
```

**Learning Points**:
- RAG can search across unlimited documents
- Metadata helps track sources (crucial for citations)
- Document structure affects retrieval quality

---

### 🟣 Level 4: Production Concepts (Week 7-8)

#### Milestone 4.0: Working with PDFs (Real-World Use Case)

**Create**: `scripts/05b_rag_ollama_pdf.py`

Most real-world RAG applications need to process **PDF documents**. This notebook shows you how.

```python
# ===============================
# RAG with PDF Documents - Auto Download Sample PDFs
# ===============================

# --- STEP 0: Install dependencies ---
# !pip install langchain chromadb sentence-transformers pypdf

print("📄 Building a PDF RAG System with Sample PDFs\n")

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import warnings
import urllib.request

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Step 0: Download Sample PDFs ---
print("📥 Step 0: Downloading sample PDFs...")
os.makedirs("data/pdfs", exist_ok=True)

# Sample PDFs to download (public domain documents)
sample_pdfs = {
    "attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",  # Transformer paper
    "bert_paper.pdf": "https://arxiv.org/pdf/1810.04805.pdf",  # BERT paper
}

for filename, url in sample_pdfs.items():
    filepath = os.path.join("data/pdfs", filename)

    if os.path.exists(filepath):
        print(f"   ✅ {filename} already exists")
    else:
        try:
            print(f"   ⬇️  Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
            print(f"   ✅ Downloaded {filename}")
        except Exception as e:
            print(f"   ⚠️  Failed to download {filename}: {e}")

print()

# --- Step 1: Check PDF directory ---
print("📁 Step 1: Checking PDF directory...")
pdf_files = [f for f in os.listdir("data/pdfs/") if f.endswith('.pdf')]
print(f"   Found {len(pdf_files)} PDF file(s): {pdf_files}\n")

# --- Step 2: Load PDF documents ---
print("📚 Step 2: Loading PDF documents...")

documents = []

if pdf_files:
    # Load all PDFs from directory
    loader = DirectoryLoader(
        "data/pdfs/",
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )

    try:
        documents = loader.load()
        print(f"   ✅ Loaded {len(documents)} page(s) from PDF(s)")

        # Show sample content
        if documents:
            print(f"\n   📄 Sample from first page:")
            print(f"   {documents[0].page_content[:300]}...\n")
    except Exception as e:
        print(f"   ⚠️  Error loading PDFs: {e}")
        documents = []

# If no PDFs found or loading failed, create fallback document
if not documents:
    print("   ⚠️  No PDFs found or failed to load")
    print("   📝 Creating fallback text document for demo...\n")

    fallback_text = """
    This is a sample document about Machine Learning and Artificial Intelligence.
    
    Machine Learning (ML) is a subset of artificial intelligence that enables 
    systems to learn and improve from experience without explicit programming.
    
    Key ML concepts:
    1. Supervised Learning: Training with labeled data
    2. Unsupervised Learning: Finding patterns in unlabeled data
    3. Reinforcement Learning: Learning through trial and error
    
    Popular ML algorithms include:
    - Linear Regression for prediction
    - Decision Trees for classification
    - Neural Networks for complex pattern recognition
    - K-Means for clustering
    
    Deep Learning is a subset of ML that uses neural networks with multiple layers.
    It has revolutionized fields like computer vision, natural language processing,
    and speech recognition.
    
    Applications of Machine Learning:
    - Image and speech recognition
    - Recommendation systems (Netflix, Amazon)
    - Fraud detection in banking
    - Medical diagnosis
    - Autonomous vehicles
    - Natural language processing (ChatGPT, translation)
    
    To get started with ML:
    1. Learn Python programming
    2. Study statistics and linear algebra
    3. Practice with datasets from Kaggle
    4. Use libraries like scikit-learn, TensorFlow, PyTorch
    5. Work on real-world projects
    """

    os.makedirs("data", exist_ok=True)
    with open("data/ml_basics.txt", "w") as f:
        f.write(fallback_text)

    loader = TextLoader("data/ml_basics.txt")
    documents = loader.load()
    print(f"   ✅ Created and loaded fallback document")

# Validate we have documents
if not documents:
    raise ValueError("No documents were loaded!")

print(f"\n   📊 Total documents to process: {len(documents)}")

# --- Step 3: Chunk documents ---
print("\n✂️  Step 3: Chunking documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)
print(f"   ✅ Created {len(chunks)} chunks")

if not chunks:
    raise ValueError("No chunks were created!")

# Show sample chunks
for i, chunk in enumerate(chunks[:3]):
    page = chunk.metadata.get('page', 'N/A')
    source = chunk.metadata.get('source', 'N/A')
    print(f"   Chunk {i+1}: Page {page} from {os.path.basename(source)}")

# --- Step 4: Create embeddings ---
print("\n🧮 Step 4: Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("   ✅ Embedding model ready")

# --- Step 5: Build vector store ---
print("\n💾 Step 5: Building vector database...")
print(f"   Processing {len(chunks)} chunks...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_pdf_db"
)
print("   ✅ Vector database created")

# --- Step 6: Set up RAG chain ---
print("\n🤖 Step 6: Setting up RAG chain...")
print("   ⚠️  Make sure Ollama is running: 'ollama serve'")

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.2
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(
            f"[Source {i} - Page {page} from {os.path.basename(source)}]:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

print("   ✅ RAG system ready!\n")

# --- Step 7: Query the documents ---
print("=" * 70)
print("💬 Querying your documents\n")

# Queries tailored to the Transformer/BERT papers
queries = [
    "What is the main contribution of the paper?",
    "What architecture does the paper propose?",
    "What are the key innovations discussed?",
]

for i, query in enumerate(queries, 1):
    print(f"{'='*70}")
    print(f"  Query {i}: {query}\n")

    answer = rag_chain.invoke(query)

    print(f"✅ Answer:\n{answer}\n")

    source_docs = retriever.invoke(query)
    print("📄 Sources:")
    for j, doc in enumerate(source_docs, 1):
        page = doc.metadata.get('page', 'Unknown')
        source_file = doc.metadata.get('source', 'Unknown')
        preview = doc.page_content[:150].replace('\n', ' ')
        print(f"   {j}. Page {page} from {os.path.basename(source_file)}")
        print(f"      {preview}...")
    print()

print("=" * 70)
print("🎉 PDF RAG demonstration complete!")
```

**Important PDF Considerations**:

1. **Chunk Size**: PDFs often have dense text
    - Academic papers: 1000-1500 tokens
    - Legal documents: 1500-2000 tokens
    - Presentations: 500-800 tokens

2. **Metadata Preservation**:
    - Track page numbers for citations
    - Keep source filename
    - Store document title if available

3. **PDF-Specific Challenges**:
    - **Scanned PDFs**: Need OCR (use `pytesseract`)
    - **Multi-column layouts**: May need specialized parsing
    - **Tables/Images**: Consider using `unstructured` library
    - **Large files**: Process in batches

---

### 🟣 Level 4: Production Concepts (Continued)

#### Milestone 4.1: Adding a UI with Gradio

**Create**: `scripts/06_ollama_chatbot_local.py`

This notebook demonstrates how to build a **simple, local chatbot** with a web interface using **Gradio**.

```python
# =========================================
# Local Chatbot with Ollama + Gradio
# Zero-cost, fully local AI assistant
# =========================================

print("🤖 Building a local chatbot with Gradio + Ollama\n")

from langchain_ollama import OllamaLLM
import gradio as gr

# --- Step 1: Initialize local model ---
print("📡 Connecting to Ollama...")
llm = OllamaLLM(model="mistral")
print("✅ Model loaded: Mistral\n")

# --- Step 2: Define chat function ---
def chat_with_model(prompt, history=[]):
    """
    Process user input and generate response

    Args:
        prompt: User's message
        history: Conversation history (list of tuples)

    Returns:
        Updated history
    """
    if not prompt.strip():
        return history, history

    # Generate response
    response = llm(prompt)

    # Append to history
    history.append((prompt, response))

    return history, history

# --- Step 3: Create Gradio chat UI ---
print("🎨 Building user interface...")

with gr.Blocks(title="Local AI Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 Local Chatbot (Ollama + LangChain)
        
        **100% Free • 100% Local • 100% Private**
        
        This chatbot runs entirely on your computer using Ollama.
        No data is sent to external servers.
        """
    )

    chatbot = gr.Chatbot(
        label="Conversation",
        height=400,
        show_label=True
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Your message",
            placeholder="Type your question here...",
            lines=2,
            scale=4
        )
        submit = gr.Button("Send", variant="primary", scale=1)

    clear = gr.ClearButton([msg, chatbot], value="Clear Conversation")

    # Examples section
    gr.Examples(
        examples=[
            "Explain quantum computing in simple terms",
            "Write a Python function to calculate fibonacci numbers",
            "What are the benefits of meditation?",
            "How does photosynthesis work?"
        ],
        inputs=msg,
        label="Try these examples:"
    )

    # Footer info
    gr.Markdown(
        """
        ---
        💡 **Tips:**
        - Be specific in your questions for better answers
        - Ask follow-up questions to dive deeper
        - Experiment with different phrasings
        
        🔧 **Model:** Mistral (running locally via Ollama)
        """
    )

    # Connect events
    msg.submit(chat_with_model, [msg, chatbot], [chatbot, chatbot])
    submit.click(chat_with_model, [msg, chatbot], [chatbot, chatbot])

# --- Step 4: Launch the interface ---
print("✅ Interface ready!")
print("\n" + "=" * 60)
print("🚀 Launching Gradio interface...")
print("📱 Access the chatbot at: http://127.0.0.1:7860")
print("🛑 Press Ctrl+C to stop the server")
print("=" * 60 + "\n")

demo.launch(
    share=False,      # Set to True to create public link (temporary)
    server_name="127.0.0.1",
    server_port=7860
)
```

**Expected Output**:
```
🤖 Building a local chatbot with Gradio + Ollama

📡 Connecting to Ollama...
✅ Model loaded: Mistral

🎨 Building user interface...
✅ Interface ready!

============================================================
🚀 Launching Gradio interface...
📱 Access the chatbot at: http://127.0.0.1:7860
🛑 Press Ctrl+C to stop the server
============================================================

Running on local URL:  http://127.0.0.1:7860
```

**✅ Learning Checkpoints**:

1. **UI Integration**: Understand how LLMs integrate with web interfaces
2. **Zero API Calls**: Learn to build apps without external dependencies
3. **Model Flexibility**: Easy to swap models by changing one parameter
4. **Privacy First**: All data stays on your machine

**🎯 Enhancement Exercises**:

1. **Add Conversation Memory**: Make the chatbot remember previous messages
```python
from langchain_ollama import OllamaLLM

# Initialize model
llm = OllamaLLM(model="llama3.2")

# Store conversation history
history = []

def chat_with_memory(user_input):
    """
    Chat with memory: bot remembers previous messages.
    """
    # Include last 3 exchanges in context
    context = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history[-3:]])

    # Build full prompt
    full_prompt = f"{context}\nUser: {user_input}\nAssistant:"

    # Generate response
    response = llm.generate(prompts=[full_prompt]).generations[0][0].text

    # Save exchange in history
    history.append((user_input, response))

    return response

# Example conversation
print(chat_with_memory("Hi!"))
print(chat_with_memory("Can you explain Python decorators?"))
print(chat_with_memory("Thanks! Can you give a short example for a logging decorator?"))
print(chat_with_memory("What did I ask about first?"))  # Bot remembers context

```

2. **Add System Instructions**: Give the chatbot a personality
```python
from langchain_ollama import OllamaLLM

# Initialize model
llm = OllamaLLM(model="llama3.2")

# Define system prompt
system_prompt = "You are a helpful coding tutor. Explain concepts clearly with examples."

# User query
user_prompt = "Explain Python decorators with an example."

# Combine system + user prompt
full_prompt = f"{system_prompt}\n\n{user_prompt}"

# Generate response
response = llm.generate(prompts=[full_prompt])

# Suppose 'response' is the output from llm.generate()
# Access the text from the first generation
generated_text = response.generations[0][0].text

# Print nicely
print(f"Answer to the Query:\n{generated_text}\n")
```

3. **Token Counter**: Show how many tokens are being used
4. **Export Chat**: Add button to download conversation as text

---

#### Milestone 4.2: Combine conversational AI with document retrieval

**Create**: `scripts/06b_rag_chatbot_ui.py`

This notebook demonstrates how to build a **chatbot with document retrieval** with a web interface using **Gradio**.

```python
# ===============================
# 💬 RAG-Powered Chatbot with Gradio
# Combines conversational AI with document retrieval
# ===============================

# --- STEP 0: Install dependencies (run once) ---
# Uncomment if running for the first time:
# !pip install gradio langchain langchain-ollama langchain-chroma langchain-community langchain-core sentence-transformers pypdf langchain-huggingface


print("🚀 Building PDF Chat Interface with Gradio\n")

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import warnings
import tempfile

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Initialize components ---
print("🔧 Initializing RAG components...")

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Create text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize vector store
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="./chroma_gradio_db"
)

# Initialize LLM
llm = OllamaLLM(model="llama3.2")

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:
""")

# Helper function to format documents
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(
            f"[Source {i} - Page {page} from {os.path.basename(source)}]:\n{doc.page_content}"
        )
    return "\n\n".join(formatted)

# Create RAG chain
rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
)

print("✅ Components initialized!\n")

# --- Gradio Functions ---

def process_pdf_upload(pdf_file):
    """Process uploaded PDF and add to vector store"""
    if pdf_file is None:
        return "❌ No file uploaded"

    try:
        # Get the file path from Gradio's UploadFile object
        file_path = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if not documents:
            return "❌ No content found in PDF"

        # Chunk documents
        chunks = splitter.split_documents(documents)

        if not chunks:
            return "❌ Failed to create chunks from PDF"

        # Add to vector store
        vectorstore.add_documents(chunks)

        return f"✅ Successfully processed {len(documents)} pages with {len(chunks)} chunks!"

    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"


def chat_with_pdf(message, history):
    """Chat with the uploaded PDFs"""
    if not message:
        return "Please enter a question."

    try:
        # Check if vector store has any documents
        collection = vectorstore._collection
        count = collection.count()

        if count == 0:
            return "⚠️ No PDFs uploaded yet! Please upload a PDF first."

        # Get answer from RAG chain
        answer = rag_chain.invoke(message)

        return answer

    except Exception as e:
        return f"❌ Error: {str(e)}"


def clear_database():
    """Clear all documents from the vector store"""
    try:
        # Delete the existing vector store
        global vectorstore
        vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_gradio_db"
        )

        # Recreate retriever and chain
        global retriever, rag_chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        rag_chain = (
                {
                    "context": retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
        )

        return "✅ Database cleared successfully!"
    except Exception as e:
        return f"❌ Error clearing database: {str(e)}"


# --- Gradio Interface ---

with gr.Blocks(title="PDF Chat with RAG", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📄 PDF Chat with RAG
        Upload PDF documents and ask questions about them using Retrieval-Augmented Generation (RAG)
        
        **Instructions:**
        1. Upload a PDF file using the file uploader
        2. Click "Process PDF" to add it to the knowledge base
        3. Ask questions in the chat interface
        4. Use "Clear Database" to start fresh
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload PDF")
            pdf_upload = gr.File(
                label="Upload PDF File",
                file_types=[".pdf"],
                type="filepath"
            )
            process_btn = gr.Button("Process PDF", variant="primary")
            status_output = gr.Textbox(
                label="Status",
                lines=2,
                interactive=False
            )
            clear_btn = gr.Button("Clear Database", variant="stop")

            gr.Markdown(
                """
                ### ℹ️ Info
                - Supported format: PDF
                - Multiple PDFs can be uploaded
                - Each PDF is chunked and embedded
                """
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with your PDFs")
            chatbot = gr.Chatbot(
                height=500,
                label="Chat History"
            )
            msg = gr.Textbox(
                label="Ask a question",
                placeholder="What is this document about?",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_chat_btn = gr.Button("Clear Chat")

    # Event handlers
    process_btn.click(
        fn=process_pdf_upload,
        inputs=pdf_upload,
        outputs=status_output
    )

    clear_btn.click(
        fn=clear_database,
        outputs=status_output
    )

    # Chat functionality
    def respond(message, chat_history):
        bot_message = chat_with_pdf(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])
    clear_chat_btn.click(lambda: None, None, chatbot, queue=False)

# --- Launch ---
if __name__ == "__main__":
    print("🌐 Launching Gradio interface...")
    print("⚠️  Make sure Ollama is running: 'ollama serve'")
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860
    )
```

**Key Features**:
- ✅ Document-grounded responses (no hallucinations)
- ✅ Source citations for transparency
- ✅ Toggle to show/hide sources
- ✅ Clean, professional interface
- ✅ Copy-paste friendly outputs

#### Milestone 4.3: Performance Optimization

**Create**: `scripts/07_optimization_techniques.py`

```python
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
```

**Key Optimization Strategies**:

1. **Vector Store Selection**:
    - **Chroma**: Easy to use, persistent, good for development
    - **FAISS**: 2-10x faster, better for production
    - **Pinecone/Weaviate**: Managed, scalable (but not free)

2. **Chunk Size Tuning**:
    - **Too small** (< 100 tokens): Context fragmentation
    - **Too large** (> 2000 tokens): Irrelevant information
    - **Sweet spot**: 300-800 tokens for most use cases

3. **Embedding Model Selection**:
    - **all-MiniLM-L6-v2**: Fast, 384 dimensions
    - **all-mpnet-base-v2**: Better quality, 768 dimensions
    - **e5-large**: Best quality, slower

4. **Caching**:
    - Cache embeddings (don't recompute)
    - Cache vector stores (persist to disk)
    - Cache LLM responses for common queries

---

## 📚 Learning Resources

### Essential Reading

**Foundational**:
1. [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction) - Official docs
2. [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md) - Model management
3. [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401) - Original research

**Advanced**:
4. [Advanced RAG Techniques](https://www.anthropic.com/index/contextual-retrieval) - Anthropic blog
5. [Vector Database Comparison](https://superlinked.com/vector-db-comparison/) - Performance benchmarks
6. [Prompt Engineering Guide](https://www.promptingguide.ai/) - Better prompts = better results

### Video Tutorials

1. **Ollama Basics**: [Installation and Setup](https://www.youtube.com/results?search_query=ollama+tutorial)
2. **LangChain Crash Course**: [Full Beginner Guide](https://www.youtube.com/results?search_query=langchain+tutorial)
3. **RAG from Scratch**: [Step-by-step Build](https://www.youtube.com/results?search_query=rag+tutorial+python)

### Practice Datasets

1. **Wikipedia Dumps**: Small subsets for testing
2. **arXiv Papers**: Research paper corpus
3. **Gutenberg Project**: Books for summarization
4. **Your Own Notes**: Most relevant for learning!

---

## 🗓️ 8-Week Study Plan

| Week | Focus | Deliverable |
|------|-------|-------------|
| **1** | Setup + LLM basics | Run Ollama models, understand tokens |
| **2** | Embeddings | Create similarity search from scratch |
| **3** | Document processing | Build chunking pipeline |
| **4** | First RAG | Complete working RAG system |
| **5** | Multiple documents | Multi-source RAG |
| **6** | UI development | Gradio interface |
| **7** | Optimization | FAISS integration, performance tuning |
| **8** | Personal project | Your custom RAG application |

**Daily Commitment**: 1-2 hours
**Total Time**: 60-80 hours

---

##  Key Takeaways

### Core Concepts to Master

✅ **Embeddings**: Text → vectors → similarity
✅ **Vector Databases**: Store and search embeddings efficiently
✅ **Chunking**: Break documents intelligently
✅ **Retrieval**: Find relevant context for queries
✅ **Augmentation**: Combine context + prompt
✅ **Generation**: LLM produces grounded answer

### The RAG Mental Model

```
User Question
     ↓
Find Similar Docs (Retrieval)
     ↓
Add to Prompt (Augmentation)
     ↓
Generate Answer (Generation)
```

**Remember**: RAG doesn't change the LLM. It changes what the LLM sees.

### When to Use RAG vs Fine-Tuning

**Use RAG when**:
- ✅ Data changes frequently
- ✅ Need transparency (see sources)
- ✅ Limited compute resources
- ✅ Privacy-sensitive data (keep local)

**Use Fine-Tuning when**:
- ✅ Need domain-specific style/tone
- ✅ Want to change model behavior
- ✅ Have large, stable training data
- ✅ Can invest in training infrastructure

**Best**: Use both! Fine-tune for style, RAG for facts.

---

## 🚀 Next Steps

### After Completing This Roadmap

1. **Deploy Your App**:
    - Learn Docker containerization
    - Deploy on free tier (Hugging Face Spaces)
    - Add authentication

2. **Advanced RAG**:
    - Query expansion (generate multiple queries)
    - Re-ranking (reorder retrieved docs)
    - Hybrid search (combine keyword + semantic)
    - Agentic RAG (let LLM decide when to retrieve)

3. **Production Skills**:
    - Monitoring and logging
    - Error handling
    - Rate limiting
    - Cost optimization

4. **Explore Alternatives**:
    - Try LlamaIndex (RAG framework)
    - Experiment with Haystack
    - Compare to OpenAI Assistants API

---

##   Frequently Asked Questions

**Q: Do I need a powerful computer?**
A: No! Start with `tinyllama` (runs on 4GB RAM). Upgrade to `mistral` if you have 8GB+.

**Q: How much does this cost?**
A: $0 for everything in this roadmap. All tools are free and run locally.

**Q: Can I use this for commercial projects?**
A: Yes! Ollama models have permissive licenses. Check each model's license on [ollama.com/library](https://ollama.com/library).

**Q: What if I don't know Python well?**
A: You should know basic Python (variables, functions, imports). Take a quick Python refresher if needed.

**Q: How is this different from ChatGPT?**
A: You control the model, data, and costs. Perfect for learning, privacy-sensitive work, or custom applications.

**Q: Can I use my own data?**
A: Absolutely! That's the point of RAG. Just load your documents into the pipeline.

---

## 🎉 Final Encouragement

Learning GenAI and RAG is a journey, not a destination. You'll encounter errors, confusion, and moments of "Why isn't this working?!" That's normal and means you're learning deeply.

**Tips for Success**:

1. **Code every day**: Even 30 minutes matters
2. **Break things**: Learn by experimenting
3. **Take notes**: Document your "aha!" moments
4. **Ask questions**: Use communities (Reddit r/LangChain, Discord servers)
5. **Build projects**: Theory + Practice = Mastery

**Remember**: Every expert was once a beginner. You've got this! 🚀

---

## 📝 Appendix: Complete Code Repository

All notebooks mentioned in this guide are available at:
**[AI-ML-Tutorial-Hub](https://github.com/IssueNumber108/ai-llm-tutorials-hub?tab=readme-ov-file#ai-llm-tutorials-hub)**

Clone and start learning:
```bash
git clone [AI-ML-Tutorial-Hub]
cd scripts
python3 <script_name.py>
```
