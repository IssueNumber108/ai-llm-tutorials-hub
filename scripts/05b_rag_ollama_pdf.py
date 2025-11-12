# ====================================================
# RAG with PDF Documents - Auto Download Sample PDFs
# ====================================================

print("Building a PDF RAG System with Sample PDFs\n")

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
    print(f"❓ Query {i}: {query}\n")

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