# ===============================
# RAG-Powered Chatbot with Gradio
# Combines conversational AI with document retrieval
# ===============================

# --- STEP 0: Install dependencies (run once) ---
# Uncomment if running for the first time:
# !pip install gradio langchain langchain-ollama langchain-chroma langchain-community langchain-core sentence-transformers pypdf langchain-huggingface


print("Building PDF Chat Interface with Gradio\n")

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