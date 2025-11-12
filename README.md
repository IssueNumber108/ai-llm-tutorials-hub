# 🎓 AI & LLM Tutorials Hub

A comprehensive learning resource for **Generative AI** and **Retrieval-Augmented Generation (RAG)** - designed for beginners who want to build AI applications from scratch.

## 📚 What You'll Learn

This repository provides a complete, hands-on learning path covering:

- **Generative AI fundamentals** - How large language models work
- **RAG (Retrieval-Augmented Generation)** - Building document-powered AI systems
- **Local development** - Run everything on your machine (no API costs!)
- **Production-ready code** - From basic concepts to deployable applications

## 🎯 Perfect For

- Developers new to GenAI/LLM development
- Anyone wanting to build RAG applications locally
- Those looking for **zero-cost**, privacy-first AI learning
- People who learn best by coding and experimenting

## 📖 Getting Started

### Complete Beginner's Guide

Our comprehensive tutorial covers everything from setup to deployment:

**📄 [Beginner's Guide](documents/BeginnersGuide.md)**

This guide includes:
- Conceptual foundations (embeddings, LLMs, RAG architecture)
- Step-by-step environment setup (Python, Jupyter, Ollama)
- Progressive learning path with 8 milestones
- Production-ready code examples
- Performance optimization techniques

### 💻 Python Scripts

Ready-to-run implementations are available in the `scripts/` directory:

```
/scripts
├── 01_embeddings_basics.py          # Understanding embeddings
├── 02_document_processing.py        # Document loading & chunking
├── 03_rag_ollama_basic.py          # First RAG pipeline
├── 04_retrieval_strategies.py       # Advanced retrieval
├── 05_multi_document_rag.py        # Multi-source RAG
├── 05b_rag_ollama_pdf.py           # Working with PDFs
├── 06_ollama_chatbot_local.py      # Simple chatbot UI
├── 06b_rag_chatbot_ui.py           # RAG-powered chat interface
└── 07_optimization_techniques.py    # Performance tuning
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- 8GB+ RAM (for Mistral model)
- Basic Python knowledge

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-llm-tutorials-hub.git
cd ai-llm-tutorials-hub

# Create virtual environment
python -m venv genai_env
source genai_env/bin/activate  # On Windows: genai_env\Scripts\activate

# Install dependencies
pip install langchain langchain-community langchain-ollama
pip install langchain-chroma langchain-huggingface
pip install chromadb sentence-transformers pypdf gradio

# Install Ollama (see https://ollama.com for platform-specific instructions)
# Then download a model
ollama pull mistral
```

### Run Your First Example

```bash
cd scripts
python 03_rag_ollama_basic.py
```

## 📋 Learning Path

Follow this 8-week progression:

| Week | Focus | Key Concepts |
|------|-------|--------------|
| 1-2 | **Foundation** | LLMs, embeddings, Ollama basics |
| 3-4 | **Basic RAG** | Document processing, vector databases, first RAG system |
| 5-6 | **Advanced RAG** | Multi-document retrieval, PDF processing, UI development |
| 7-8 | **Production** | Optimization, deployment, custom projects |

Steps are mentioned here: [Beginner's Guide](documents/BeginnersGuide.md)

## 🛠️ Tech Stack

- **LLMs**: Ollama (local inference)
- **Framework**: LangChain
- **Vector Databases**: ChromaDB, FAISS
- **Embeddings**: Sentence Transformers
- **UI**: Gradio
- **Documents**: PyPDF, Unstructured

## ✨ Key Features

✅ **100% Local** - No API keys, no cloud dependencies  
✅ **Zero Cost** - Everything runs on your machine  
✅ **Privacy First** - Your data never leaves your computer  
✅ **Production Ready** - Real-world, deployable code  
✅ **Well Documented** - Clear explanations with examples  
✅ **Hands-On** - Learn by building actual applications

## 📊 What You'll Build

By completing this tutorial, you'll create:

1. **Document Q&A System** - Ask questions about your PDFs
2. **Local Chatbot** - Conversational AI with custom knowledge
3. **Multi-Source RAG** - Query across multiple documents
4. **Web Interface** - User-friendly chat application

## 🎯 Next Steps

**Coming Soon:**

- 🔄 Advanced RAG patterns (query expansion, re-ranking)
- 🤖 Agentic RAG with tool use
- 📊 RAG evaluation and metrics
- 🚀 Deployment guides (Docker, cloud platforms)
- 🔍 Hybrid search techniques
- 🎨 Custom UI templates
- 📈 Monitoring and observability
- 🔐 Authentication and security best practices

Stay tuned for updates!

## 📚 Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [ChromaDB Guides](https://docs.trychroma.com/)

### Learning Materials
- [Complete Beginner's Guide](documents/BeginnersGuide.md) - Start here!
- [RAG Paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Anthropic's Contextual Retrieval](https://www.anthropic.com/index/contextual-retrieval)

## 🤝 Contributing

Contributions are welcome! Whether it's:
- 🐛 Bug fixes
- 📝 Documentation improvements
- ✨ New tutorials or examples
- 💡 Feature suggestions

Please open an issue or submit a pull request.

## ❓ FAQ

**Q: Do I need a powerful GPU?**  
A: No! Everything runs on CPU. Start with smaller models (Mistral) and scale up as needed.

**Q: How much does this cost?**  
A: $0 - All tools and models are free and open-source.

**Q: Can I use this for commercial projects?**  
A: Yes! Check individual model licenses on [Ollama's library](https://ollama.com/library).

**Q: What if I'm new to Python?**  
A: You need basic Python knowledge (variables, functions, imports). Consider a Python refresher course first.

## 📧 Support
- 📖 Check the [Beginner's Guide](documents/BeginnersGuide.md) for detailed explanations
- 💬 Open an issue for questions or bugs
- 🌟 Star this repo if you find it helpful!

---

**Last Updated**: November 2025  
**Maintained by**: Atul Pandey

**Start your GenAI journey today!** Head to the [Beginner's Guide](documents/BeginnersGuide.md) to begin.