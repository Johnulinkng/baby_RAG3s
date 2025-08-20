# NanAI - Your Baby's AI Nanny 👶

A RAG-powered AI assistant that helps parents with baby care advice, combining document knowledge with real-time web information. Built with Streamlit, Google's Gemini Flash model and Ollama nomic-embed-text open embedding model, and FAISS for efficient semantic search.

NanAI is an intelligent assistant designed to help parents with various aspects of baby care. It provides expert guidance on baby care, safety, development, product information, and parenting advice through an easy-to-use web interface.

## 💭 Inspiration

As a new parent to a beautiful baby boy, I found myself constantly searching for answers to various questions about baby care, safety, and development. The journey of parenthood is filled with countless moments of uncertainty and the need for quick, reliable information. 

This project was born from my personal experience of:

- Wanting instant access to trusted information about baby care
- Needing to quickly reference product manuals and safety guidelines
- Seeking answers to common parenting questions
- Managing and organizing various baby care resources in one place

NanAI is my attempt to create a helpful tool that combines my birthing class materials, product documentation, and web resources into an easily accessible knowledge base. I hope it can help other parents navigate the wonderful, yet sometimes overwhelming, journey of raising a child.

PS, the pdfs are from a birthing class in Canada and some of the baby products I have purchased for my newborn son 👶


![NanAI Interface](images/user-interface.png)

## ✨ Features

- 🤖 AI-powered responses to baby care questions
- 📚 Knowledge base that can be expanded with custom documents
- 🌐 Support for adding information from web URLs
- 💬 Interactive chat interface
- 📱 User-friendly web application built with Streamlit

## 🛠️ Technology Stack

### Key Technologies

- **[Streamlit](https://streamlit.io/)**: Provides the web interface and real-time interaction capabilities
- **[Google Generative AI](https://ai.google.dev/)**: Powers the core language model for understanding and generating responses
- **[LlamaIndex](https://www.llamaindex.ai/)**: Handles document processing, chunking, and indexing
- **[FAISS](https://github.com/facebookresearch/faiss)**: Facebook AI Similarity Search for efficient vector similarity search and clustering
- **Async Processing**: Handles concurrent requests and document processing

This demo showcases how these technologies can be combined to create an intelligent assistant that:

1. Maintains a growing knowledge base through:
   - Document uploads (PDF, TXT, DOCX)
   - Web content ingestion via URLs
2. Provides contextually relevant responses using RAG (Retrieval-Augmented Generation)
3. Learns from new information sources in real-time
4. Delivers interactive responses through a user-friendly interface

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

### Installation

#### Option 1: Clone and Install

1. Clone the repository:

```bash
git clone <repository-url>
cd baby-care-agent
```

2. Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies using uv:

```bash
uv pip install -e .
```

#### Option 2: Quick Start with uv

1. Create a new directory and initialize the project:

```bash
mkdir baby-care-agent
cd baby-care-agent
```

2. Initialize with uv:

```bash
uv venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Clone and set up the project:

```bash
git clone https://github.com/rohinigaonkar/baby-care-agent.git .
uv pip install -e .
```

4. Set up environment variables:

```bash
cp env-template .env
# Windows PowerShell
Copy-Item env-template .env
```

Edit the `.env` file with your required API keys and configurations.

### Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` by default.

### Configuring Embedding Service (Ollama)

By default, embeddings are requested from a local Ollama server:
- Base URL: http://localhost:11434
- Model: nomic-embed-text

You can override via environment variables in `.env`:

```bash
# optional overrides
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```

On Windows, if the `ollama` command is not on PATH, use the full path:

```powershell
& "C:\\Program Files\\Ollama\\ollama.exe" serve
& "C:\\Program Files\\Ollama\\ollama.exe" pull nomic-embed-text
```

On Linux/EC2 (CPU-only is fine for this project):
- Install Ollama for Linux
- Start the service: `ollama serve`
- Pull model: `ollama pull nomic-embed-text`
- Ensure the app host can reach `${OLLAMA_BASE_URL}/api/embeddings`

## 💡 Usage

1. **Ask Questions**: Use the chat interface to ask any baby care related questions
2. **Add Knowledge**: Upload documents or provide URLs to expand the assistant's knowledge base
3. **Example Questions**: Try the pre-loaded example questions to get started

## 📁 Project Structure

- `app.py` - Main Streamlit application
- `agent.py` - Core agent implementation
- `perception.py` - Handles input processing
- `memory.py` - Manages knowledge storage and retrieval
- `decision.py` - Decision-making logic
- `action.py` - Action execution
- `models.py` - Data models
- `documents/` - Directory for storing knowledge base documents
- `faiss_index/` - Vector store for document embeddings

## 📦 Dependencies

- streamlit - Web application framework
- faiss-cpu - Vector similarity search
- google-genai - AI model integration
- llama-index - Document processing and indexing
- Other dependencies listed in `pyproject.toml`


