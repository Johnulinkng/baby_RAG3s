# BabyCare RAG Tool 🍼

A powerful, standalone RAG (Retrieval-Augmented Generation) system specifically designed for baby care knowledge management and question answering. This tool can be easily integrated into team projects to provide intelligent baby care assistance.

## 🌟 Features

- **Hybrid Search**: Combines BM25 and vector search with Reciprocal Rank Fusion
- **Multi-format Document Support**: PDF, DOCX, TXT, HTML, and more
- **Intelligent Question Answering**: Multi-step reasoning with source attribution
- **Easy Integration**: Simple Python API for team projects
- **Command-line Tools**: Ready-to-use CLI for testing and demonstration
- **Configurable**: Flexible configuration for different use cases
- **Memory Management**: Persistent knowledge base with FAISS indexing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd baby-care-agent

# Install dependencies
pip install -e .

# Set up environment variables
cp env-template .env
# Edit .env with your API keys
```

### Basic Usage

```python
from babycare_rag import BabyCareRAG

# Initialize the RAG system
rag = BabyCareRAG()

# Ask a question
response = rag.query("What temperature should a baby's room be?")
print(response.answer)

# Add a document
rag.add_document("path/to/baby_care_guide.pdf")

# Search documents
results = rag.search_documents("baby feeding schedule")
```

### Using the API Wrapper

```python
from babycare_rag.api import BabyCareRAGAPI

# Initialize API
api = BabyCareRAGAPI()

# Query with error handling
result = api.query("How often should I feed my newborn?")
if result["success"]:
    print(result["data"]["answer"])
else:
    print(f"Error: {result['error']}")
```

### Quick Functions

```python
from babycare_rag.api import quick_query, quick_add_document

# Simple query
answer = quick_query("What are signs of teething?")
print(answer)

# Add document quickly
success = quick_add_document("baby_guide.pdf")
```

## 🛠️ Command-line Tools

### Interactive CLI

```bash
# Run interactive CLI
python test_tools/cli_test.py

# Or use the installed script
babycare-rag-cli
```

### API Testing

```bash
# Test API functionality
python test_tools/api_test.py --all

# Interactive API test
python test_tools/api_test.py --interactive
```

### Integration Examples

```bash
# See integration examples
python test_tools/integration_example.py --all
```

## 📁 Project Structure

```
babycare_rag/           # Core RAG package
├── __init__.py         # Package initialization
├── config.py           # Configuration management
├── models.py           # Data models
├── core.py             # Main RAG class
├── api.py              # API wrapper
├── document_processor.py  # Document handling
└── search_engine.py    # Hybrid search engine

test_tools/             # Testing and demonstration tools
├── cli_test.py         # Command-line interface
├── api_test.py         # API testing script
└── integration_example.py  # Integration examples

documents/              # Document storage
faiss_index/           # Vector index storage
```

## ⚙️ Configuration

The system uses environment variables for configuration:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional (with defaults)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```

### Custom Configuration

```python
from babycare_rag import RAGConfig, BabyCareRAG

config = RAGConfig(
    max_steps=3,
    top_k=5,
    chunk_size=1000,
    search_top_k=20
)

rag = BabyCareRAG(config)
```

## 🔧 Integration Guide

### For Team Projects

1. **Install as a dependency**:
   ```bash
   pip install -e path/to/babycare-rag
   ```

2. **Basic integration**:
   ```python
   from babycare_rag.api import BabyCareRAGAPI
   
   class YourApp:
       def __init__(self):
           self.rag_api = BabyCareRAGAPI()
       
       def get_baby_advice(self, question):
           result = self.rag_api.query(question)
           return result["data"]["answer"] if result["success"] else None
   ```

3. **Add your own documents**:
   ```python
   # Add team-specific baby care documents
   api.add_document(file_path="your_baby_guide.pdf")
   
   # Add from URL
   api.add_document(url="https://example.com/baby-care-article")
   
   # Add text content
   api.add_document(
       text_content="Your custom baby care knowledge...",
       title="Team Baby Care Guidelines"
   )
   ```

### API Reference

#### Main Classes

- `BabyCareRAG`: Core RAG system
- `BabyCareRAGAPI`: API wrapper with error handling
- `RAGConfig`: Configuration management

#### Key Methods

- `query(question)`: Ask a question
- `add_document(file_path)`: Add document from file
- `add_document_from_url(url)`: Add document from URL
- `add_document_from_text(text, title)`: Add text content
- `search_documents(query)`: Search knowledge base
- `list_documents()`: List all documents
- `get_stats()`: Get system statistics

## 🧪 Testing

```bash
# Run basic tests
python test_tools/api_test.py --basic

# Test document management
python test_tools/api_test.py --docs

# Interactive testing
python test_tools/cli_test.py
```

## 📊 System Requirements

- Python 3.10+
- Google Gemini API key
- Ollama server (for embeddings)
- 2GB+ RAM (depending on document size)
- 1GB+ disk space (for index storage)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the test tools and examples
2. Review the configuration settings
3. Open an issue on GitHub

## 🔄 Updates

To update the system:
1. Pull latest changes
2. Rebuild the index: `rag.rebuild_index()`
3. Test with your documents
