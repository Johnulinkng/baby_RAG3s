# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Build and Development Commands

### Initial Setup
```bash
# Install dependencies (from project root)
pip install -e .

# Setup environment
cp env-template .env
# Edit .env to add OPENAI_API_KEY or configure AWS Secrets Manager

# Verify setup and build initial indexes
python setup_rag.py
```

### Running the System

**Main Agent System (Conversational AI with multi-step reasoning):**
```bash
# Interactive agent mode
python agent.py

# Test agent programmatically
python -c "import asyncio; from agent import main; print(asyncio.run(main('What is ideal baby room temperature?')))"
```

**RAG API (Direct search and query):**
```bash
# Interactive CLI
python test_tools/cli_test.py

# API test interface
python test_tools/api_test.py --interactive

# Quick test
python test_tools/api_test.py --all
```

### Testing

```bash
# Run all API tests
python test_tools/api_test.py --all

# Test integration example
python examples/simple_integration/my_baby_app.py --demo all

# Test search functionality
python test_tools/cli_test.py -q "baby room temperature"

# Health check
python -c "from babycare_rag.api import BabyCareRAGAPI; api = BabyCareRAGAPI(); print(api.health_check())"
```

### Document Management

```bash
# Add document via CLI
python test_tools/cli_test.py --add-doc "path/to/document.pdf"

# List documents
python -c "from babycare_rag.api import BabyCareRAGAPI; api = BabyCareRAGAPI(); print(api.list_documents())"
```

## Architecture Overview

This codebase implements a hybrid RAG (Retrieval-Augmented Generation) system specialized for baby care information. It consists of two main subsystems that share the same document index but serve different purposes:

### 1. Agent System (agent.py + MCP Tools)
The agent implements a multi-step reasoning loop with memory management:

```
User Input → Perception (intent/entities) → Memory Retrieval → Decision (LLM)
    ↓                                                              ↓
Tool Execution ← Tool Selection                         Final Answer
    ↓
Update Memory → Loop (max 2 steps)
```

**Key Components:**
- `agent.py`: Main orchestrator with reasoning loop
- `perception.py`: Intent recognition and entity extraction
- `decision.py`: LLM-based plan generation with tool selection
- `memory.py`: Session-based memory management
- `action.py`: Tool execution interface
- `math_mcp_embeddings.py`: MCP server providing search_documents tool

**Evidence-Based Answering:** When search_documents returns results, the system enforces evidence-grounded responses using a specialized prompt that only synthesizes from retrieved documents.

### 2. RAG Module (babycare_rag/)
Direct search and retrieval system with hybrid search algorithm:

```
Query → Synonym Expansion → Parallel Search
                            ├─ BM25 (keyword)
                            └─ FAISS (semantic)
                                    ↓
                            RRF Fusion → Ranked Results
```

**Core Classes:**
- `BabyCareRAG`: Main interface combining all components
- `SearchEngine`: Implements BM25 + FAISS + RRF fusion
- `DocumentProcessor`: Handles document ingestion and chunking
- `BabyCareRAGAPI`: Error-handled API wrapper

**Search Algorithm:** Uses Reciprocal Rank Fusion (RRF) with k=60 to combine BM25 keyword matching and FAISS semantic search results.

### Document Processing Pipeline

```
Document → MarkItDown → Text Chunking → Embedding → FAISS Index
            ↓              ↓                ↓
       Text Extraction  256 words/chunk  OpenAI API
```

All documents are stored in:
- `documents/`: Original files
- `faiss_index/`: Vector embeddings and metadata
- JSON metadata files for chunk mappings

### Configuration System

The system uses a hierarchical configuration approach:
1. Environment variables (.env file)
2. AWS Secrets Manager (for production)
3. Default values in RAGConfig class

Priority: AWS Secrets > Environment Variables > Defaults

### Key Design Decisions

1. **Dual Search Architecture**: MCP tools for agent, direct SearchEngine for API/CLI
2. **Evidence Enforcement**: Agent must ground answers in search results for factual queries
3. **Session Memory**: Agent maintains conversation context across tool calls
4. **Hybrid Search**: Combines keyword (BM25) and semantic (embeddings) search
5. **Chunking Strategy**: 256-word chunks with 40-word overlap for context preservation

## Environment Variables

Required (one of these options):
- `OPENAI_API_KEY`: Direct OpenAI API key
- `SECRET_ID` + `AWS_REGION`: AWS Secrets Manager configuration

Optional:
- `OPENAI_LLM_MODEL`: Default "gpt-4o-mini"
- `OPENAI_EMBED_MODEL`: Default "text-embedding-3-small"
- `RAG_MAX_STEPS`: Agent reasoning steps (default 2)
- `RAG_TOP_K`: Number of search results (default 5)

## Performance Considerations

- Initial query includes embedding generation overhead (~1-2s)
- Subsequent queries benefit from cached embeddings
- FAISS index loads into memory on first use
- Agent system adds ~1s overhead for reasoning loop
- Document processing: ~1000 docs/minute depending on size

## Common Development Tasks

### Debugging Search Issues
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test search directly
from babycare_rag.search_engine import SearchEngine
engine = SearchEngine()
results = engine.search("test query", top_k=5)
for r in results: print(f"Score: {r.score}, Source: {r.source}")
```

### Adding Custom Tools to Agent
Tools are defined in `math_mcp_embeddings.py`. Add new tools by:
1. Defining the tool function
2. Registering with `@server.tool()` decorator
3. Update tool descriptions in the server

### Modifying Search Weights
Adjust BM25 vs vector search balance in `RAGConfig`:
- `bm25_weight`: Weight for keyword matching (default 0.3)
- `vector_weight`: Weight for semantic search (default 0.7)

## Deployment

### Docker Deployment
```bash
cd deploy
docker-compose up -d
```

### AWS EC2 Setup
```bash
# Install Python 3.10+
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip

# Clone and setup
git clone <repository>
cd babycare_RAG_CMD
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure AWS credentials for Secrets Manager
export AWS_REGION=us-east-2
export SECRET_ID=Opean_AI_KEY_IOSAPP

# Run setup
python setup_rag.py
```

## Troubleshooting

### "No response generated" from Agent
- Check document index exists: `ls faiss_index/`
- Verify search is working: `python test_tools/cli_test.py -q "test"`
- Increase max_steps if needed

### OpenAI API Errors
- Verify API key: `echo $OPENAI_API_KEY`
- Check AWS Secrets access if using: `aws secretsmanager get-secret-value --secret-id $SECRET_ID`
- Monitor rate limits in OpenAI dashboard

### Memory Issues
- Default FAISS index loads fully into memory
- For large document sets (>10k docs), consider using FAISS with disk-based index
- Monitor with: `python -c "from babycare_rag.api import BabyCareRAGAPI; api = BabyCareRAGAPI(); print(api.get_stats())"`
