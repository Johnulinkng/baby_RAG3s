# BabyCare RAG Integration Guide 🔧

This guide provides detailed instructions for integrating the BabyCare RAG system into your team projects.

## 📋 Prerequisites

1. **Python 3.10+**
2. **OpenAI API Key** 或配置 **AWS Secrets Manager**（SECRET_ID/AWS_REGION）

## 🚀 Quick Setup

### 1. Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd baby-care-agent

# Install the package
pip install -e .

# Run setup script
python setup_rag.py
```

### 2. Configure Environment

```bash
# Copy environment template
cp env-template .env

# Edit .env file
# Option A: OpenAI
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
# Option B: AWS Secrets
SECRET_ID=Opean_AI_KEY_IOSAPP
AWS_REGION=us-east-2
```

### 3. Initialize Index & Health Check

```bash
python setup_rag.py
python -c "from babycare_rag.api import BabyCareRAGAPI; api = BabyCareRAGAPI(); print(api.health_check())"
```

### 4. Test the Setup

```bash
# Test the system
python test_tools/cli_test.py --help

# Run interactive test
python test_tools/api_test.py --interactive
```

## 🏗️ Integration Patterns

### Pattern 1: Direct Integration

```python
# your_project/baby_care_service.py
from babycare_rag import BabyCareRAG, RAGConfig

class BabyCareService:
    def __init__(self):
        # Initialize with custom config
        config = RAGConfig(
            max_steps=3,
            top_k=5,
            chunk_size=800
        )
        self.rag = BabyCareRAG(config)
    
    def get_advice(self, question: str) -> dict:
        """Get baby care advice."""
        response = self.rag.query(question)
        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }
    
    def add_team_knowledge(self, content: str, title: str):
        """Add team-specific knowledge."""
        return self.rag.add_document_from_text(content, title)
```

### Pattern 2: API Wrapper Integration

```python
# your_project/baby_care_api.py
from babycare_rag.api import BabyCareRAGAPI

class BabyCareAPI:
    def __init__(self):
        self.api = BabyCareRAGAPI()
    
    def ask_question(self, question: str) -> dict:
        """Ask a question with error handling."""
        result = self.api.query(question)
        
        if result["success"]:
            return {
                "status": "success",
                "answer": result["data"]["answer"],
                "sources": result["data"]["sources"],
                "confidence": result["data"]["confidence"]
            }
        else:
            return {
                "status": "error",
                "message": result["error"]
            }
    
    def health_check(self) -> bool:
        """Check if the system is healthy."""
        health = self.api.health_check()
        return health["success"] and health["data"]["status"] == "healthy"
```

### Pattern 3: Microservice Integration

```python
# your_project/baby_care_microservice.py
from flask import Flask, request, jsonify
from babycare_rag.api import BabyCareRAGAPI

app = Flask(__name__)
rag_api = BabyCareRAGAPI()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    health = rag_api.health_check()
    return jsonify(health)

@app.route('/query', methods=['POST'])
def query():
    """Query endpoint."""
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    result = rag_api.query(question)
    return jsonify(result)

@app.route('/documents', methods=['GET'])
def list_documents():
    """List documents endpoint."""
    result = rag_api.list_documents()
    return jsonify(result)

@app.route('/documents', methods=['POST'])
def add_document():
    """Add document endpoint."""
    data = request.get_json()
    result = rag_api.add_document(**data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## 🔧 Configuration Options

### Basic Configuration

```python
from babycare_rag import RAGConfig

config = RAGConfig(
    # LLM Settings
    llm_model="gpt-4o-mini",

    # Embedding Settings
    embed_model="text-embedding-3-small",

    # RAG Parameters
    max_steps=5,
    top_k=3,
    chunk_size=1000,
    chunk_overlap=200,

    # Search Settings
    search_top_k=20,
    bm25_weight=0.3,
    vector_weight=0.7
)
```

### Environment-based Configuration

```python
# Use environment variables (recommended)
config = RAGConfig.from_env()

# Override specific settings
config.max_steps = 3
config.top_k = 5
```

## 📚 Document Management

### Adding Documents

```python
# From file
rag.add_document("path/to/baby_guide.pdf")

# From URL
rag.add_document_from_url("https://example.com/baby-care-tips")

# From text
rag.add_document_from_text(
    text="Important baby care information...",
    title="Team Guidelines"
)

# Using API wrapper
api.add_document(file_path="guide.pdf")
api.add_document(url="https://example.com/article")
api.add_document(text_content="...", title="Guidelines")
```

### Managing Documents

```python
# List all documents
documents = rag.list_documents()
for doc in documents:
    print(f"{doc.title}: {doc.chunk_count} chunks")

# Remove a document
rag.remove_document(doc_id="document_id")

# Rebuild index after changes
rag.rebuild_index()
```

## 🔍 Querying and Search

### Basic Queries

```python
# Simple query
response = rag.query("What temperature should baby's room be?")
print(response.answer)

# With custom parameters
response = rag.query("How to soothe crying baby?", max_steps=3)
```

### Advanced Search

```python
# Search documents directly
results = rag.search_documents("baby feeding schedule", top_k=5)
for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.source}")
    print(f"Text: {result.text[:100]}...")
```

### Batch Processing

```python
questions = [
    "When to start solid foods?",
    "How to burp a baby?",
    "Normal sleep patterns for newborns?"
]

answers = []
for question in questions:
    response = rag.query(question)
    answers.append({
        "question": question,
        "answer": response.answer,
        "confidence": response.confidence
    })
```

## 🚨 Error Handling

### Robust Error Handling

```python
from babycare_rag.api import BabyCareRAGAPI

api = BabyCareRAGAPI()

def safe_query(question: str) -> dict:
    """Query with comprehensive error handling."""
    try:
        result = api.query(question)
        
        if result["success"]:
            return {
                "status": "success",
                "data": result["data"]
            }
        else:
            return {
                "status": "error",
                "error": result["error"],
                "fallback": "I'm sorry, I couldn't process your question right now."
            }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "fallback": "An unexpected error occurred."
        }
```

### Health Monitoring

```python
def monitor_rag_health():
    """Monitor RAG system health."""
    health = api.health_check()
    
    if not health["success"]:
        print("❌ RAG system is unhealthy")
        return False
    
    data = health["data"]
    if data["status"] != "healthy":
        print(f"⚠️  RAG system status: {data['status']}")
        if "error" in data:
            print(f"Error: {data['error']}")
        return False
    
    print("✅ RAG system is healthy")
    print(f"Documents: {data['total_documents']}")
    print(f"Chunks: {data['total_chunks']}")
    return True
```

## 🧪 Testing Your Integration

### Unit Tests

```python
import unittest
from your_project.baby_care_service import BabyCareService

class TestBabyCareService(unittest.TestCase):
    def setUp(self):
        self.service = BabyCareService()
    
    def test_get_advice(self):
        """Test getting advice."""
        result = self.service.get_advice("What is the ideal room temperature?")
        
        self.assertIsInstance(result, dict)
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("confidence", result)
    
    def test_add_knowledge(self):
        """Test adding knowledge."""
        success = self.service.add_team_knowledge(
            "Test knowledge content",
            "Test Document"
        )
        self.assertTrue(success)
```

### Integration Tests

```python
def test_full_workflow():
    """Test complete workflow."""
    api = BabyCareRAGAPI()
    
    # Add document
    result = api.add_document(
        text_content="Babies should sleep on their backs.",
        title="Sleep Safety"
    )
    assert result["success"]
    
    # Query the new content
    result = api.query("How should babies sleep?")
    assert result["success"]
    assert "back" in result["data"]["answer"].lower()
    
    # Search for content
    result = api.search_documents("sleep safety")
    assert result["success"]
    assert len(result["data"]) > 0
```

## 📊 Performance Optimization

### Caching Responses

```python
from functools import lru_cache

class CachedBabyCareService:
    def __init__(self):
        self.rag = BabyCareRAG()
    
    @lru_cache(maxsize=100)
    def get_cached_advice(self, question: str) -> str:
        """Get cached advice for common questions."""
        response = self.rag.query(question)
        return response.answer
```

### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncBabyCareService:
    def __init__(self):
        self.rag = BabyCareRAG()
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_advice_async(self, question: str) -> dict:
        """Get advice asynchronously."""
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self.executor,
            self.rag.query,
            question
        )
        return {
            "answer": response.answer,
            "sources": response.sources,
            "confidence": response.confidence
        }
```

## 🔒 Security Considerations

### Input Validation

```python
import re

def validate_question(question: str) -> bool:
    """Validate user input."""
    if not question or len(question.strip()) < 3:
        return False
    
    if len(question) > 1000:  # Prevent very long inputs
        return False
    
    # Check for potentially harmful patterns
    harmful_patterns = [
        r'<script',
        r'javascript:',
        r'eval\(',
        r'exec\('
    ]
    
    for pattern in harmful_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            return False
    
    return True

def safe_query(question: str) -> dict:
    """Query with input validation."""
    if not validate_question(question):
        return {
            "status": "error",
            "error": "Invalid question format"
        }
    
    # Proceed with query...
```

### Rate Limiting

```python
from time import time
from collections import defaultdict

class RateLimitedService:
    def __init__(self):
        self.rag = BabyCareRAG()
        self.request_counts = defaultdict(list)
        self.rate_limit = 10  # requests per minute
    
    def is_rate_limited(self, client_id: str) -> bool:
        """Check if client is rate limited."""
        now = time()
        minute_ago = now - 60
        
        # Clean old requests
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > minute_ago
        ]
        
        return len(self.request_counts[client_id]) >= self.rate_limit
    
    def query_with_rate_limit(self, question: str, client_id: str) -> dict:
        """Query with rate limiting."""
        if self.is_rate_limited(client_id):
            return {
                "status": "error",
                "error": "Rate limit exceeded"
            }
        
        self.request_counts[client_id].append(time())
        response = self.rag.query(question)
        
        return {
            "status": "success",
            "answer": response.answer,
            "sources": response.sources
        }
```

## 📈 Monitoring and Logging

### Logging Setup

```python
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('babycare_rag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('babycare_rag')

class LoggedBabyCareService:
    def __init__(self):
        self.rag = BabyCareRAG()
    
    def get_advice(self, question: str, user_id: str = None) -> dict:
        """Get advice with logging."""
        start_time = datetime.now()
        
        logger.info(f"Query started - User: {user_id}, Question: {question[:50]}...")
        
        try:
            response = self.rag.query(question)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Query completed - Duration: {duration:.2f}s, "
                       f"Confidence: {response.confidence:.2f}, "
                       f"Sources: {len(response.sources)}")
            
            return {
                "answer": response.answer,
                "sources": response.sources,
                "confidence": response.confidence,
                "duration": duration
            }
        
        except Exception as e:
            logger.error(f"Query failed - Error: {str(e)}")
            raise
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install -e .

# Copy application code
COPY . .

# Create directories
RUN mkdir -p documents faiss_index

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "your_project/baby_care_microservice.py"]
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  babycare-rag:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_LLM_MODEL=${OPENAI_LLM_MODEL:-gpt-4o-mini}
      - OPENAI_EMBED_MODEL=${OPENAI_EMBED_MODEL:-text-embedding-3-small}
      - SECRET_ID=${SECRET_ID}
      - AWS_REGION=${AWS_REGION}
    volumes:
      - ./documents:/app/documents
      - ./faiss_index:/app/faiss_index
```

## 🆘 Troubleshooting

### Common Issues

1. **"OpenAI API key not found"**
   - Check env: `echo $OPENAI_API_KEY` or PowerShell: `$env:OPENAI_API_KEY`
   - Or check AWS Secrets env: `echo $SECRET_ID $AWS_REGION`

2. **"No documents found"**
   - Add docs into `documents/`
   - Run `python setup_rag.py` to build index

4. **"No documents found"**
   - Add documents to the `documents/` folder
   - Run `rag.rebuild_index()`

5. **"Low confidence scores"**
   - Add more relevant documents
   - Improve document quality
   - Adjust search parameters

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('babycare_rag').setLevel(logging.DEBUG)

# Test system health
from babycare_rag.api import BabyCareRAGAPI
api = BabyCareRAGAPI()
health = api.health_check()
print(health)
```

## 📞 Support

For additional support:
1. Check the test tools and examples
2. Review the configuration settings
3. Run the setup script: `python setup_rag.py`
4. Open an issue on GitHub
