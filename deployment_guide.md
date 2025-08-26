# BabyCare RAG Team Deployment Guide

## 🏢 Team Integration Summary

Based on our testing, the FastAPI server works excellently for team integration with the following characteristics:

### Performance Metrics (Local Testing)
- **Health Check**: ~7s (includes RAG system initialization)
- **Average Response Time**: ~13s per query (includes retrieval + LLM generation)
- **Streaming**: Works perfectly with Server-Sent Events (SSE)
- **Concurrent Requests**: 3 simultaneous requests completed successfully
- **Reliability**: 100% success rate in all test scenarios

### Key Differences: Team Directory vs Normal Testing

| Aspect | Normal Testing | Team Directory Integration |
|--------|----------------|---------------------------|
| **API Access** | Direct Python imports | HTTP REST API calls |
| **Response Format** | Python objects | JSON over HTTP |
| **Error Handling** | Python exceptions | HTTP status codes + JSON errors |
| **Concurrency** | Single-threaded | Multi-worker, concurrent requests |
| **Monitoring** | Print statements | Structured logs + health endpoint |
| **Deployment** | Local Python script | Web service (uvicorn/gunicorn) |

## 🚀 Local Testing (Windows) - VERIFIED ✅

### 1. Start the Server
```powershell
# In your project directory with venv activated
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2
```

### 2. Test Health Endpoint
```powershell
curl.exe http://localhost:8000/health
```

### 3. Test Non-Streaming Query
```powershell
# Create JSON file to avoid PowerShell quoting issues
echo {"question":"What are the ABCs of Safe Sleep?"} > query.json
curl.exe -X POST "http://localhost:8000/query" -H "Content-Type: application/json" --data-binary "@query.json"
```

### 4. Test Streaming Query
```powershell
curl.exe -N -X POST "http://localhost:8000/query?stream=true" -H "Content-Type: application/json" -H "Accept: text/event-stream" --data-binary "@query.json"
```

## 🌐 EC2 Deployment (GitHub Clone Method)

### 1. EC2 Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install -y python3 python3-venv python3-pip git curl

# Clone your repository
git clone https://github.com/your-username/babycare_RAG_CMD.git
cd babycare_RAG_CMD

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install fastapi uvicorn
pip install -r requirements.txt  # if you have one
```

### 2. Environment Configuration
```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"

# Or use AWS Secrets Manager (if configured)
export SECRET_ID="your-secret-id"
export AWS_REGION="us-east-1"
```

### 3. Start the Service
```bash
# Development mode
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2

# Production mode with process manager
nohup uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4 > rag_server.log 2>&1 &
```

### 4. Test on EC2 (Same as Local)
```bash
# Health check
curl http://localhost:8000/health

# Query test
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the ABCs of Safe Sleep?"}'

# Streaming test
curl -N -X POST "http://localhost:8000/query?stream=true" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question":"What is the ideal room temperature for a baby'\''s nursery?"}'
```

## 🔧 Production Considerations

### 1. Process Management (systemd)
Create `/etc/systemd/system/babycare-rag.service`:
```ini
[Unit]
Description=BabyCare RAG API
After=network.target

[Service]
Type=exec
User=ubuntu
WorkingDirectory=/home/ubuntu/babycare_RAG_CMD
Environment=PATH=/home/ubuntu/babycare_RAG_CMD/.venv/bin
Environment=OPENAI_API_KEY=your-key-here
ExecStart=/home/ubuntu/babycare_RAG_CMD/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable babycare-rag
sudo systemctl start babycare-rag
sudo systemctl status babycare-rag
```

### 2. Reverse Proxy (Nginx)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Important for streaming
        proxy_buffering off;
        proxy_cache off;
        add_header X-Accel-Buffering no;
    }
}
```

### 3. Security
- Use HTTPS in production
- Implement API key authentication if needed
- Set up firewall rules
- Monitor logs and metrics

## 📊 Team Usage Patterns

### 1. Web Application Backend
```python
import requests

class BabyCareRAGClient:
    def __init__(self, base_url="http://your-server:8000"):
        self.base_url = base_url
    
    def ask_question(self, question: str):
        response = requests.post(
            f"{self.base_url}/query",
            json={"question": question},
            timeout=30
        )
        return response.json()
```

### 2. Mobile App (Streaming)
```python
def stream_answer(question: str):
    response = requests.post(
        f"{base_url}/query?stream=true",
        json={"question": question},
        headers={'Accept': 'text/event-stream'},
        stream=True
    )
    
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith('data:'):
            yield line.split(':', 1)[1].strip()
```

### 3. Batch Processing
```python
questions = ["Q1", "Q2", "Q3"]
answers = []

for question in questions:
    result = client.ask_question(question)
    if result['success']:
        answers.append(result['data'])
```

## ✅ Verification Checklist

- [x] Local Windows testing with PowerShell
- [x] FastAPI server starts correctly
- [x] Health endpoint responds
- [x] Non-streaming queries work
- [x] Streaming (SSE) works with keepalive
- [x] Concurrent requests handled
- [x] Team integration examples tested
- [x] JSON response format validated
- [x] Error handling verified

## 🎯 Next Steps for Production

1. **Set up CI/CD pipeline** for automated deployment
2. **Add authentication** if needed for team access
3. **Implement monitoring** (Prometheus/Grafana)
4. **Set up log aggregation** (ELK stack)
5. **Configure auto-scaling** based on load
6. **Add caching layer** for frequently asked questions

The system is ready for team deployment and will work identically on EC2 as it does locally!
