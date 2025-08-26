# 🚀 Quick Deploy Guide - BabyCare RAG 3s

## 📋 Overview

This guide helps you deploy BabyCare RAG 3s on EC2 or any Linux server in under 5 minutes.

## ⚡ One-Command Deploy (EC2/Linux)

```bash
# Download and run the deployment script
curl -fsSL https://raw.githubusercontent.com/Johnulinkng/baby_RAG3s/main/deploy.sh | bash
```

## 🔧 Manual Deployment

### Step 1: Server Setup

```bash
# Update system (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip git curl

# For CentOS/RHEL
# sudo yum update -y
# sudo yum install -y python3 python3-pip git curl
```

### Step 2: Clone and Setup

```bash
# Clone repository
git clone https://github.com/Johnulinkng/baby_RAG3s.git
cd baby_RAG3s

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install fastapi uvicorn rank-bm25
```

### Step 3: Configure Environment

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-openai-api-key-here"

# Or create .env file
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
```

### Step 4: Initialize System

```bash
# Build indexes (takes ~30 seconds)
python setup_rag.py
```

### Step 5: Start Service

```bash
# Development mode
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 2

# Production mode (background)
nohup uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4 > rag_server.log 2>&1 &
```

### Step 6: Test Deployment

```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the ABCs of Safe Sleep?"}'

# Test streaming
curl -N -X POST "http://localhost:8000/query?stream=true" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question":"What is the ideal room temperature for a baby'\''s nursery?"}'
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Clone repository
git clone https://github.com/Johnulinkng/baby_RAG3s.git
cd baby_RAG3s

# Set environment variables
export OPENAI_API_KEY="your-openai-api-key-here"

# Build and run
docker-compose up -d
```

### Manual Docker Build

```bash
# Build image
docker build -t babycare-rag .

# Run container
docker run -d \
  --name babycare-rag \
  -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key-here" \
  babycare-rag
```

## 🔄 Production Setup (systemd)

### Create Service File

```bash
sudo tee /etc/systemd/system/babycare-rag.service > /dev/null <<EOF
[Unit]
Description=BabyCare RAG API
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/.venv/bin
Environment=OPENAI_API_KEY=your-openai-api-key-here
ExecStart=$PWD/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
```

### Enable and Start Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service
sudo systemctl enable babycare-rag

# Start service
sudo systemctl start babycare-rag

# Check status
sudo systemctl status babycare-rag

# View logs
sudo journalctl -u babycare-rag -f
```

## 🌐 Nginx Reverse Proxy (Optional)

### Install Nginx

```bash
sudo apt install -y nginx
```

### Configure Nginx

```bash
sudo tee /etc/nginx/sites-available/babycare-rag > /dev/null <<EOF
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # Important for streaming
        proxy_buffering off;
        proxy_cache off;
        add_header X-Accel-Buffering no;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/babycare-rag /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 🔒 Security Considerations

### Firewall Setup

```bash
# Allow HTTP and HTTPS
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 8000  # If accessing directly

# Enable firewall
sudo ufw enable
```

### SSL Certificate (Let's Encrypt)

```bash
# Install certbot
sudo apt install -y certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Monitoring

### Basic Monitoring

```bash
# Check service status
sudo systemctl status babycare-rag

# View logs
sudo journalctl -u babycare-rag -f

# Check resource usage
htop
df -h
free -h
```

### Health Check Script

```bash
#!/bin/bash
# health_check.sh
HEALTH_URL="http://localhost:8000/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" $HEALTH_URL)

if [ $RESPONSE -eq 200 ]; then
    echo "✅ Service is healthy"
    exit 0
else
    echo "❌ Service is unhealthy (HTTP $RESPONSE)"
    exit 1
fi
```

## 🚨 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   sudo lsof -i :8000
   sudo kill -9 <PID>
   ```

2. **Permission denied**
   ```bash
   sudo chown -R $USER:$USER /path/to/baby_RAG3s
   chmod +x deploy.sh
   ```

3. **Python version issues**
   ```bash
   python3 --version  # Should be 3.10+
   sudo apt install -y python3.11 python3.11-venv
   ```

4. **Memory issues**
   ```bash
   free -h  # Check available memory
   # Reduce workers if needed: --workers 1
   ```

### Log Locations

- **systemd service**: `sudo journalctl -u babycare-rag`
- **Manual run**: `./rag_server.log`
- **Docker**: `docker logs babycare-rag`
- **Nginx**: `/var/log/nginx/access.log`, `/var/log/nginx/error.log`

## 📞 Support

- **GitHub Issues**: https://github.com/Johnulinkng/baby_RAG3s/issues
- **Documentation**: See README.md for detailed usage
- **Performance**: Expected 3-4 second response times

---

**🎉 Your BabyCare RAG 3s is now deployed and ready for production use!**
