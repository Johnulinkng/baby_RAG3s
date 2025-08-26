#!/bin/bash

# BabyCare RAG 3s - Automated Deployment Script
# Usage: curl -fsSL https://raw.githubusercontent.com/Johnulinkng/baby_RAG3s/main/deploy.sh | bash

set -e

echo "🍼 BabyCare RAG 3s - Automated Deployment"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root for security reasons"
   exit 1
fi

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    if command -v apt-get &> /dev/null; then
        OS="ubuntu"
        INSTALL_CMD="sudo apt-get install -y"
        UPDATE_CMD="sudo apt-get update && sudo apt-get upgrade -y"
    elif command -v yum &> /dev/null; then
        OS="centos"
        INSTALL_CMD="sudo yum install -y"
        UPDATE_CMD="sudo yum update -y"
    else
        print_error "Unsupported Linux distribution"
        exit 1
    fi
else
    print_error "This script only supports Linux"
    exit 1
fi

print_status "Detected OS: $OS"

# Check system requirements
print_status "Checking system requirements..."

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc -l) -eq 1 ]]; then
        print_status "Python $PYTHON_VERSION found ✓"
    else
        print_warning "Python $PYTHON_VERSION found, but 3.10+ recommended"
    fi
else
    print_status "Python3 not found, will install..."
fi

# Update system
print_status "Updating system packages..."
$UPDATE_CMD

# Install required packages
print_status "Installing required packages..."
if [[ "$OS" == "ubuntu" ]]; then
    $INSTALL_CMD python3 python3-venv python3-pip git curl bc
elif [[ "$OS" == "centos" ]]; then
    $INSTALL_CMD python3 python3-pip git curl bc
fi

# Check available memory
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ $MEMORY_GB -lt 4 ]]; then
    print_warning "Only ${MEMORY_GB}GB RAM available. 4GB+ recommended for optimal performance."
    WORKERS=1
else
    print_status "Memory check passed: ${MEMORY_GB}GB RAM available ✓"
    WORKERS=2
fi

# Clone repository
print_status "Cloning BabyCare RAG 3s repository..."
if [[ -d "baby_RAG3s" ]]; then
    print_warning "Directory baby_RAG3s already exists, updating..."
    cd baby_RAG3s
    git pull origin main
else
    git clone https://github.com/Johnulinkng/baby_RAG3s.git
    cd baby_RAG3s
fi

# Create virtual environment
print_status "Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
print_status "Installing Python dependencies..."
pip install --upgrade pip
pip install -e .
pip install fastapi uvicorn rank-bm25

# Check for OpenAI API key
if [[ -z "$OPENAI_API_KEY" ]]; then
    print_warning "OPENAI_API_KEY environment variable not set"
    echo -e "${BLUE}Please enter your OpenAI API key:${NC}"
    read -s OPENAI_API_KEY
    export OPENAI_API_KEY
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" > .env
    print_status "API key saved to .env file"
else
    print_status "OpenAI API key found ✓"
fi

# Initialize system
print_status "Initializing RAG system (building indexes)..."
python setup_rag.py

# Test the system
print_status "Testing system health..."
python -c "from babycare_rag.api import BabyCareRAGAPI; api = BabyCareRAGAPI(); result = api.health_check(); print('✓ Health check passed' if result['success'] else '✗ Health check failed')"

# Create systemd service
print_status "Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/babycare-rag.service"
sudo tee $SERVICE_FILE > /dev/null <<EOF
[Unit]
Description=BabyCare RAG API
After=network.target

[Service]
Type=exec
User=$USER
WorkingDirectory=$PWD
Environment=PATH=$PWD/.venv/bin
Environment=OPENAI_API_KEY=$OPENAI_API_KEY
ExecStart=$PWD/.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000 --workers $WORKERS
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable babycare-rag
sudo systemctl start babycare-rag

# Wait for service to start
print_status "Starting service..."
sleep 5

# Check service status
if sudo systemctl is-active --quiet babycare-rag; then
    print_status "Service started successfully ✓"
else
    print_error "Service failed to start"
    sudo systemctl status babycare-rag
    exit 1
fi

# Test API endpoints
print_status "Testing API endpoints..."
sleep 2

# Health check
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [[ $HEALTH_RESPONSE -eq 200 ]]; then
    print_status "Health endpoint working ✓"
else
    print_error "Health endpoint failed (HTTP $HEALTH_RESPONSE)"
fi

# Test query endpoint
print_status "Testing query endpoint..."
QUERY_RESPONSE=$(curl -s -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the ABCs of Safe Sleep?"}' \
  -w "%{http_code}" -o /tmp/query_test.json)

if [[ $QUERY_RESPONSE -eq 200 ]]; then
    print_status "Query endpoint working ✓"
    ANSWER=$(cat /tmp/query_test.json | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data']['answer'][:100] + '...' if data.get('success') else 'Failed')")
    print_status "Sample answer: $ANSWER"
else
    print_error "Query endpoint failed (HTTP $QUERY_RESPONSE)"
fi

# Get server IP
SERVER_IP=$(curl -s ifconfig.me || echo "localhost")

# Print deployment summary
echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo -e "${GREEN}Service Status:${NC} $(sudo systemctl is-active babycare-rag)"
echo -e "${GREEN}Service URL:${NC} http://$SERVER_IP:8000"
echo -e "${GREEN}Health Check:${NC} http://$SERVER_IP:8000/health"
echo -e "${GREEN}Workers:${NC} $WORKERS"
echo -e "${GREEN}Memory:${NC} ${MEMORY_GB}GB RAM"
echo ""
echo "📋 Management Commands:"
echo "  sudo systemctl status babycare-rag    # Check status"
echo "  sudo systemctl restart babycare-rag   # Restart service"
echo "  sudo systemctl stop babycare-rag      # Stop service"
echo "  sudo journalctl -u babycare-rag -f    # View logs"
echo ""
echo "🧪 Test Commands:"
echo "  curl http://localhost:8000/health"
echo "  curl -X POST \"http://localhost:8000/query\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"question\":\"What are the ABCs of Safe Sleep?\"}'"
echo ""
echo "📚 Documentation: https://github.com/Johnulinkng/baby_RAG3s"
echo ""
print_status "BabyCare RAG 3s is now running and ready for production use!"

# Cleanup
rm -f /tmp/query_test.json
