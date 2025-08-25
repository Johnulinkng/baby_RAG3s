#!/usr/bin/env python3
"""
Setup script for BabyCare RAG system.

This script helps initialize the RAG system and verify the setup.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def check_environment():
    """Check if environment is properly set up."""
    print("🔍 Checking environment setup...")
    
    # Load environment variables
    load_dotenv()
    
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 10):
        issues.append("Python 3.10+ is required")
    else:
        print("✅ Python version OK")
    
    # Check OpenAI credentials (env or AWS Secrets)
    openai_key = os.getenv("OPENAI_API_KEY")
    secret_id = os.getenv("SECRET_ID")
    aws_region = os.getenv("AWS_REGION")
    if not (openai_key or (secret_id and aws_region)):
        issues.append("OpenAI credentials missing: set OPENAI_API_KEY or configure SECRET_ID and AWS_REGION")
    else:
        print("✅ OpenAI credentials configured (env or AWS Secrets)")

    # Show model choices
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    llm_model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    print(f"ℹ️  Embedding model: {embed_model}")
    print(f"ℹ️  LLM model: {llm_model}")

    return issues

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        "faiss",
        "openai",
        "markitdown",
        "mcp",
        "PIL",
        "rich",
        "scipy",
        "tqdm",
        "requests",
        "pydantic",
        "numpy",
        "dotenv"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
            elif package == "dotenv":
                import dotenv
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    return missing

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "documents",
        "faiss_index",
        "babycare_rag"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ {directory}/")

def test_rag_system():
    """Test if the RAG system can be initialized."""
    print("\n🧪 Testing RAG system...")
    
    try:
        # Try to import the RAG system
        from babycare_rag import BabyCareRAG, RAGConfig
        print("✅ BabyCare RAG import successful")
        
        # Try to create config
        config = RAGConfig.from_env()
        print("✅ Configuration created")
        
        # Try to initialize RAG (this might fail if Ollama is not running)
        try:
            rag = BabyCareRAG(config)
            print("✅ RAG system initialized")
            
            # Get stats
            stats = rag.get_stats()
            print(f"✅ System stats: {stats.total_documents} documents, {stats.total_chunks} chunks")
            
            return True
            
        except Exception as e:
            print(f"⚠️  RAG initialization failed: {e}")
            print("   This might be because Ollama is not running or not accessible")
            return False
            
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")
        return False

def test_ollama_connection():
    """Test connection to Ollama server."""
    print("\n🔗 Testing Ollama connection...")
    
    import requests
    
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama server is accessible")
            
            # Check if embedding model is available
            models = response.json().get("models", [])
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            
            model_names = [model.get("name", "").split(":")[0] for model in models]
            if embed_model in model_names:
                print(f"✅ Embedding model '{embed_model}' is available")
                return True
            else:
                print(f"⚠️  Embedding model '{embed_model}' not found")
                print(f"   Available models: {', '.join(model_names)}")
                print(f"   You may need to run: ollama pull {embed_model}")
                return False
        else:
            print(f"❌ Ollama server returned status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Ollama server: {e}")
        print("   Make sure Ollama is installed and running")
        print("   Install: https://ollama.ai/")
        print("   Start: ollama serve")
        return False

def create_env_file():
    """Create .env file if it doesn't exist."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("\n📝 Creating .env file...")
    
    # Check if template exists
    template_file = Path("env-template")
    if template_file.exists():
        import shutil
        shutil.copy(template_file, env_file)
        print("✅ .env file created from template")
        print("   Please edit .env and add your OPENAI_API_KEY or configure SECRET_ID/AWS_REGION")
    else:
        # Create basic .env file
        env_content = """# BabyCare RAG Configuration

# Option A: OpenAI direct
OPENAI_API_KEY=sk-...
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Option B: AWS Secrets Manager (production)
# SECRET_ID corresponds to your secret name (e.g., Opean_AI_KEY_IOSAPP)
SECRET_ID=Opean_AI_KEY_IOSAPP
AWS_REGION=us-east-2

# Optional app tables
MSG_TABLE_NAME=ChatMessages
STATE_TABLE_NAME=ActiveAgentState
"""
        env_file.write_text(env_content)
        print("✅ Basic .env file created")
        print("   Please edit .env and add your OPENAI_API_KEY or configure SECRET_ID/AWS_REGION")

def main():
    """Main setup function."""
    print("🍼 BabyCare RAG Setup Script")
    print("=" * 50)
    
    # Create .env file if needed
    create_env_file()
    
    # Check environment
    env_issues = check_environment()
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    # Create directories
    create_directories()
    
    # Test RAG system
    rag_ok = test_rag_system()
    
    # Summary
    print("\n📋 Setup Summary")
    print("=" * 30)
    
    if env_issues:
        print("❌ Environment Issues:")
        for issue in env_issues:
            print(f"   - {issue}")
    
    if missing_deps:
        print("❌ Missing Dependencies:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("   Run: pip install -e .")
    

    
    if env_issues or missing_deps:
        print("\n❌ Setup incomplete. Please fix the issues above.")
        return False
    
    if rag_ok:
        print("\n✅ Setup complete! The BabyCare RAG system is ready to use.")
        print("\nNext steps:")
        print("   1. Add documents: python test_tools/cli_test.py")
        print("   2. Test queries: python test_tools/api_test.py")
        print("   3. See examples: python test_tools/integration_example.py")
        return True
    else:
        print("\n⚠️  Setup issues detected. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
