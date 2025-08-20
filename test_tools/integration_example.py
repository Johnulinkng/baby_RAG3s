#!/usr/bin/env python3
"""
Integration example for BabyCare RAG system.

This example shows how to integrate the BabyCare RAG system into your own project.
"""

import sys
from pathlib import Path

# Add parent directory to path to import babycare_rag
sys.path.insert(0, str(Path(__file__).parent.parent))

from babycare_rag import BabyCareRAG, RAGConfig
from babycare_rag.api import BabyCareRAGAPI


class MyBabyCareApp:
    """Example application using BabyCare RAG."""
    
    def __init__(self):
        """Initialize the application."""
        # Option 1: Use the direct RAG class
        self.rag = BabyCareRAG()
        
        # Option 2: Use the API wrapper (recommended for most use cases)
        self.api = BabyCareRAGAPI()
        
        print("✅ BabyCare RAG integrated successfully!")
    
    def ask_question(self, question: str) -> dict:
        """Ask a question and get a structured response."""
        # Using the API wrapper for better error handling
        result = self.api.query(question)
        
        if result["success"]:
            return {
                "answer": result["data"]["answer"],
                "sources": result["data"]["sources"],
                "confidence": result["data"]["confidence"],
                "success": True
            }
        else:
            return {
                "answer": "Sorry, I couldn't process your question.",
                "error": result["error"],
                "success": False
            }
    
    def add_knowledge(self, content: str, title: str) -> bool:
        """Add new knowledge to the system."""
        result = self.api.add_document(text_content=content, title=title)
        return result["success"]
    
    def get_system_info(self) -> dict:
        """Get information about the RAG system."""
        stats = self.api.get_stats()
        if stats["success"]:
            return stats["data"]
        return {}


def example_usage():
    """Example of how to use the BabyCare RAG in your application."""
    
    print("🍼 BabyCare RAG Integration Example")
    print("=" * 50)
    
    # Initialize your application
    app = MyBabyCareApp()
    
    # Get system information
    print("\n📊 System Information:")
    info = app.get_system_info()
    print(f"Documents: {info.get('total_documents', 0)}")
    print(f"Chunks: {info.get('total_chunks', 0)}")
    print(f"Model: {info.get('llm_model', 'Unknown')}")
    
    # Add some knowledge (optional)
    print("\n📚 Adding Custom Knowledge:")
    success = app.add_knowledge(
        content="Important: Never leave a baby unattended on a changing table. Always keep one hand on the baby.",
        title="Changing Table Safety"
    )
    print(f"Knowledge added: {'✅' if success else '❌'}")
    
    # Ask questions
    print("\n❓ Asking Questions:")
    
    questions = [
        "What temperature should a baby's room be?",
        "How often should I feed my newborn?",
        "What should I do if my baby won't stop crying?",
        "Is it safe to use a blanket for a 3-month-old baby?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        response = app.ask_question(question)
        
        if response["success"]:
            print(f"A: {response['answer'][:200]}...")
            if response["sources"]:
                print(f"📚 Sources: {', '.join(response['sources'][:2])}")
            print(f"🎯 Confidence: {response['confidence']:.2f}")
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")


def custom_config_example():
    """Example of using custom configuration."""
    
    print("\n⚙️ Custom Configuration Example")
    print("=" * 50)
    
    # Create custom configuration
    custom_config = RAGConfig(
        max_steps=3,  # Limit reasoning steps
        top_k=2,      # Return fewer documents
        chunk_size=500,  # Smaller chunks
    )
    
    # Initialize with custom config
    rag = BabyCareRAG(custom_config)
    
    # Test with custom settings
    response = rag.query("What are the signs of teething in babies?")
    print(f"Answer: {response.answer[:200]}...")
    print(f"Processing steps: {len(response.processing_steps)}")


def batch_processing_example():
    """Example of processing multiple questions in batch."""
    
    print("\n🔄 Batch Processing Example")
    print("=" * 50)
    
    api = BabyCareRAGAPI()
    
    questions = [
        "When should I start solid foods?",
        "How to burp a baby properly?",
        "What are normal sleep patterns for newborns?",
        "How to soothe a colicky baby?"
    ]
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"Processing question {i}/{len(questions)}...")
        result = api.query(question)
        results.append({
            "question": question,
            "answer": result["data"]["answer"] if result["success"] else "Error",
            "success": result["success"]
        })
    
    # Display results
    print("\n📋 Batch Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result["success"] else "❌"
        print(f"{i}. {status} {result['question']}")
        print(f"   {result['answer'][:100]}...")
        print()


def main():
    """Main function with different examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BabyCare RAG Integration Examples")
    parser.add_argument("--basic", action="store_true", help="Run basic integration example")
    parser.add_argument("--config", action="store_true", help="Run custom configuration example")
    parser.add_argument("--batch", action="store_true", help="Run batch processing example")
    parser.add_argument("--all", action="store_true", help="Run all examples")
    
    args = parser.parse_args()
    
    if args.all:
        example_usage()
        custom_config_example()
        batch_processing_example()
        return
    
    if args.basic or not any([args.config, args.batch]):
        example_usage()
    
    if args.config:
        custom_config_example()
    
    if args.batch:
        batch_processing_example()


if __name__ == "__main__":
    main()
