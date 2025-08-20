#!/usr/bin/env python3
"""
API test script for BabyCare RAG system.

This script demonstrates how to use the BabyCare RAG API programmatically.
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import babycare_rag
sys.path.insert(0, str(Path(__file__).parent.parent))

from babycare_rag.api import BabyCareRAGAPI, quick_query, quick_add_document


def test_basic_api():
    """Test basic API functionality."""
    print("🧪 Testing BabyCare RAG API...")
    
    # Initialize API
    api = BabyCareRAGAPI()
    
    # Test health check
    print("\n1. Health Check:")
    health = api.health_check()
    print(json.dumps(health, indent=2))
    
    # Test getting stats
    print("\n2. System Statistics:")
    stats = api.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Test listing documents
    print("\n3. List Documents:")
    docs = api.list_documents()
    print(json.dumps(docs, indent=2))
    
    # Test search
    print("\n4. Search Test:")
    search_result = api.search_documents("baby temperature")
    print(json.dumps(search_result, indent=2))
    
    # Test query
    print("\n5. Query Test:")
    query_result = api.query("What is the ideal temperature for a baby's room?")
    print(json.dumps(query_result, indent=2))


def test_quick_functions():
    """Test quick convenience functions."""
    print("\n🚀 Testing Quick Functions...")
    
    # Test quick query
    print("\n1. Quick Query:")
    answer = quick_query("How often should I feed my newborn baby?")
    print(f"Answer: {answer}")
    
    # Test with custom config
    print("\n2. Quick Query with Custom Config:")
    config = {
        "max_steps": 3,
        "top_k": 2
    }
    answer = quick_query("What should I do if my baby has a fever?", config)
    print(f"Answer: {answer}")


def test_document_management():
    """Test document management features."""
    print("\n📚 Testing Document Management...")
    
    api = BabyCareRAGAPI()
    
    # Test adding document from text
    print("\n1. Adding Document from Text:")
    result = api.add_document(
        text_content="Baby sleep safety: Always place babies on their back to sleep. Use a firm mattress and avoid loose bedding.",
        title="Sleep Safety Guidelines"
    )
    print(json.dumps(result, indent=2))
    
    # List documents again
    print("\n2. Updated Document List:")
    docs = api.list_documents()
    print(f"Total documents: {len(docs['data']) if docs['success'] else 0}")
    
    # Test search with new document
    print("\n3. Search New Content:")
    search_result = api.search_documents("baby sleep safety")
    print(json.dumps(search_result, indent=2))


def interactive_test():
    """Interactive test mode."""
    print("\n🎮 Interactive Test Mode")
    print("Type 'quit' to exit, or ask any baby care question:")
    
    api = BabyCareRAGAPI()
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not question:
                continue
            
            print("🤔 Processing...")
            result = api.query(question)
            
            if result["success"]:
                print(f"\n🤖 Answer: {result['data']['answer']}")
                if result['data']['sources']:
                    print(f"📚 Sources: {', '.join(result['data']['sources'])}")
                print(f"🎯 Confidence: {result['data']['confidence']:.2f}")
            else:
                print(f"❌ Error: {result['error']}")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BabyCare RAG API Test Tool")
    parser.add_argument("--basic", action="store_true", help="Run basic API tests")
    parser.add_argument("--quick", action="store_true", help="Test quick functions")
    parser.add_argument("--docs", action="store_true", help="Test document management")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run interactive mode")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    if args.all:
        test_basic_api()
        test_quick_functions()
        test_document_management()
        return
    
    if args.basic:
        test_basic_api()
    
    if args.quick:
        test_quick_functions()
    
    if args.docs:
        test_document_management()
    
    if args.interactive:
        interactive_test()
    
    if not any([args.basic, args.quick, args.docs, args.interactive, args.all]):
        # Default: run interactive mode
        interactive_test()


if __name__ == "__main__":
    main()
