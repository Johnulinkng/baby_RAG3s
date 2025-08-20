#!/usr/bin/env python3
"""
Simple Baby Care Application Example

This example shows how to integrate BabyCare RAG into a simple application.
"""

import sys
from pathlib import Path

# Add the parent directory to path to import babycare_rag
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from babycare_rag.api import BabyCareRAGAPI


class MyBabyApp:
    """Simple baby care application using RAG."""
    
    def __init__(self):
        """Initialize the application."""
        print("🍼 Initializing My Baby App...")
        
        try:
            self.rag_api = BabyCareRAGAPI()
            print("✅ BabyCare RAG system loaded successfully!")
            
            # Check system health
            health = self.rag_api.health_check()
            if health["success"] and health["data"]["status"] == "healthy":
                print(f"📚 Knowledge base: {health['data']['total_documents']} documents")
            else:
                print("⚠️  RAG system may have issues")
                
        except Exception as e:
            print(f"❌ Failed to initialize RAG system: {e}")
            print("Please check your configuration and try again.")
            sys.exit(1)
    
    def ask_baby_question(self, question: str) -> dict:
        """Ask a baby care question."""
        print(f"\n❓ Question: {question}")
        print("🤔 Thinking...")
        
        result = self.rag_api.query(question)
        
        if result["success"]:
            data = result["data"]
            response = {
                "answer": data["answer"],
                "sources": data["sources"],
                "confidence": data["confidence"],
                "success": True
            }
            
            print(f"🤖 Answer: {response['answer']}")
            if response["sources"]:
                print(f"📚 Sources: {', '.join(response['sources'][:3])}")
            print(f"🎯 Confidence: {response['confidence']:.2f}")
            
            return response
        else:
            print(f"❌ Error: {result['error']}")
            return {
                "answer": "Sorry, I couldn't answer that question.",
                "error": result["error"],
                "success": False
            }
    
    def add_custom_knowledge(self, content: str, title: str) -> bool:
        """Add custom knowledge to the system."""
        print(f"\n📝 Adding knowledge: {title}")
        
        result = self.rag_api.add_document(
            text_content=content,
            title=title
        )
        
        if result["success"]:
            print("✅ Knowledge added successfully!")
            return True
        else:
            print(f"❌ Failed to add knowledge: {result['error']}")
            return False
    
    def get_baby_advice_batch(self, questions: list) -> list:
        """Get advice for multiple questions."""
        print(f"\n🔄 Processing {len(questions)} questions...")
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\nProcessing {i}/{len(questions)}: {question[:50]}...")
            result = self.ask_baby_question(question)
            results.append({
                "question": question,
                "result": result
            })
        
        return results
    
    def search_knowledge(self, query: str, top_k: int = 3) -> list:
        """Search the knowledge base."""
        print(f"\n🔍 Searching for: {query}")
        
        result = self.rag_api.search_documents(query, top_k)
        
        if result["success"]:
            search_results = result["data"]
            print(f"Found {len(search_results)} results:")
            
            for i, res in enumerate(search_results, 1):
                print(f"\n{i}. {res['source']} (Score: {res['score']:.3f})")
                print(f"   {res['text'][:100]}...")
            
            return search_results
        else:
            print(f"❌ Search failed: {result['error']}")
            return []
    
    def show_system_stats(self):
        """Show system statistics."""
        print("\n📊 System Statistics:")
        
        stats = self.rag_api.get_stats()
        if stats["success"]:
            data = stats["data"]
            print(f"   Documents: {data['total_documents']}")
            print(f"   Chunks: {data['total_chunks']}")
            print(f"   Index Size: {data['index_size'] / 1024 / 1024:.2f} MB")
            print(f"   Model: {data['llm_model']}")
        else:
            print(f"   ❌ Failed to get stats: {stats['error']}")


def demo_basic_usage():
    """Demonstrate basic usage."""
    print("🎯 Demo: Basic Usage")
    print("=" * 30)
    
    app = MyBabyApp()
    
    # Ask some questions
    questions = [
        "What temperature should a baby's room be?",
        "How often should I feed my newborn?",
        "What should I do if my baby won't stop crying?"
    ]
    
    for question in questions:
        app.ask_baby_question(question)
    
    # Show stats
    app.show_system_stats()


def demo_custom_knowledge():
    """Demonstrate adding custom knowledge."""
    print("\n🎯 Demo: Custom Knowledge")
    print("=" * 30)
    
    app = MyBabyApp()
    
    # Add custom knowledge
    custom_knowledge = """
    Our Family Baby Care Guidelines:
    
    1. Room Temperature: We keep the nursery at 68-70°F (20-21°C)
    2. Feeding Schedule: Every 2-3 hours for newborns
    3. Sleep Routine: Bath at 7 PM, feeding at 7:30 PM, bedtime at 8 PM
    4. Emergency Contacts: Pediatrician Dr. Smith (555-1234)
    """
    
    success = app.add_custom_knowledge(custom_knowledge, "Family Guidelines")
    
    if success:
        # Test the new knowledge
        app.ask_baby_question("What is our family's preferred room temperature?")
        app.ask_baby_question("What is our bedtime routine?")


def demo_batch_processing():
    """Demonstrate batch processing."""
    print("\n🎯 Demo: Batch Processing")
    print("=" * 30)
    
    app = MyBabyApp()
    
    questions = [
        "When should I start solid foods?",
        "How to burp a baby properly?",
        "What are normal sleep patterns for newborns?",
        "How to soothe a colicky baby?"
    ]
    
    results = app.get_baby_advice_batch(questions)
    
    # Summary
    print("\n📋 Batch Results Summary:")
    successful = sum(1 for r in results if r["result"]["success"])
    print(f"   Successful: {successful}/{len(results)}")
    
    for result in results:
        status = "✅" if result["result"]["success"] else "❌"
        print(f"   {status} {result['question'][:50]}...")


def demo_search():
    """Demonstrate search functionality."""
    print("\n🎯 Demo: Search Functionality")
    print("=" * 30)
    
    app = MyBabyApp()
    
    search_queries = [
        "baby temperature",
        "feeding schedule",
        "sleep safety"
    ]
    
    for query in search_queries:
        app.search_knowledge(query, top_k=2)


def interactive_mode():
    """Run in interactive mode."""
    print("\n🎮 Interactive Mode")
    print("=" * 30)
    print("Type 'quit' to exit, 'help' for commands, or ask any baby care question:")
    
    app = MyBabyApp()
    
    while True:
        try:
            user_input = input("\n❓ Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            elif user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Ask any baby care question")
                print("  - 'stats' - Show system statistics")
                print("  - 'search <query>' - Search knowledge base")
                print("  - 'quit' - Exit")
            
            elif user_input.lower() == 'stats':
                app.show_system_stats()
            
            elif user_input.lower().startswith('search '):
                query = user_input[7:].strip()
                if query:
                    app.search_knowledge(query)
                else:
                    print("Please provide a search query.")
            
            else:
                app.ask_baby_question(user_input)
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="My Baby App - BabyCare RAG Integration Example")
    parser.add_argument("--demo", choices=["basic", "knowledge", "batch", "search", "all"], 
                       help="Run specific demo")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.demo == "basic":
        demo_basic_usage()
    elif args.demo == "knowledge":
        demo_custom_knowledge()
    elif args.demo == "batch":
        demo_batch_processing()
    elif args.demo == "search":
        demo_search()
    elif args.demo == "all":
        demo_basic_usage()
        demo_custom_knowledge()
        demo_batch_processing()
        demo_search()
    elif args.interactive:
        interactive_mode()
    else:
        # Default: show all demos
        demo_basic_usage()
        demo_custom_knowledge()
        demo_batch_processing()
        demo_search()
        
        # Ask if user wants interactive mode
        response = input("\n🎮 Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode()


if __name__ == "__main__":
    main()
