#!/usr/bin/env python3
"""
Command-line test tool for BabyCare RAG system.

This tool allows you to test the RAG system through command-line interface.
You can ask questions and get answers from the baby care knowledge base.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import babycare_rag
sys.path.insert(0, str(Path(__file__).parent.parent))

from babycare_rag import BabyCareRAG, RAGConfig
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
import argparse


class BabyCareRAGCLI:
    """Command-line interface for BabyCare RAG system."""
    
    def __init__(self):
        self.console = Console()
        self.rag = None
        self.setup_rag()
    
    def setup_rag(self):
        """Initialize the RAG system."""
        try:
            self.console.print("[yellow]Initializing BabyCare RAG system...[/yellow]")
            
            # Try to load configuration
            config = RAGConfig.from_env()
            self.rag = BabyCareRAG(config)
            
            self.console.print("[green]✓ RAG system initialized successfully![/green]")
            
            # Show system stats
            stats = self.rag.get_stats()
            self.console.print(f"[blue]Documents: {stats.total_documents}, Chunks: {stats.total_chunks}[/blue]")
            
        except Exception as e:
            self.console.print(f"[red]Error initializing RAG system: {e}[/red]")
            self.console.print("[yellow]Please check your configuration and try again.[/yellow]")
            sys.exit(1)
    
    def show_welcome(self):
        """Show welcome message."""
        welcome_text = """
🍼 Welcome to BabyCare RAG CLI Test Tool! 🍼

This tool allows you to test the baby care knowledge system.
Ask any questions about baby care, safety, development, or parenting.

Available commands:
• Type your question and press Enter
• 'help' - Show this help message
• 'stats' - Show system statistics
• 'docs' - List all documents
• 'search <query>' - Search documents
• 'add <file_path>' - Add a document
• 'quit' or 'exit' - Exit the program
        """
        
        self.console.print(Panel(welcome_text, title="BabyCare RAG CLI", border_style="blue"))
    
    def show_help(self):
        """Show help information."""
        help_table = Table(title="Available Commands")
        help_table.add_column("Command", style="cyan", no_wrap=True)
        help_table.add_column("Description", style="white")
        help_table.add_column("Example", style="yellow")
        
        help_table.add_row("question", "Ask a baby care question", "What temperature should baby's room be?")
        help_table.add_row("help", "Show this help message", "help")
        help_table.add_row("stats", "Show system statistics", "stats")
        help_table.add_row("docs", "List all documents", "docs")
        help_table.add_row("search <query>", "Search documents", "search baby feeding")
        help_table.add_row("add <file_path>", "Add a document", "add /path/to/document.pdf")
        help_table.add_row("quit/exit", "Exit the program", "quit")
        
        self.console.print(help_table)
    
    def show_stats(self):
        """Show system statistics."""
        try:
            stats = self.rag.get_stats()
            
            stats_table = Table(title="System Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            
            stats_table.add_row("Total Documents", str(stats.total_documents))
            stats_table.add_row("Total Chunks", str(stats.total_chunks))
            stats_table.add_row("Index Size", f"{stats.index_size / 1024 / 1024:.2f} MB")
            stats_table.add_row("Storage Used", f"{stats.storage_used / 1024 / 1024:.2f} MB")
            stats_table.add_row("Embedding Model", stats.embedding_model)
            stats_table.add_row("LLM Model", stats.llm_model)
            stats_table.add_row("Last Updated", stats.last_updated)
            
            self.console.print(stats_table)
            
        except Exception as e:
            self.console.print(f"[red]Error getting stats: {e}[/red]")
    
    def list_documents(self):
        """List all documents."""
        try:
            documents = self.rag.list_documents()
            
            if not documents:
                self.console.print("[yellow]No documents found in the knowledge base.[/yellow]")
                return
            
            docs_table = Table(title="Documents in Knowledge Base")
            docs_table.add_column("Title", style="cyan")
            docs_table.add_column("Chunks", style="white")
            docs_table.add_column("Type", style="yellow")
            docs_table.add_column("Added", style="green")
            
            for doc in documents:
                docs_table.add_row(
                    doc.title,
                    str(doc.chunk_count),
                    doc.doc_type or "unknown",
                    doc.added_date
                )
            
            self.console.print(docs_table)
            
        except Exception as e:
            self.console.print(f"[red]Error listing documents: {e}[/red]")
    
    def search_documents(self, query: str):
        """Search documents."""
        try:
            self.console.print(f"[yellow]Searching for: {query}[/yellow]")
            results = self.rag.search_documents(query, top_k=5)
            
            if not results:
                self.console.print("[yellow]No results found.[/yellow]")
                return
            
            for i, result in enumerate(results, 1):
                panel_title = f"Result {i} - {result.source} (Score: {result.score:.3f})"
                text_preview = result.text[:200] + "..." if len(result.text) > 200 else result.text
                self.console.print(Panel(text_preview, title=panel_title, border_style="green"))
            
        except Exception as e:
            self.console.print(f"[red]Error searching documents: {e}[/red]")
    
    def add_document(self, file_path: str):
        """Add a document."""
        try:
            self.console.print(f"[yellow]Adding document: {file_path}[/yellow]")
            
            if not os.path.exists(file_path):
                self.console.print(f"[red]File not found: {file_path}[/red]")
                return
            
            success = self.rag.add_document(file_path)
            
            if success:
                self.console.print("[green]✓ Document added successfully![/green]")
            else:
                self.console.print("[red]✗ Failed to add document.[/red]")
            
        except Exception as e:
            self.console.print(f"[red]Error adding document: {e}[/red]")
    
    def ask_question(self, question: str):
        """Ask a question to the RAG system."""
        try:
            self.console.print(f"[yellow]Processing question: {question}[/yellow]")
            
            with self.console.status("[bold green]Thinking..."):
                response = self.rag.query(question)
            
            # Display the answer
            self.console.print(Panel(
                response.answer,
                title="🤖 BabyCare Assistant Answer",
                border_style="blue"
            ))
            
            # Display sources if available
            if response.sources:
                sources_text = "📚 Sources: " + ", ".join(response.sources)
                self.console.print(f"[dim]{sources_text}[/dim]")
            
            # Display confidence
            confidence_color = "green" if response.confidence > 0.7 else "yellow" if response.confidence > 0.4 else "red"
            self.console.print(f"[{confidence_color}]Confidence: {response.confidence:.2f}[/{confidence_color}]")
            
        except Exception as e:
            self.console.print(f"[red]Error processing question: {e}[/red]")
    
    def run(self):
        """Run the CLI interface."""
        self.show_welcome()
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]BabyCare RAG[/bold cyan]").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit']:
                    self.console.print("[green]Goodbye! 👋[/green]")
                    break
                
                elif user_input.lower() == 'help':
                    self.show_help()
                
                elif user_input.lower() == 'stats':
                    self.show_stats()
                
                elif user_input.lower() == 'docs':
                    self.list_documents()
                
                elif user_input.lower().startswith('search '):
                    query = user_input[7:].strip()
                    if query:
                        self.search_documents(query)
                    else:
                        self.console.print("[red]Please provide a search query.[/red]")
                
                elif user_input.lower().startswith('add '):
                    file_path = user_input[4:].strip()
                    if file_path:
                        self.add_document(file_path)
                    else:
                        self.console.print("[red]Please provide a file path.[/red]")
                
                else:
                    # Treat as a question
                    self.ask_question(user_input)
            
            except KeyboardInterrupt:
                self.console.print("\n[green]Goodbye! 👋[/green]")
                break
            except Exception as e:
                self.console.print(f"[red]Unexpected error: {e}[/red]")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="BabyCare RAG CLI Test Tool")
    parser.add_argument("--question", "-q", help="Ask a single question and exit")
    parser.add_argument("--add-doc", help="Add a document and exit")
    parser.add_argument("--search", help="Search documents and exit")
    
    args = parser.parse_args()
    
    cli = BabyCareRAGCLI()
    
    # Handle single commands
    if args.question:
        cli.ask_question(args.question)
        return
    
    if args.add_doc:
        cli.add_document(args.add_doc)
        return
    
    if args.search:
        cli.search_documents(args.search)
        return
    
    # Run interactive mode
    cli.run()


if __name__ == "__main__":
    main()
