"""Configuration management for BabyCare RAG system."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class RAGConfig(BaseModel):
    """Configuration for the RAG system."""
    
    # API Keys and URLs
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Google Gemini API key for LLM"
    )
    
    ollama_base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        description="Ollama server base URL for embeddings"
    )
    
    # Model Configuration
    embed_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        description="Embedding model name"
    )
    
    llm_model: str = Field(
        default="gemini-1.5-flash",
        description="LLM model name for generation"
    )
    
    # RAG Parameters
    max_steps: int = Field(
        default=5,
        description="Maximum reasoning steps for agent"
    )
    
    top_k: int = Field(
        default=3,
        description="Number of top documents to retrieve"
    )
    
    chunk_size: int = Field(
        default=1000,
        description="Document chunk size for processing"
    )
    
    chunk_overlap: int = Field(
        default=200,
        description="Overlap between document chunks"
    )
    
    # Storage Paths
    documents_dir: str = Field(
        default="documents",
        description="Directory for storing documents"
    )
    
    index_dir: str = Field(
        default="faiss_index",
        description="Directory for FAISS index storage"
    )
    
    # Search Configuration
    search_top_k: int = Field(
        default=20,
        description="Initial search results before reranking"
    )
    
    bm25_weight: float = Field(
        default=0.3,
        description="Weight for BM25 in hybrid search"
    )
    
    vector_weight: float = Field(
        default=0.7,
        description="Weight for vector search in hybrid search"
    )
    
    def validate_config(self) -> bool:
        """Validate the configuration."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required")
        
        if self.bm25_weight + self.vector_weight != 1.0:
            raise ValueError("BM25 and vector weights must sum to 1.0")
        
        return True
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()
