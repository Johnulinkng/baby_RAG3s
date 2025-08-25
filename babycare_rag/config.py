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
    openai_api_key: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key for LLM/Embeddings"
    )

    # Embedding/Model Configuration
    embed_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        description="Embedding model name"
    )

    llm_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
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
        # Allow either OPENAI_API_KEY env or AWS Secrets (SECRET_ID + AWS_REGION)
        if not (self.openai_api_key or (os.getenv("SECRET_ID") and os.getenv("AWS_REGION"))):
            raise ValueError("OpenAI credentials missing: set OPENAI_API_KEY or configure SECRET_ID and AWS_REGION")

        if abs((self.bm25_weight + self.vector_weight) - 1.0) > 1e-6:
            raise ValueError("BM25 and vector weights must sum to 1.0")

        return True
    
    @classmethod
    def from_env(cls) -> "RAGConfig":
        """Create configuration from environment variables."""
        return cls()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return self.model_dump()
