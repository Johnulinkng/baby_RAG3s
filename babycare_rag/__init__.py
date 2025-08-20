"""
BabyCare RAG - A RAG-powered baby care assistant tool for team integration.

This package provides a complete RAG (Retrieval-Augmented Generation) system
specifically designed for baby care knowledge management and question answering.
"""

from .core import BabyCareRAG
from .config import RAGConfig
from .models import RAGResponse, DocumentInfo, SearchResult, SystemStats

__version__ = "0.1.0"
__all__ = [
    "BabyCareRAG",
    "RAGConfig", 
    "RAGResponse",
    "DocumentInfo",
    "SearchResult",
    "SystemStats"
]
