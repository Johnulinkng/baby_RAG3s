"""API service layer for BabyCare RAG system."""

import json
import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path

from .core import BabyCareRAG
from .config import RAGConfig
from .models import (
    RAGResponse, DocumentInfo, SearchResult, SystemStats,
    QueryRequest, AddDocumentRequest
)


class BabyCareRAGAPI:
    """API wrapper for BabyCare RAG system."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the API."""
        self.rag = BabyCareRAG(config)
    
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The user's question
            **kwargs: Additional parameters (max_steps, session_id, etc.)
        
        Returns:
            Dictionary containing the response
        """
        try:
            request = QueryRequest(question=question, **kwargs)
            response = self.rag.process_request(request)
            
            return {
                "success": True,
                "data": response.model_dump(),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def add_document(self, **kwargs) -> Dict[str, Any]:
        """
        Add a document to the knowledge base.
        
        Args:
            **kwargs: Document parameters (file_path, url, text_content, title, etc.)
        
        Returns:
            Dictionary containing the result
        """
        try:
            request = AddDocumentRequest(**kwargs)
            success = self.rag.add_document_request(request)
            
            return {
                "success": success,
                "data": {"added": success},
                "error": None if success else "Failed to add document"
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def list_documents(self) -> Dict[str, Any]:
        """List all documents in the knowledge base."""
        try:
            documents = self.rag.list_documents()
            
            return {
                "success": True,
                "data": [doc.model_dump() for doc in documents],
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def remove_document(self, doc_id: str) -> Dict[str, Any]:
        """Remove a document from the knowledge base."""
        try:
            success = self.rag.remove_document(doc_id)
            
            return {
                "success": success,
                "data": {"removed": success},
                "error": None if success else "Failed to remove document"
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def search_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search documents."""
        try:
            results = self.rag.search_documents(query, top_k)
            
            return {
                "success": True,
                "data": [result.model_dump() for result in results],
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            stats = self.rag.get_stats()
            
            return {
                "success": True,
                "data": stats.model_dump(),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def rebuild_index(self) -> Dict[str, Any]:
        """Rebuild the search index."""
        try:
            success = self.rag.rebuild_index()
            
            return {
                "success": success,
                "data": {"rebuilt": success},
                "error": None if success else "Failed to rebuild index"
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
        try:
            health_data = self.rag.health_check()
            
            return {
                "success": True,
                "data": health_data,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def update_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Update system configuration."""
        try:
            config = RAGConfig(**config_dict)
            success = self.rag.update_config(config)
            
            return {
                "success": success,
                "data": {"updated": success},
                "error": None if success else "Failed to update configuration"
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        try:
            config = self.rag.get_config()
            
            return {
                "success": True,
                "data": config.model_dump(),
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": None,
                "error": str(e),
                "traceback": traceback.format_exc()
            }


# Convenience functions for direct usage
def create_rag_api(config_dict: Optional[Dict[str, Any]] = None) -> BabyCareRAGAPI:
    """Create a RAG API instance."""
    config = RAGConfig(**config_dict) if config_dict else None
    return BabyCareRAGAPI(config)


def quick_query(question: str, config_dict: Optional[Dict[str, Any]] = None) -> str:
    """Quick query function for simple usage."""
    api = create_rag_api(config_dict)
    result = api.query(question)
    
    if result["success"]:
        return result["data"]["answer"]
    else:
        return f"Error: {result['error']}"


def quick_add_document(file_path: str, config_dict: Optional[Dict[str, Any]] = None) -> bool:
    """Quick function to add a document."""
    api = create_rag_api(config_dict)
    result = api.add_document(file_path=file_path)
    return result["success"]
