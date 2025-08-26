"""Core BabyCare RAG system implementation."""

import os
import time
import asyncio
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from .config import RAGConfig
from .models import (
    RAGResponse, DocumentInfo, SearchResult, SystemStats,
    QueryRequest, AddDocumentRequest
)
from .document_processor import DocumentProcessor
from .search_engine import SearchEngine


class BabyCareRAG:
    """Main BabyCare RAG system class."""
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG system."""
        self.config = config or RAGConfig.from_env()
        self.config.validate_config()
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.config)
        self.search_engine = SearchEngine(self.config)
        
        # Ensure directories exist
        Path(self.config.documents_dir).mkdir(exist_ok=True)
        Path(self.config.index_dir).mkdir(exist_ok=True)
        
        print(f"BabyCare RAG initialized with {len(self.list_documents())} documents")
    
    def add_document(self, file_path: str, doc_type: str = "auto") -> bool:
        """Add a document from file path."""
        try:
            success = self.document_processor.add_document_from_file(file_path)
            if success:
                # Rebuild search index to include new document
                self.search_engine.rebuild_index()
            return success
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def add_document_from_url(self, url: str) -> bool:
        """Add a document from URL."""
        try:
            success = self.document_processor.add_document_from_url(url)
            if success:
                # Rebuild search index to include new document
                self.search_engine.rebuild_index()
            return success
        except Exception as e:
            print(f"Error adding document from URL: {e}")
            return False
    
    def add_document_from_text(self, text: str, title: str) -> bool:
        """Add a document from text content."""
        try:
            success = self.document_processor.add_document_from_text(text, title)
            if success:
                # Rebuild search index to include new document
                self.search_engine.rebuild_index()
            return success
        except Exception as e:
            print(f"Error adding document from text: {e}")
            return False
    
    def list_documents(self) -> List[DocumentInfo]:
        """List all documents in the knowledge base."""
        return self.document_processor.list_documents()
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        try:
            success = self.document_processor.remove_document(doc_id)
            if success:
                # Rebuild search index after removal
                self.search_engine.rebuild_index()
            return success
        except Exception as e:
            print(f"Error removing document: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search documents and return relevant chunks with timing."""
        print(f"[DEBUG] Starting search for: {query[:50]}...")
        t_start = time.perf_counter()
        results = self.search_engine.search(query, top_k)
        t_end = time.perf_counter()
        print(f"[DEBUG] Search completed in {round((t_end - t_start) * 1000)}ms, found {len(results)} results")
        return results
    
    def query(self, question: str, max_steps: int = 5) -> RAGResponse:
        """Process a query using evidence-based template (consistent with Agent path)."""
        try:
            # Import evidence templates from agent
            import sys
            sys.path.append(str(Path(__file__).parent.parent))
            from agent import (EVIDENCE_BASED_ANSWER_SYSTEM_PROMPT, EVIDENCE_BASED_ANSWER_USER_PROMPT,
                             GENERAL_ANSWER_SYSTEM_PROMPT, GENERAL_ANSWER_USER_PROMPT)
            from openai import OpenAI

            # Initialize OpenAI client
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    try:
                        from babycare_rag.aws_secrets import get_openai_api_key_from_aws
                        api_key = get_openai_api_key_from_aws()
                    except Exception:
                        api_key = None
                openai_client = OpenAI(api_key=api_key) if api_key else OpenAI()
            except Exception:
                openai_client = OpenAI()

            # 1) Run retrieval first (single path for CLI/API)
            t0 = time.perf_counter()
            search_results = self.search_documents(question, self.config.top_k)
            t1 = time.perf_counter()
            sources = []
            for sr in search_results:
                src = (sr.source or "").strip()
                if src and src not in sources:
                    sources.append(src)

            # 2) Build evidence block from search results
            evidence_lines = []
            for sr in search_results[:10]:  # limit to top 10 for evidence
                src = sr.source or ''
                txt = sr.text or ''
                snippet = txt[:400] + ('...' if len(txt) > 400 else '')
                evidence_lines.append(f"[Source: {src}]\n{snippet}")
            evidence = "\n\n".join(evidence_lines) if evidence_lines else ""

            # Debug: print evidence content for troubleshooting
            print(f"[DEBUG] Evidence block length: {len(evidence)}")
            if evidence:
                print(f"[DEBUG] Evidence preview: {evidence[:500]}...")
            else:
                print("[DEBUG] No evidence found from search results")

            # 3) Choose appropriate template based on evidence availability
            if evidence.strip():
                # Use shared evidence-based prompt from agent module
                system_prompt = EVIDENCE_BASED_ANSWER_SYSTEM_PROMPT
                user_prompt = EVIDENCE_BASED_ANSWER_USER_PROMPT.format(query=question, evidence=evidence)
            else:
                # Use general knowledge template when no evidence is found
                system_prompt = GENERAL_ANSWER_SYSTEM_PROMPT
                user_prompt = GENERAL_ANSWER_USER_PROMPT.format(query=question)

            t2 = time.perf_counter()
            response = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=30  # Add timeout to prevent hanging
            )
            t3 = time.perf_counter()
            answer_text = (response.choices[0].message.content or "").strip()

            # 4) Ensure sources formatting if evidence existed
            if sources and "Sources:" not in answer_text:
                answer_text = f"{answer_text}\n\nSources: {', '.join(sources)}"

            # 5) Determine confidence based on evidence availability
            confidence = 0.8 if search_results else 0.6
            processing_steps = [
                "Analyzed user question",
                f"Retrieved documents in {round((t1 - t0) * 1000)} ms" if search_results else "No relevant documents found",
                f"LLM generation in {round((t3 - t2) * 1000)} ms",
                f"Total time {round((t3 - t0) * 1000)} ms"
            ]

            return RAGResponse(
                answer=answer_text,
                sources=sources,
                confidence=confidence,
                processing_steps=processing_steps,
                search_results=search_results
            )

        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            return RAGResponse(
                answer=f"Sorry, I encountered an error processing your question: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_steps=["Error occurred during processing"]
            )

    def update_config(self, config: RAGConfig) -> bool:
        """Update the system configuration."""
        try:
            config.validate_config()
            self.config = config
            
            # Reinitialize components with new config
            self.document_processor = DocumentProcessor(self.config)
            self.search_engine = SearchEngine(self.config)
            
            return True
        except Exception as e:
            print(f"Error updating config: {e}")
            return False
    
    def get_config(self) -> RAGConfig:
        """Get current system configuration."""
        return self.config
    
    def rebuild_index(self) -> bool:
        """Rebuild the search index."""
        return self.search_engine.rebuild_index()
    
    def get_stats(self) -> SystemStats:
        """Get system statistics."""
        try:
            documents = self.list_documents()
            # Prefer counting chunks from metadata if available to reflect actual index content
            total_chunks = 0
            try:
                import json
                meta_path = Path(self.config.index_dir) / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        md = json.load(f)
                        if isinstance(md, dict) and 'chunks' in md:
                            total_chunks = len(md['chunks'])
                        elif isinstance(md, list):
                            total_chunks = len(md)
                if total_chunks == 0:
                    total_chunks = sum(doc.chunk_count for doc in documents)
            except Exception:
                total_chunks = sum(doc.chunk_count for doc in documents)

            # Calculate storage used
            storage_used = 0
            index_dir = Path(self.config.index_dir)
            if index_dir.exists():
                for file_path in index_dir.rglob('*'):
                    if file_path.is_file():
                        storage_used += file_path.stat().st_size

            # Get index size
            index_file = index_dir / "index.bin"
            index_size = index_file.stat().st_size if index_file.exists() else 0

            return SystemStats(
                total_documents=len(documents),
                total_chunks=total_chunks,
                index_size=index_size,
                last_updated=datetime.now().isoformat(),
                storage_used=storage_used,
                embedding_model=self.config.embed_model,
                llm_model=self.config.llm_model
            )
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return SystemStats(
                total_documents=0,
                total_chunks=0,
                index_size=0,
                last_updated=datetime.now().isoformat(),
                storage_used=0,
                embedding_model=self.config.embed_model,
                llm_model=self.config.llm_model
            )
    
    # Convenience methods for API-style usage
    def process_request(self, request: QueryRequest) -> RAGResponse:
        """Process a query request."""
        return self.query(
            question=request.question,
            max_steps=request.max_steps or self.config.max_steps
        )
    
    def add_document_request(self, request: AddDocumentRequest) -> bool:
        """Process an add document request."""
        if request.file_path:
            return self.add_document(request.file_path, request.doc_type)
        elif request.url:
            return self.add_document_from_url(request.url)
        elif request.text_content and request.title:
            return self.add_document_from_text(request.text_content, request.title)
        else:
            raise ValueError("Must provide either file_path, url, or text_content with title")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the system."""
        try:
            stats = self.get_stats()
            
            # Test search functionality
            test_results = self.search_documents("baby care", top_k=1)
            search_working = len(test_results) > 0
            
            # Check if index exists
            index_file = Path(self.config.index_dir) / "index.bin"
            index_exists = index_file.exists()
            
            return {
                "status": "healthy" if search_working and index_exists else "degraded",
                "total_documents": stats.total_documents,
                "total_chunks": stats.total_chunks,
                "index_exists": index_exists,
                "search_working": search_working,
                "embedding_model": stats.embedding_model,
                "llm_model": stats.llm_model,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
