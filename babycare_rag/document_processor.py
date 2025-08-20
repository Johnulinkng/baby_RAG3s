"""Document processing module for BabyCare RAG system."""

import os
import json
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from markitdown import MarkItDown
from tqdm import tqdm
import faiss
import numpy as np

from .config import RAGConfig
from .models import DocumentInfo


class DocumentProcessor:
    """Handles document processing, chunking, and indexing."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.md = MarkItDown()
        self.documents_dir = Path(config.documents_dir)
        self.index_dir = Path(config.index_dir)
        
        # Ensure directories exist
        self.documents_dir.mkdir(exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)
    
    def add_document_from_file(self, file_path: str, title: Optional[str] = None) -> bool:
        """Add a document from a file path."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Copy file to documents directory if not already there
            if not file_path.is_relative_to(self.documents_dir):
                dest_path = self.documents_dir / file_path.name
                import shutil
                shutil.copy2(file_path, dest_path)
                file_path = dest_path
            
            # Process the document
            return self._process_single_document(file_path, title)
            
        except Exception as e:
            print(f"Error adding document from file: {e}")
            return False
    
    def add_document_from_url(self, url: str, title: Optional[str] = None) -> bool:
        """Add a document from a URL."""
        try:
            # Download the content
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Generate filename from URL
            from urllib.parse import urlparse
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                filename = f"webpage_{hashlib.md5(url.encode()).hexdigest()[:8]}.html"
            
            # Save to documents directory
            file_path = self.documents_dir / filename
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Process the document
            return self._process_single_document(file_path, title or url)
            
        except Exception as e:
            print(f"Error adding document from URL: {e}")
            return False
    
    def add_document_from_text(self, text: str, title: str) -> bool:
        """Add a document from text content."""
        try:
            # Create a text file
            filename = f"{hashlib.md5(title.encode()).hexdigest()[:8]}_{title[:50]}.txt"
            # Clean filename
            filename = "".join(c for c in filename if c.isalnum() or c in "._- ").strip()
            file_path = self.documents_dir / filename
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Process the document
            return self._process_single_document(file_path, title)
            
        except Exception as e:
            print(f"Error adding document from text: {e}")
            return False
    
    def _process_single_document(self, file_path: Path, title: Optional[str] = None) -> bool:
        """Process a single document and add to index."""
        try:
            # Convert document to markdown
            result = self.md.convert(str(file_path))
            content = result.text_content
            
            if not content.strip():
                print(f"Warning: No content extracted from {file_path}")
                return False
            
            # Create document info
            doc_info = DocumentInfo(
                doc_id=hashlib.md5(str(file_path).encode()).hexdigest(),
                title=title or file_path.stem,
                file_path=str(file_path),
                added_date=str(file_path.stat().st_mtime),
                chunk_count=0,  # Will be updated after chunking
                file_size=file_path.stat().st_size,
                doc_type=file_path.suffix.lower()
            )
            
            # Chunk the document
            chunks = self._chunk_document(content, doc_info.doc_id)
            doc_info.chunk_count = len(chunks)
            
            # Update metadata
            self._update_metadata(doc_info, chunks)
            
            print(f"Successfully processed document: {doc_info.title} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            print(f"Error processing document {file_path}: {e}")
            return False
    
    def _chunk_document(self, content: str, doc_id: str) -> List[Dict[str, Any]]:
        """Chunk a document into smaller pieces."""
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        # Simple chunking by character count
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = -1
                for i in range(end, search_start, -1):
                    if content[i] in '.!?':
                        sentence_end = i + 1
                        break
                
                if sentence_end > 0:
                    end = sentence_end
            
            chunk_text = content[start:end].strip()
            
            if chunk_text:
                chunks.append({
                    'id': f"{doc_id}_{chunk_id}",
                    'doc_id': doc_id,
                    'chunk_id': chunk_id,
                    'text': chunk_text,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - overlap if end < len(content) else end
        
        return chunks
    
    def _update_metadata(self, doc_info: DocumentInfo, chunks: List[Dict[str, Any]]):
        """Update metadata file with new document and chunks."""
        metadata_file = self.index_dir / "metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {'documents': {}, 'chunks': []}
        
        # Add document info
        metadata['documents'][doc_info.doc_id] = doc_info.model_dump()
        
        # Add chunks
        metadata['chunks'].extend(chunks)
        
        # Save updated metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def list_documents(self) -> List[DocumentInfo]:
        """List all documents in the knowledge base."""
        metadata_file = self.index_dir / "metadata.json"
        
        if not metadata_file.exists():
            return []
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            documents = []
            for doc_data in metadata.get('documents', {}).values():
                documents.append(DocumentInfo(**doc_data))
            
            return documents
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the knowledge base."""
        try:
            metadata_file = self.index_dir / "metadata.json"
            
            if not metadata_file.exists():
                return False
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Remove document
            if doc_id not in metadata.get('documents', {}):
                return False
            
            doc_info = metadata['documents'][doc_id]
            del metadata['documents'][doc_id]
            
            # Remove chunks
            metadata['chunks'] = [
                chunk for chunk in metadata['chunks'] 
                if chunk.get('doc_id') != doc_id
            ]
            
            # Save updated metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Remove file if it exists
            try:
                file_path = Path(doc_info['file_path'])
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass  # File removal is not critical
            
            print(f"Successfully removed document: {doc_id}")
            return True
            
        except Exception as e:
            print(f"Error removing document: {e}")
            return False
