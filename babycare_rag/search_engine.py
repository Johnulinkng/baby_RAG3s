"""Search engine module for BabyCare RAG system."""

import json
import math
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import faiss
import numpy as np
import requests
from tqdm import tqdm

from .config import RAGConfig
from .models import SearchResult


class SearchEngine:
    """Hybrid search engine combining BM25 and vector search."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.index_dir = Path(config.index_dir)
        self.embedding_url = f"{config.ollama_base_url.rstrip('/')}/api/embeddings"
        self.embed_model = config.embed_model
        
        # Load synonyms for query expansion
        self.synonyms = self._load_synonyms()
        
        # Initialize search components
        self.faiss_index = None
        self.metadata = None
        self._load_index()
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonyms for query expansion."""
        synonyms_file = Path("babycare_synonyms.json")
        if synonyms_file.exists():
            try:
                with open(synonyms_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Default synonyms for baby care
        return {
            "baby": ["infant", "newborn", "child", "toddler"],
            "temperature": ["temp", "fever", "热度", "体温"],
            "feeding": ["nursing", "breastfeeding", "bottle", "milk"],
            "sleep": ["nap", "rest", "bedtime", "sleeping"],
            "crying": ["fussing", "upset", "distressed"],
            "diaper": ["nappy", "changing"],
            "safety": ["secure", "protection", "safe"]
        }
    
    def _load_index(self):
        """Load FAISS index and metadata."""
        try:
            index_file = self.index_dir / "index.bin"
            metadata_file = self.index_dir / "metadata.json"

            if index_file.exists() and metadata_file.exists():
                self.faiss_index = faiss.read_index(str(index_file))
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                    # Handle old format (array) vs new format (object)
                    if isinstance(existing_data, list):
                        # Convert old format to new format
                        self.metadata = {'documents': {}, 'chunks': existing_data}
                    else:
                        self.metadata = existing_data

                print(f"Loaded index with {len(self.metadata.get('chunks', []))} chunks")
            else:
                print("No existing index found. Will create new index when documents are added.")

        except Exception as e:
            print(f"Error loading index: {e}")
            self.faiss_index = None
            self.metadata = None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama."""
        try:
            response = requests.post(
                self.embedding_url,
                json={"model": self.embed_model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise
    
    def _expand_query_with_synonyms(self, query: str) -> str:
        """Expand query with synonyms."""
        expanded_terms = []
        words = re.findall(r'\b\w+\b', query.lower())
        
        for word in words:
            expanded_terms.append(word)
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        return " ".join(expanded_terms)
    
    def _bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform BM25 search on document chunks."""
        if not self.metadata or not self.metadata.get('chunks'):
            return []
        
        chunks = self.metadata['chunks']
        query_terms = re.findall(r'\b\w+\b', query.lower())
        
        if not query_terms:
            return []
        
        # Calculate document frequencies
        doc_freq = defaultdict(int)
        total_docs = len(chunks)
        
        for chunk in chunks:
            text = (chunk.get('text') or chunk.get('chunk') or '').lower()
            text_terms = set(re.findall(r'\b\w+\b', text))
            for term in query_terms:
                if term in text_terms:
                    doc_freq[term] += 1
        
        # Calculate BM25 scores
        k1, b = 1.5, 0.75
        scores = []
        
        for i, chunk in enumerate(chunks):
            text = (chunk.get('text') or chunk.get('chunk') or '').lower()
            text_terms = re.findall(r'\b\w+\b', text)
            doc_len = len(text_terms)

            if doc_len == 0:
                scores.append((i, 0.0))
                continue

            # Calculate average document length
            avg_doc_len = sum(len(re.findall(r'\b\w+\b', (c.get('text') or c.get('chunk') or '').lower())) for c in chunks) / total_docs
            
            score = 0.0
            for term in query_terms:
                tf = text_terms.count(term)
                if tf > 0:
                    idf = math.log((total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
                    score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
            
            scores.append((i, score))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _vector_search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Perform vector search using FAISS."""
        if not self.faiss_index or not self.metadata:
            return []
        
        try:
            query_embedding = self._get_embedding(query).reshape(1, -1)
            distances, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Convert distances to similarity scores (higher is better)
            scores = []
            for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
                if idx >= 0:  # Valid index
                    similarity = 1.0 / (1.0 + dist)  # Convert distance to similarity
                    scores.append((idx, similarity))
            
            return scores
            
        except Exception as e:
            print(f"Error in vector search: {e}")
            return []
    
    def _reciprocal_rank_fusion(self, bm25_results: List[Tuple[int, float]], 
                               vector_results: List[Tuple[int, float]], 
                               k: int = 60) -> List[Tuple[int, float]]:
        """Combine BM25 and vector search results using Reciprocal Rank Fusion."""
        
        # Create rank dictionaries
        bm25_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(bm25_results)}
        vector_ranks = {idx: rank + 1 for rank, (idx, _) in enumerate(vector_results)}
        
        # Get all unique indices
        all_indices = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = []
        for idx in all_indices:
            bm25_rank = bm25_ranks.get(idx, len(bm25_results) + 1)
            vector_rank = vector_ranks.get(idx, len(vector_results) + 1)
            
            rrf_score = (1.0 / (k + bm25_rank)) + (1.0 / (k + vector_rank))
            rrf_scores.append((idx, rrf_score))
        
        # Sort by RRF score
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        return rrf_scores
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Perform hybrid search combining BM25 and vector search."""
        if not self.metadata or not self.metadata.get('chunks'):
            return []
        
        try:
            # Expand query with synonyms
            expanded_query = self._expand_query_with_synonyms(query)
            
            # Perform BM25 search
            bm25_results = self._bm25_search(expanded_query, self.config.search_top_k)
            
            # Perform vector search
            vector_results = self._vector_search(query, self.config.search_top_k)
            
            # Combine results using RRF
            if bm25_results and vector_results:
                combined_results = self._reciprocal_rank_fusion(bm25_results, vector_results)
            elif bm25_results:
                combined_results = bm25_results
            elif vector_results:
                combined_results = vector_results
            else:
                return []
            
            # Convert to SearchResult objects
            search_results = []
            chunks = self.metadata['chunks']
            documents = self.metadata.get('documents', {})

            for idx, score in combined_results[:top_k]:
                if idx < len(chunks):
                    chunk = chunks[idx]

                    # Handle both old and new chunk formats
                    chunk_text = chunk.get('text') or chunk.get('chunk', '')
                    doc_id = chunk.get('doc_id', 'unknown')
                    source_name = chunk.get('doc', 'Unknown Document')

                    # Try to get better source name from documents metadata
                    if doc_id in documents:
                        doc_info = documents[doc_id]
                        source_name = doc_info.get('title', source_name)

                    search_results.append(SearchResult(
                        text=chunk_text,
                        source=source_name,
                        score=score,
                        chunk_id=chunk.get('id') or chunk.get('chunk_id'),
                        metadata={
                            'doc_id': doc_id,
                            'chunk_id': chunk.get('chunk_id'),
                            'file_path': documents.get(doc_id, {}).get('file_path')
                        }
                    ))
            
            return search_results
            
        except Exception as e:
            print(f"Error in search: {e}")
            return []
    
    def rebuild_index(self) -> bool:
        """Rebuild the FAISS index from scratch."""
        try:
            metadata_file = self.index_dir / "metadata.json"
            if not metadata_file.exists():
                print("No metadata file found. Nothing to rebuild.")
                return False
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            chunks = metadata.get('chunks', [])
            if not chunks:
                print("No chunks found. Nothing to rebuild.")
                return False
            
            print(f"Rebuilding index for {len(chunks)} chunks...")
            
            # Get embeddings for all chunks
            embeddings = []
            for chunk in tqdm(chunks, desc="Generating embeddings"):
                embedding = self._get_embedding(chunk['text'])
                embeddings.append(embedding)
            
            # Create FAISS index
            embeddings_array = np.vstack(embeddings)
            dimension = embeddings_array.shape[1]
            
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)
            
            # Save index
            index_file = self.index_dir / "index.bin"
            faiss.write_index(index, str(index_file))
            
            # Update in-memory index
            self.faiss_index = index
            self.metadata = metadata
            
            print(f"Successfully rebuilt index with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            print(f"Error rebuilding index: {e}")
            return False
