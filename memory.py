# memory.py

import numpy as np
import faiss
import requests
from typing import List, Optional, Literal
from pydantic import BaseModel
from datetime import datetime
import os


class MemoryItem(BaseModel):
    text: str
    type: Literal["preference", "tool_output", "fact", "query", "system"] = "fact"
    timestamp: Optional[str] = datetime.now().isoformat()
    tool_name: Optional[str] = None
    user_query: Optional[str] = None
    tags: List[str] = []
    session_id: Optional[str] = None


class MemoryManager:
    def __init__(self, embedding_model_url=None, model_name=None):
        # Use OpenAI embeddings instead of Ollama
        from openai import OpenAI
        self.openai_client = OpenAI()
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.index = None
        self.data: List[MemoryItem] = []
        self.embeddings: List[np.ndarray] = []
        # In-memory LRU cache for embeddings to reduce latency and cost
        # Key design: f"{self.embed_model}:{hash(text)}" to avoid conflicts across models
        from functools import lru_cache
        @lru_cache(maxsize=int(os.getenv("MEMORY_EMBED_CACHE_SIZE", "2048")))
        def _embed_cache(key: str) -> np.ndarray:  # key = model:text
            model, txt = key.split("::", 1)
            resp = self.openai_client.embeddings.create(model=model, input=txt)
            return np.array(resp.data[0].embedding, dtype=np.float32)
        self._embed_cache = _embed_cache

    def _get_embedding(self, text: str) -> np.ndarray:
        # Stable cache key: model + '::' + text
        key = f"{self.embed_model}::{text}"
        try:
            return self._embed_cache(key)
        except Exception as e:
            # Fallback with timeout and retry for network issues
            print(f"Cache failed, using direct embedding: {e}")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = self.openai_client.embeddings.create(
                        model=self.embed_model,
                        input=text,
                        timeout=30
                    )
                    return np.array(resp.data[0].embedding, dtype=np.float32)
                except Exception as retry_e:
                    if attempt == max_retries - 1:
                        raise retry_e
                    print(f"Memory embedding attempt {attempt + 1} failed: {retry_e}, retrying...")
                    import time
                    time.sleep(1)

    def add(self, item: MemoryItem):
        try:
            emb = self._get_embedding(item.text)
            if emb is None or len(emb) == 0:
                print(f"Warning: Invalid embedding for text: {item.text[:50]}...")
                return

            self.embeddings.append(emb)
            self.data.append(item)

            # Initialize or add to index with better error handling
            if self.index is None:
                try:
                    self.index = faiss.IndexFlatL2(len(emb))
                    if self.index is None:
                        raise ValueError("Failed to create FAISS index")
                except Exception as e:
                    print(f"Error creating FAISS index: {e}")
                    return

            try:
                self.index.add(np.stack([emb]))
            except Exception as e:
                print(f"Error adding to FAISS index: {e}")
                # Reset index if it becomes corrupted
                self.index = None
        except Exception as e:
            print(f"Error in MemoryManager.add: {e}")

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        type_filter: Optional[str] = None,
        tag_filter: Optional[List[str]] = None,
        session_filter: Optional[str] = None
    ) -> List[MemoryItem]:
        if not self.index or len(self.data) == 0:
            return []

        query_vec = self._get_embedding(query).reshape(1, -1)
        D, I = self.index.search(query_vec, top_k * 2)  # Overfetch to allow filtering

        results = []
        for idx in I[0]:
            if idx >= len(self.data):
                continue
            item = self.data[idx]

            # Filter by type
            if type_filter and item.type != type_filter:
                continue

            # Filter by tags
            if tag_filter and not any(tag in item.tags for tag in tag_filter):
                continue

            # Filter by session
            if session_filter and item.session_id != session_filter:
                continue

            results.append(item)
            if len(results) >= top_k:
                break

        return results

    def bulk_add(self, items: List[MemoryItem]):
        for item in items:
            self.add(item)
