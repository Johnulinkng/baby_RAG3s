"""
内存优化建议和实现
"""
import gc
import weakref
from functools import lru_cache
from typing import Dict, Any
import threading
import time

class MemoryOptimizedRAG:
    """内存优化的 RAG 实现"""
    
    def __init__(self):
        self._session_cache = weakref.WeakValueDictionary()
        self._query_cache = {}
        self._cache_lock = threading.RLock()
        self._last_cleanup = time.time()
    
    @lru_cache(maxsize=100)  # 限制缓存大小
    def get_embedding_cached(self, text: str):
        """缓存嵌入向量，避免重复计算"""
        from math_mcp_embeddings import get_embedding
        return get_embedding(text)
    
    def cleanup_memory(self):
        """定期内存清理"""
        with self._cache_lock:
            # 清理过期查询缓存
            current_time = time.time()
            if current_time - self._last_cleanup > 300:  # 5分钟清理一次
                self._query_cache.clear()
                self.get_embedding_cached.cache_clear()
                gc.collect()
                self._last_cleanup = current_time
    
    def process_query_with_cleanup(self, query: str):
        """带内存清理的查询处理"""
        try:
            # 处理查询
            result = self._process_query(query)
            return result
        finally:
            # 确保清理临时对象
            self.cleanup_memory()

# 服务器部署配置
SERVER_CONFIG = {
    "memory_limits": {
        "max_workers": 50,  # 限制最大并发
        "max_memory_per_worker": "100MB",
        "total_memory_limit": "4GB"
    },
    "optimization": {
        "enable_query_cache": True,
        "cache_ttl": 300,  # 5分钟缓存
        "gc_interval": 60,  # 1分钟垃圾回收
        "session_timeout": 1800  # 30分钟会话超时
    }
}

def monitor_memory_usage():
    """内存监控函数"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    if memory_mb > 2048:  # 超过 2GB 警告
        print(f" 内存使用过高: {memory_mb:.2f} MB")
        gc.collect()  # 强制垃圾回收
    
    return memory_mb
