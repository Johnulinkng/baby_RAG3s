"""Data models for BabyCare RAG system."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentInfo(BaseModel):
    """Information about a document in the knowledge base."""
    
    doc_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title")
    file_path: str = Field(description="Path to the document file")
    added_date: str = Field(description="Date when document was added")
    chunk_count: int = Field(description="Number of chunks in the document")
    file_size: Optional[int] = Field(default=None, description="File size in bytes")
    doc_type: Optional[str] = Field(default=None, description="Document type (pdf, docx, txt, etc.)")


class SearchResult(BaseModel):
    """Result from document search."""
    
    text: str = Field(description="Retrieved text content")
    source: str = Field(description="Source document name")
    score: float = Field(description="Relevance score")
    chunk_id: Optional[str] = Field(default=None, description="Chunk identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class RAGResponse(BaseModel):
    """Response from RAG query."""
    
    answer: str = Field(description="Generated answer")
    sources: List[str] = Field(description="Source documents used")
    confidence: float = Field(description="Confidence score (0-1)")
    processing_steps: List[str] = Field(description="Steps taken during processing")
    search_results: Optional[List[SearchResult]] = Field(default=None, description="Raw search results")
    reasoning_chain: Optional[List[str]] = Field(default=None, description="Reasoning steps")
    tool_calls: Optional[List[str]] = Field(default=None, description="Tools called during processing")


class SystemStats(BaseModel):
    """System statistics and health information."""
    
    total_documents: int = Field(description="Total number of documents")
    total_chunks: int = Field(description="Total number of chunks")
    index_size: int = Field(description="FAISS index size")
    last_updated: str = Field(description="Last index update time")
    storage_used: int = Field(description="Storage used in bytes")
    embedding_model: str = Field(description="Current embedding model")
    llm_model: str = Field(description="Current LLM model")


class MemoryItem(BaseModel):
    """Memory item for the RAG system."""
    
    text: str = Field(description="Memory content")
    type: Literal["preference", "tool_output", "fact", "query", "system"] = Field(
        default="fact", description="Type of memory item"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp when memory was created"
    )
    tool_name: Optional[str] = Field(default=None, description="Tool that generated this memory")
    user_query: Optional[str] = Field(default=None, description="User query that led to this memory")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class PerceptionResult(BaseModel):
    """Result from perception/intent analysis."""
    
    user_input: str = Field(description="Original user input")
    intent: Optional[str] = Field(default=None, description="Detected intent")
    entities: List[str] = Field(default_factory=list, description="Extracted entities")
    tool_hint: Optional[str] = Field(default=None, description="Suggested tool to use")
    confidence: float = Field(default=0.0, description="Confidence in intent detection")


class ToolCallResult(BaseModel):
    """Result from tool execution."""
    
    tool_name: str = Field(description="Name of the executed tool")
    arguments: Dict[str, Any] = Field(description="Arguments passed to the tool")
    result: Any = Field(description="Tool execution result")
    raw_response: Any = Field(description="Raw response from tool")
    sources: List[str] = Field(default_factory=list, description="Source documents if applicable")
    execution_time: Optional[float] = Field(default=None, description="Execution time in seconds")
    success: bool = Field(default=True, description="Whether execution was successful")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class QueryRequest(BaseModel):
    """Request for RAG query."""
    
    question: str = Field(description="User question")
    max_steps: Optional[int] = Field(default=5, description="Maximum reasoning steps")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    include_sources: bool = Field(default=True, description="Whether to include source information")
    include_reasoning: bool = Field(default=False, description="Whether to include reasoning chain")


class AddDocumentRequest(BaseModel):
    """Request to add a document."""
    
    file_path: Optional[str] = Field(default=None, description="Path to document file")
    url: Optional[str] = Field(default=None, description="URL to fetch document from")
    text_content: Optional[str] = Field(default=None, description="Direct text content")
    title: Optional[str] = Field(default=None, description="Document title")
    doc_type: str = Field(default="auto", description="Document type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
