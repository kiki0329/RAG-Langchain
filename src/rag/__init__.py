"""
RAG LangChain Package.
Complete, modular Retrieval Augmented Generation system.
"""

from .config import RAGConfig, default_config
from .loader import DocumentLoader
from .splitter import DocumentSplitter
from .embeddings import EmbeddingManager
from .vectorstore import VectorStore
from .retriever import RAGRetriever
from .generator import RAGGenerator
from .pipeline import RAGPipeline
from .cli import main

__all__ = [
    "RAGConfig",
    "default_config",
    "DocumentLoader",
    "DocumentSplitter",
    "EmbeddingManager",
    "VectorStore",
    "RAGRetriever",
    "RAGGenerator",
    "RAGPipeline",
    "main"
]
