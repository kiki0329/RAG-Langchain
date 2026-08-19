"""
Configuration module for RAG application.
Loads settings from environment variables and provides structured configuration.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()


@dataclass
class RAGConfig:
    """RAG configuration parameters."""
    
    # Project paths
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    data_dir: Path = project_root / "data"
    pdf_dir: Path = data_dir / "pdf"
    text_dir: Path = data_dir / "text_files"
    
    # Groq LLM settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    groq_temperature: float = float(os.getenv("GROQ_TEMPERATURE", "0.1"))
    groq_max_tokens: int = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
    
    # Embedding settings
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # ChromaDB Vector Store settings
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(project_root / "data" / "data" / "vector_store"))
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents")
    
    # Chunking settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    
    # Retrieval settings
    top_k: int = int(os.getenv("TOP_K", "5"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.0"))


# Default global configuration instance
default_config = RAGConfig()
