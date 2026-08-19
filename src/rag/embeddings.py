"""
Embedding Manager module for RAG application.
Encapsulates SentenceTransformer and LangChain HuggingFace embeddings generation.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingManager:
    """Manages text embedding generation using SentenceTransformer / HuggingFace models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the EmbeddingManager.

        Args:
            model_name: HuggingFace model identifier.
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self._langchain_embeddings: Optional[HuggingFaceEmbeddings] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the underlying SentenceTransformer model."""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            dim = self.model.get_embedding_dimension()
            print(f"Model loaded successfully. Embedding dimension: {dim}")
        except Exception as e:
            print(f"Error loading embedding model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate dense vector embeddings for a list of strings.

        Args:
            texts: List of text strings to embed.
            show_progress: Whether to show progress bar during inference.

        Returns:
            Numpy array of shape (len(texts), embedding_dim).
        """
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        if not texts:
            return np.empty((0, self.get_dimension()))

        print(f"Generating embeddings for {len(texts)} text chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=show_progress)
        print(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def get_dimension(self) -> int:
        """Return the vector dimensionality of the embedding model."""
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        return self.model.get_embedding_dimension()

    def get_langchain_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Return a LangChain-compatible HuggingFaceEmbeddings instance.

        Returns:
            HuggingFaceEmbeddings instance.
        """
        if self._langchain_embeddings is None:
            self._langchain_embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name
            )
        return self._langchain_embeddings
