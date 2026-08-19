"""
Vector Store module for RAG application.
Manages persistent ChromaDB vector storage and indexing.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document

from .embeddings import EmbeddingManager


class VectorStore:
    """Persistent Vector Store backed by ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/data/vector_store",
        collection_name: str = "pdf_documents"
    ):
        """
        Initialize the ChromaDB persistent client and collection.

        Args:
            persist_directory: Directory path to persist ChromaDB files.
            collection_name: Name of the vector collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self) -> None:
        """Initialize ChromaDB PersistentClient and get/create collection."""
        try:
            persist_path = Path(self.persist_directory)
            persist_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=str(persist_path),
                settings=Settings(allow_reset=True)
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            print(f"Vector store initialized at '{self.persist_directory}'. Collection: '{self.collection_name}'")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(
        self,
        documents: List[Document],
        embeddings: np.ndarray,
        batch_size: int = 200
    ) -> None:
        """
        Add documents and their pre-computed embeddings to ChromaDB.

        Args:
            documents: List of Document objects.
            embeddings: Numpy array of corresponding embeddings.
            batch_size: Batch size for ChromaDB insertions.
        """
        if not documents:
            print("No documents to add.")
            return

        if len(documents) != len(embeddings):
            raise ValueError(
                f"Count mismatch: {len(documents)} documents vs {len(embeddings)} embeddings."
            )

        ids: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        documents_text: List[str] = []
        embeddings_list: List[List[float]] = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Sanitize metadata for ChromaDB (Chroma allows str, int, float, bool)
            metadata: Dict[str, Any] = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k] = v
                elif v is not None:
                    metadata[k] = str(v)

            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        try:
            total = len(ids)
            for start_idx in range(0, total, batch_size):
                end_idx = min(start_idx + batch_size, total)
                self.collection.add(
                    ids=ids[start_idx:end_idx],
                    embeddings=embeddings_list[start_idx:end_idx],
                    metadatas=metadatas[start_idx:end_idx],
                    documents=documents_text[start_idx:end_idx]
                )

            print(f"Successfully added {len(documents)} documents to vector store.")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def add_chunks_with_embeddings(
        self,
        chunks: List[Document],
        embedding_manager: EmbeddingManager
    ) -> None:
        """
        Compute embeddings and add chunks to the vector store in one step.

        Args:
            chunks: List of chunked Document objects.
            embedding_manager: EmbeddingManager instance.
        """
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        self.add_documents(chunks, embeddings)

    def count(self) -> int:
        """Return the number of items in the collection."""
        return self.collection.count() if self.collection else 0

    def clear(self) -> None:
        """Clear all documents from the collection."""
        if self.client and self.collection:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Collection '{self.collection_name}' has been cleared.")
