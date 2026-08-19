"""
Document Splitting module for RAG application.
Implements text chunking using LangChain's RecursiveCharacterTextSplitter.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitter:
    """Splits documents into smaller semantic chunks for vector embedding."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the DocumentSplitter.

        Args:
            chunk_size: Maximum number of characters in each chunk.
            chunk_overlap: Number of characters shared between consecutive chunks.
            separators: Optional custom separators list.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )

    def split_documents(
        self,
        documents: List[Document],
        verbose: bool = True
    ) -> List[Document]:
        """
        Split a list of Document objects into smaller chunks.

        Args:
            documents: Input documents to split.
            verbose: If True, prints chunking statistics.

        Returns:
            List of chunked Document objects with updated metadata.
        """
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)

        # Enrich chunk metadata with index and length
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_length"] = len(chunk.page_content)

        if verbose:
            print(f"Split {len(documents)} source documents into {len(chunks)} chunks.")
            if chunks:
                lengths = [len(c.page_content) for c in chunks]
                print(f"  Avg chunk size: {sum(lengths)//len(lengths)} characters")
                print(f"  Min chunk size: {min(lengths)} characters")
                print(f"  Max chunk size: {max(lengths)} characters")

        return chunks
