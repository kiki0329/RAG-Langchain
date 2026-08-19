"""
Document Loader module for RAG application.
Supports loading text files and PDF documents with metadata extraction.
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    PyPDFLoader,
    PyMuPDFLoader
)


class DocumentLoader:
    """Handles loading documents from various file formats and directories."""

    @staticmethod
    def load_text_file(file_path: str, encoding: str = "utf-8") -> List[Document]:
        """
        Load a single text file into LangChain Documents.

        Args:
            file_path: Path to the text file.
            encoding: Text encoding (default: utf-8).

        Returns:
            List of Document objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        loader = TextLoader(str(path), encoding=encoding)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = "txt"
        return docs

    @staticmethod
    def load_text_directory(
        directory_path: str,
        glob_pattern: str = "**/*.txt",
        encoding: str = "utf-8"
    ) -> List[Document]:
        """
        Load all text files from a directory.

        Args:
            directory_path: Directory containing text files.
            glob_pattern: Glob pattern to match files.
            encoding: Text encoding.

        Returns:
            List of Document objects.
        """
        dir_path = Path(directory_path)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        loader = DirectoryLoader(
            str(dir_path),
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": encoding},
            show_progress=False
        )
        docs = loader.load()
        for doc in docs:
            source_path = Path(doc.metadata.get("source", ""))
            doc.metadata["source_file"] = source_path.name
            doc.metadata["file_type"] = "txt"
        return docs

    @staticmethod
    def load_pdf_file(file_path: str, use_pymupdf: bool = False) -> List[Document]:
        """
        Load a single PDF file into LangChain Documents.

        Args:
            file_path: Path to the PDF file.
            use_pymupdf: If True, uses PyMuPDFLoader; otherwise PyPDFLoader.

        Returns:
            List of Document objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        loader_cls = PyMuPDFLoader if use_pymupdf else PyPDFLoader
        loader = loader_cls(str(path))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_type"] = "pdf"
        return docs

    @staticmethod
    def process_all_pdfs(
        pdf_directory: str,
        use_pymupdf: bool = False,
        verbose: bool = True
    ) -> List[Document]:
        """
        Process all PDF files in a directory recursively.

        Args:
            pdf_directory: Path to directory containing PDF files.
            use_pymupdf: If True, uses PyMuPDFLoader; otherwise PyPDFLoader.
            verbose: If True, prints loading progress.

        Returns:
            List of Document objects from all PDFs.
        """
        all_documents: List[Document] = []
        pdf_dir = Path(pdf_directory)

        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")

        pdf_files = list(pdf_dir.glob("**/*.pdf"))
        if verbose:
            print(f"Found {len(pdf_files)} PDF files to process in '{pdf_directory}'")

        loader_cls = PyMuPDFLoader if use_pymupdf else PyPDFLoader

        for pdf_file in pdf_files:
            if verbose:
                print(f"\nProcessing: {pdf_file.name}")
            try:
                loader = loader_cls(str(pdf_file))
                documents = loader.load()

                for doc in documents:
                    doc.metadata["source_file"] = pdf_file.name
                    doc.metadata["file_type"] = "pdf"

                all_documents.extend(documents)
                if verbose:
                    print(f"  ✓ Loaded successfully: {len(documents)} pages")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error loading {pdf_file.name}: {e}")

        if verbose:
            print(f"\nTotal PDF documents loaded: {len(all_documents)}")
        return all_documents

    @classmethod
    def load_all(
        cls,
        data_directory: str,
        use_pymupdf: bool = False,
        verbose: bool = True
    ) -> List[Document]:
        """
        Load all supported document types (PDF and TXT) from a directory.

        Args:
            data_directory: Root directory to search for documents.
            use_pymupdf: If True, uses PyMuPDFLoader for PDFs.
            verbose: If True, prints progress.

        Returns:
            Combined list of Document objects.
        """
        root = Path(data_directory)
        all_docs: List[Document] = []

        # 1. Load PDFs
        pdf_docs = cls.process_all_pdfs(str(root), use_pymupdf=use_pymupdf, verbose=verbose)
        all_docs.extend(pdf_docs)

        # 2. Load Text Files
        txt_files = list(root.glob("**/*.txt"))
        if verbose:
            print(f"\nFound {len(txt_files)} text files to process")

        for txt_file in txt_files:
            try:
                docs = cls.load_text_file(str(txt_file))
                all_docs.extend(docs)
                if verbose:
                    print(f"  ✓ Loaded text file: {txt_file.name} ({len(docs)} documents)")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error loading {txt_file.name}: {e}")

        if verbose:
            print(f"\nTotal documents loaded across all formats: {len(all_docs)}")
        return all_docs
