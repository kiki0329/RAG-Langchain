"""
End-to-End RAG Pipeline Orchestrator.
Coordinates data ingestion, chunking, embedding, vector storage, retrieval, and generation.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import RAGConfig, default_config
from .loader import DocumentLoader
from .splitter import DocumentSplitter
from .embeddings import EmbeddingManager
from .vectorstore import VectorStore
from .retriever import RAGRetriever
from .generator import RAGGenerator


class RAGPipeline:
    """Complete modular RAG Pipeline."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize the RAG Pipeline with all underlying modules.

        Args:
            config: RAGConfig instance (defaults to global default_config).
        """
        self.config = config or default_config

        # 1. Initialize Embedding Manager
        self.embedding_manager = EmbeddingManager(
            model_name=self.config.embedding_model_name
        )

        # 2. Initialize Vector Store
        self.vector_store = VectorStore(
            persist_directory=self.config.chroma_persist_dir,
            collection_name=self.config.chroma_collection_name
        )

        # 3. Initialize Document Splitter
        self.splitter = DocumentSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        # 4. Initialize Retriever
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_manager=self.embedding_manager
        )

        # 5. Initialize Generator (LLM)
        self.generator: Optional[RAGGenerator] = None
        if self.config.groq_api_key:
            self.generator = RAGGenerator(
                api_key=self.config.groq_api_key,
                model_name=self.config.groq_model,
                temperature=self.config.groq_temperature,
                max_tokens=self.config.groq_max_tokens
            )
        else:
            print("Warning: GROQ_API_KEY not provided. Generation features will be disabled until key is set.")

    def ingest(
        self,
        data_directory: Optional[str] = None,
        clear_existing: bool = False,
        use_pymupdf: bool = False
    ) -> int:
        """
        Run the complete data ingestion pipeline:
        Load files -> Split into chunks -> Generate embeddings -> Store in ChromaDB.

        Args:
            data_directory: Directory containing PDFs/Text files.
            clear_existing: If True, clears existing vector DB collection before adding.
            use_pymupdf: If True, uses PyMuPDFLoader for PDFs.

        Returns:
            Number of chunks successfully indexed.
        """
        target_dir = data_directory or str(self.config.data_dir)
        print(f"\n{'='*50}\nStarting Data Ingestion Pipeline from: {target_dir}\n{'='*50}")

        if clear_existing:
            self.vector_store.clear()

        # Step 1: Load documents
        raw_documents = DocumentLoader.load_all(
            data_directory=target_dir,
            use_pymupdf=use_pymupdf,
            verbose=True
        )

        if not raw_documents:
            print("No documents found to ingest.")
            return 0

        # Step 2: Split documents into semantic chunks
        chunks = self.splitter.split_documents(raw_documents, verbose=True)

        # Step 3: Embed and add chunks to Vector Store
        print("\nIndexing chunks into ChromaDB...")
        self.vector_store.add_chunks_with_embeddings(
            chunks=chunks,
            embedding_manager=self.embedding_manager
        )

        print(f"\n✓ Ingestion complete! Total items in vector store: {self.vector_store.count()}\n{'='*50}")
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute full RAG query:
        Retrieve relevant chunks -> Format context -> Generate answer with citations.

        Args:
            question: The user's question string.
            top_k: Number of chunks to retrieve (defaults to config.top_k).
            score_threshold: Minimum similarity score (defaults to config.score_threshold).

        Returns:
            Dictionary containing 'question', 'answer', 'sources', and 'context'.
        """
        k = top_k if top_k is not None else self.config.top_k
        threshold = score_threshold if score_threshold is not None else self.config.score_threshold

        # Step 1: Retrieve matching documents
        retrieved_docs = self.retriever.retrieve(
            query=question,
            top_k=k,
            score_threshold=threshold,
            verbose=True
        )

        # Step 2: Format context
        context_str = self.retriever.format_context(retrieved_docs)

        # Step 3: Generate Answer
        if not self.generator:
            if not self.config.groq_api_key:
                raise ValueError("Cannot generate answer: GROQ_API_KEY is not configured.")
            self.generator = RAGGenerator(
                api_key=self.config.groq_api_key,
                model_name=self.config.groq_model,
                temperature=self.config.groq_temperature,
                max_tokens=self.config.groq_max_tokens
            )

        result = self.generator.answer_with_citations(
            question=question,
            retrieved_docs=retrieved_docs,
            context_str=context_str
        )
        result["context"] = context_str
        return result

    def ask(self, question: str) -> str:
        """
        Convenience method to return only the generated answer text.

        Args:
            question: Question string.

        Returns:
            Answer string.
        """
        result = self.query(question)
        return result["answer"]

    def interactive_chat(self) -> None:
        """Launch an interactive terminal Q&A session with the RAG pipeline."""
        print("\n" + "=" * 60)
        print("🤖 RAG Interactive Chat Assistant (Powered by LangChain & Groq)")
        print("Type your questions below. Type 'exit', 'quit', or 'q' to end.")
        print("=" * 60 + "\n")

        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("Goodbye! 👋")
                    break

                response = self.query(user_input)
                print(f"\n🤖 Assistant:\n{response['answer']}")

                if response.get("sources"):
                    print("\n📚 Sources referenced:")
                    for s in response["sources"]:
                        page_str = f", Page {s['page'] + 1}" if s.get("page") is not None else ""
                        print(f"  - {s['source_file']}{page_str} (Score: {s['similarity_score']:.2f})")

            except KeyboardInterrupt:
                print("\nSession ended.")
                break
            except Exception as e:
                print(f"\nError: {e}")
