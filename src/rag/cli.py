"""
Command-Line Interface (CLI) for RAG application.
Provides commands for document ingestion, querying, interactive chat, and vector DB info.
"""

import argparse
import sys
from .config import RAGConfig
from .pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG LangChain Assistant - Ingest, Retrieve, and Query Documents"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF & Text documents into ChromaDB")
    ingest_parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing documents (default: ./data)"
    )
    ingest_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before ingestion"
    )
    ingest_parser.add_argument(
        "--pymupdf",
        action="store_true",
        help="Use PyMuPDFLoader instead of PyPDFLoader"
    )

    # Command: query
    query_parser = subparsers.add_parser("query", help="Query the RAG knowledge base")
    query_parser.add_argument(
        "question",
        type=str,
        help="The question to ask"
    )
    query_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)"
    )
    query_parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Similarity score threshold (default: 0.0)"
    )
    query_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved context text in output"
    )

    # Command: chat
    subparsers.add_parser("chat", help="Start an interactive chat session")

    # Command: info
    subparsers.add_parser("info", help="Display vector store and system information")

    return parser


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = RAGConfig()
    pipeline = RAGPipeline(config=config)

    if args.command == "ingest":
        pipeline.ingest(
            data_directory=args.data_dir,
            clear_existing=args.clear,
            use_pymupdf=args.pymupdf
        )

    elif args.command == "query":
        result = pipeline.query(
            question=args.question,
            top_k=args.top_k,
            score_threshold=args.threshold
        )
        print("\n" + "=" * 60)
        print(f"❓ Question: {result['question']}")
        print("=" * 60)
        print(f"\n💡 Answer:\n{result['answer']}")

        if result.get("sources"):
            print("\n📚 Sources referenced:")
            for s in result["sources"]:
                page_str = f", Page {s['page'] + 1}" if s.get("page") is not None else ""
                print(f"  - {s['source_file']}{page_str} (Score: {s['similarity_score']:.2f})")

        if args.show_context and result.get("context"):
            print("\n📄 Retrieved Context:")
            print(result["context"])
        print("\n" + "=" * 60)

    elif args.command == "chat":
        pipeline.interactive_chat()

    elif args.command == "info":
        count = pipeline.vector_store.count()
        print("\n" + "=" * 50)
        print("ℹ️  RAG Knowledge Base Status")
        print("=" * 50)
        print(f"  - ChromaDB Persist Directory : {config.chroma_persist_dir}")
        print(f"  - Collection Name            : {config.chroma_collection_name}")
        print(f"  - Total Document Chunks      : {count}")
        print(f"  - Embedding Model            : {config.embedding_model_name}")
        print(f"  - LLM Model (Groq)           : {config.groq_model}")
        print(f"  - Default Chunk Size/Overlap : {config.chunk_size} / {config.chunk_overlap}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
