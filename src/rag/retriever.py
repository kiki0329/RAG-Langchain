"""
Retriever module for RAG application.
Handles similarity search, score filtering, and context construction.
"""

from typing import List, Dict, Any, Optional
from .vectorstore import VectorStore
from .embeddings import EmbeddingManager


class RAGRetriever:
    """Handles query-based retrieval from the vector store with similarity scoring."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: Vector store containing document embeddings.
            embedding_manager: Manager for generating query embeddings.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: The search query string.
            top_k: Number of top results to return.
            score_threshold: Minimum similarity score threshold.
            filter_metadata: Optional ChromaDB metadata filter dictionary.
            verbose: If True, prints retrieval progress.

        Returns:
            List of dictionaries containing retrieved documents, scores, and metadata.
        """
        if verbose:
            print(f"\nRetrieving documents for query: '{query}'")
            print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # 1. Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings(
            [query], show_progress=False
        )[0]

        # 2. Search in ChromaDB
        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding.tolist()],
                "n_results": top_k
            }
            if filter_metadata:
                query_kwargs["where"] = filter_metadata

            results = self.vector_store.collection.query(**query_kwargs)

            # 3. Process & Rank results
            retrieved_docs: List[Dict[str, Any]] = []

            if results.get("documents") and results["documents"][0]:
                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # ChromaDB cosine distance: similarity = 1 - distance
                    similarity_score = 1.0 - float(distance)

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": round(similarity_score, 4),
                            "distance": round(float(distance), 4),
                            "rank": len(retrieved_docs) + 1
                        })

                if verbose:
                    print(f"Retrieved {len(retrieved_docs)} matching chunks (after filtering).")
            else:
                if verbose:
                    print("No documents found in vector store.")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context block with source annotations.

        Args:
            retrieved_docs: List of retrieved document dictionaries.

        Returns:
            Formatted string context for LLM prompt.
        """
        if not retrieved_docs:
            return ""

        context_blocks = []
        for doc in retrieved_docs:
            source = doc["metadata"].get("source_file", "Unknown Source")
            page = doc["metadata"].get("page", None)
            page_info = f" (Page {int(page) + 1})" if page is not None else ""
            score = doc.get("similarity_score", 0.0)

            header = f"[Source: {source}{page_info} | Relevance: {score:.2f}]"
            context_blocks.append(f"{header}\n{doc['content']}")

        return "\n\n---\n\n".join(context_blocks)
