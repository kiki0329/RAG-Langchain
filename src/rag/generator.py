"""
Generator module for RAG application.
Integrates Groq LLM with LangChain ChatPromptTemplate and LCEL for answer generation.
"""

from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


DEFAULT_RAG_SYSTEM_PROMPT = """You are a helpful, precise, and knowledgeable AI assistant.
Answer the user's question accurately and concisely using ONLY the provided context below.

Rules:
1. Base your answer strictly on the provided context. Do not invent information.
2. If the context does not contain enough information to answer the question, clearly state: "I don't have enough information in the provided documents to answer this question."
3. When helpful, cite the source file and page numbers mentioned in the context headers.
4. Keep the answer well-structured with bullet points or brief paragraphs where appropriate.

Context:
{context}
"""


class RAGGenerator:
    """Handles prompt construction and LLM text generation using LangChain and ChatGroq."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the RAG Generator.

        Args:
            api_key: Groq API Key.
            model_name: Groq model identifier.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens in completion.
            system_prompt: Custom system prompt template.
        """
        if not api_key:
            raise ValueError("GROQ_API_KEY is required to initialize RAGGenerator.")

        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 1. Initialize Groq Chat Model
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 2. Setup Prompt Template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt or DEFAULT_RAG_SYSTEM_PROMPT),
            ("human", "{question}")
        ])

        # 3. Build LCEL Chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer given a question and pre-formatted context.

        Args:
            question: The user's query string.
            context: Formatted context string from retrieved documents.

        Returns:
            Generated answer string.
        """
        if not context.strip():
            return "No relevant context was found in the documents to answer this question."

        try:
            response = self.chain.invoke({
                "context": context,
                "question": question
            })
            return response
        except Exception as e:
            return f"Error generating answer with LLM ({self.model_name}): {e}"

    def answer_with_citations(
        self,
        question: str,
        retrieved_docs: List[Dict[str, Any]],
        context_str: str
    ) -> Dict[str, Any]:
        """
        Generate answer and bundle it with detailed source citations.

        Args:
            question: User question.
            retrieved_docs: List of retrieved document dictionaries from RAGRetriever.
            context_str: Formatted context string.

        Returns:
            Dictionary with 'answer', 'sources', and 'chunks_used'.
        """
        answer = self.generate(question, context_str)

        sources = []
        for doc in retrieved_docs:
            meta = doc.get("metadata", {})
            sources.append({
                "source_file": meta.get("source_file", "Unknown"),
                "page": meta.get("page", None),
                "similarity_score": doc.get("similarity_score", 0.0),
                "rank": doc.get("rank", 1),
                "snippet": doc.get("content", "")[:200] + "..."
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources)
        }
