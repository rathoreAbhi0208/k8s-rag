from typing import Any

from src.generation.generator import Generator
from src.retrieval.retriever import Retriever


class RAGPipeline:
    """End-to-end retrieval-augmented generation pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
    ):
        self.retriever = retriever
        self.generator = generator

    def ask(self, query: str) -> dict[str, Any]:
        """Retrieve context and generate an answer."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        results = self.retriever.retrieve(query)

        answer = self.generator.generate(
            query=query,
            results=results,
        )

        return {
            "query": query,
            "answer": answer,
            "sources": results,
        }