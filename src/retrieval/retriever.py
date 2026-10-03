from typing import Any

from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic_search import SemanticSearcher


class Retriever:
    """Retrieve relevant documents using semantic search and reranking."""

    def __init__(
        self,
        semantic_searcher: SemanticSearcher,
        reranker: CrossEncoderReranker,
        candidate_k: int = 10,
        top_k: int = 3,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if top_k > candidate_k:
            raise ValueError("top_k cannot be greater than candidate_k")

        self.semantic_searcher = semantic_searcher
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Retrieve and rerank documents for a query."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        semantic_results = self.semantic_searcher.search(
            query=query,
            top_k=self.candidate_k,
        )

        if not semantic_results:
            return []

        return self.reranker.rerank(
            query=query,
            results=semantic_results,
            top_k=self.top_k,
        )