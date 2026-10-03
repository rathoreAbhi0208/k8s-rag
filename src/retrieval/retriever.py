from typing import Any

from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.semantic_search import SemanticSearcher


class Retriever:
    """Retrieve relevant documents using semantic search and optional reranking."""

    def __init__(
        self,
        semantic_searcher: SemanticSearcher,
        reranker: CrossEncoderReranker | None = None,
        candidate_k: int = 10,
        top_k: int = 3,
        use_reranker: bool = True,
    ):
        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if top_k > candidate_k:
            raise ValueError(
                "top_k cannot be greater than candidate_k"
            )

        if use_reranker and reranker is None:
            raise ValueError(
                "reranker is required when use_reranker=True"
            )

        self.semantic_searcher = semantic_searcher
        self.reranker = reranker
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.use_reranker = use_reranker

    def retrieve(
        self,
        query: str,
    ) -> list[dict[str, Any]]:
        """Retrieve documents for a query."""

        if not query.strip():
            raise ValueError(
                "Query cannot be empty"
            )

        # ---------------------------------------------------------
        # Semantic retrieval
        # ---------------------------------------------------------

        semantic_results = self.semantic_searcher.search(
            query=query,
            top_k=self.candidate_k,
        )

        if not semantic_results:
            return []

        # ---------------------------------------------------------
        # Semantic-only mode
        # ---------------------------------------------------------

        if not self.use_reranker:
            return semantic_results[: self.top_k]

        # ---------------------------------------------------------
        # Reranking mode
        # ---------------------------------------------------------

        return self.reranker.rerank(
            query=query,
            results=semantic_results,
            top_k=self.top_k,
        )