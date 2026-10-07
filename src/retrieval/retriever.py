from typing import Any

from src.retrieval.reranker import MetadataReranker
from src.retrieval.semantic_search import SemanticSearcher


class Retriever:
    """Retrieve documents using semantic and hierarchical retrieval."""

    def __init__(
        self,
        semantic_searcher: SemanticSearcher,
        reranker: MetadataReranker | None = None,
        candidate_k: int = 20,
        top_k: int = 3,
        use_reranker: bool = True,
        use_hierarchy: bool = True,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

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
        self.use_hierarchy = use_hierarchy

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        semantic_results = self.semantic_searcher.search(
            query=query,
            top_k=self.candidate_k,
        )

        if not semantic_results:
            return []

        if self.use_hierarchy:
            semantic_results = self._expand_hierarchy(
                semantic_results
            )

        if not self.use_reranker:
            return semantic_results[: self.top_k]

        return self.reranker.rerank(
            query=query,
            results=semantic_results,
            top_k=self.top_k,
        )

    def _expand_hierarchy(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        expanded = []

        seen_ids = set()

        for result in results:
            result = result.copy()
            result["hierarchy_expanded"] = False

            result_id = result.get("id")

            if result_id not in seen_ids:
                expanded.append(result)
                seen_ids.add(result_id)

            heading_path = result.get("heading_path", "")

            if not heading_path:
                continue

            related_chunks = (
                self.semantic_searcher.repository
                .get_chunks_by_heading_path(heading_path)
            )

            for chunk in related_chunks:
                chunk_id = chunk.get("id")

                if chunk_id in seen_ids:
                    continue

                chunk["hierarchy_expanded"] = True
                chunk["distance"] = result["distance"]
                expanded.append(chunk)
                seen_ids.add(chunk_id)

        return expanded