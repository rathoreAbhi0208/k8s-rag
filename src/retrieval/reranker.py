import re
from typing import Any


class MetadataReranker:
    """Rerank semantic search results using query/heading/content signals."""

    def __init__(
        self,
        semantic_weight: float = 0.5,
        heading_weight: float = 0.3,
        path_weight: float = 0.2,
    ):
        total = semantic_weight + heading_weight + path_weight

        if total <= 0:
            raise ValueError("Reranker weights must sum to a positive value")

        self.semantic_weight = semantic_weight
        self.heading_weight = heading_weight
        self.path_weight = path_weight

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not results:
            return []

        query_terms = self._tokenize(query)

        reranked = []

        for result in results:
            semantic_score = float(result.get("distance", 0.0))

            heading = result.get("heading", "")
            heading_path = result.get("heading_path", "")
            content = result.get("content", "")

            heading_score = self._term_overlap(
                query_terms,
                heading,
            )

            path_score = self._term_overlap(
                query_terms,
                heading_path,
            )

            content_score = self._term_overlap(
                query_terms,
                content,
            )

            final_score = (
                self.semantic_weight * semantic_score
                + self.heading_weight * heading_score
                + self.path_weight * path_score
            )
            
            if result.get("hierarchy_expanded"):
                final_score += 0.15

            reranked_result = result.copy()

            reranked_result["semantic_score"] = semantic_score
            reranked_result["heading_score"] = heading_score
            reranked_result["path_score"] = path_score
            reranked_result["content_score"] = content_score
            reranked_result["rerank_score"] = final_score

            reranked.append(reranked_result)

        reranked.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        return reranked[:top_k]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\b[a-zA-Z0-9-]+\b", text.lower())
            if len(token) > 2
        }

    @staticmethod
    def _term_overlap(
        query_terms: set[str],
        text: str,
    ) -> float:
        if not query_terms:
            return 0.0

        text_terms = MetadataReranker._tokenize(text)

        if not text_terms:
            return 0.0

        return len(query_terms & text_terms) / len(query_terms)