import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.semantic_search import SemanticSearcher


class HybridSearcher:
    """Combine semantic similarity and full-corpus BM25 retrieval."""

    def __init__(
        self,
        semantic_searcher: SemanticSearcher,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        if semantic_weight < 0 or keyword_weight < 0:
            raise ValueError("Search weights must be non-negative")

        if semantic_weight + keyword_weight == 0:
            raise ValueError(
                "At least one search weight must be greater than 0"
            )

        self.semantic_searcher = semantic_searcher
        self.repository = semantic_searcher.repository
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        candidate_k = max(candidate_k, top_k)

        # ---------------------------------------------------------
        # 1. Semantic retrieval from Milvus
        # ---------------------------------------------------------
        semantic_results = self.semantic_searcher.search(
            query=query,
            top_k=candidate_k,
        )

        # ---------------------------------------------------------
        # 2. Retrieve the complete corpus for BM25
        # ---------------------------------------------------------
        corpus = self.repository.get_all_chunks()

        if not corpus:
            return []

        documents = [
            f"{document.get('heading', '')} "
            f"{document.get('content', '')}"
            for document in corpus
        ]

        tokenized_documents = [
            self._tokenize(document)
            for document in documents
        ]

        query_tokens = self._tokenize(query)

        bm25 = BM25Okapi(tokenized_documents)
        bm25_scores = bm25.get_scores(query_tokens).tolist()

        # ---------------------------------------------------------
        # 3. Get top BM25 candidates
        # ---------------------------------------------------------
        bm25_ranked_indices = sorted(
            range(len(corpus)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )[:candidate_k]

        # ---------------------------------------------------------
        # 4. Normalize semantic scores
        # ---------------------------------------------------------
        semantic_scores = [
            result["distance"]
            for result in semantic_results
        ]

        normalized_semantic = self._normalize_scores(
            semantic_scores
        )

        semantic_by_id = {
            result["id"]: normalized_semantic[index]
            for index, result in enumerate(semantic_results)
        }

        # ---------------------------------------------------------
        # 5. Normalize BM25 scores over the full corpus
        # ---------------------------------------------------------
        normalized_bm25 = self._normalize_scores(bm25_scores)

        bm25_by_id = {
            corpus[index]["id"]: normalized_bm25[index]
            for index in range(len(corpus))
        }

        # ---------------------------------------------------------
        # 6. Merge semantic + BM25 candidates
        # ---------------------------------------------------------
        candidate_ids = set(semantic_by_id)

        candidate_ids.update(
            corpus[index]["id"]
            for index in bm25_ranked_indices
        )

        corpus_by_id = {
            document["id"]: document
            for document in corpus
        }

        # ---------------------------------------------------------
        # 7. Calculate hybrid score
        # ---------------------------------------------------------
        hybrid_results = []

        for document_id in candidate_ids:
            document = corpus_by_id[document_id]

            semantic_score = semantic_by_id.get(
                document_id,
                0.0,
            )

            keyword_score = bm25_by_id.get(
                document_id,
                0.0,
            )

            hybrid_score = (
                self.semantic_weight * semantic_score
                + self.keyword_weight * keyword_score
            )

            result = {
                "id": document_id,
                "distance": semantic_score,
                "content": document.get("content", ""),
                "source": document.get("source", ""),
                "filename": document.get("filename", ""),
                "heading": document.get("heading", ""),
                "chunk_index": document.get("chunk_index", 0),
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "hybrid_score": hybrid_score,
            }

            hybrid_results.append(result)

        # ---------------------------------------------------------
        # 8. Rank by hybrid score
        # ---------------------------------------------------------
        hybrid_results.sort(
            key=lambda result: result["hybrid_score"],
            reverse=True,
        )

        return hybrid_results[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0] * len(scores)

        return [
            (score - minimum) / (maximum - minimum)
            for score in scores
        ]