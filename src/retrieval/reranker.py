from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved documents using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Rerank search results and return the top results."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if not results:
            return []

        top_k = min(top_k, len(results))

        # ---------------------------------------------------------
        # Prepare query/document pairs
        # ---------------------------------------------------------

        pairs = [
            (
                query,
                result.get("content", ""),
            )
            for result in results
        ]

        # ---------------------------------------------------------
        # Generate reranker scores
        # ---------------------------------------------------------

        scores = self.model.predict(pairs)

        # ---------------------------------------------------------
        # Attach scores
        # ---------------------------------------------------------

        reranked_results = []

        for result, score in zip(results, scores):
            reranked_result = result.copy()
            reranked_result["rerank_score"] = float(score)
            reranked_results.append(reranked_result)

        # ---------------------------------------------------------
        # Sort by reranker score
        # ---------------------------------------------------------

        reranked_results.sort(
            key=lambda result: result["rerank_score"],
            reverse=True,
        )

        # ---------------------------------------------------------
        # Debug output
        # ---------------------------------------------------------

        # print("\n" + "=" * 80)
        # print("RERANKER RESULTS")
        # print("=" * 80)

        # print(f"Query: {query}")
        # print(f"Candidates: {len(results)}")
        # print(f"Returning: {top_k}")

        # for index, result in enumerate(
        #     reranked_results,
        #     start=1,
        # ):
        #     print(
        #         f"\n{index}. "
        #         f"score={result['rerank_score']:.4f}"
        #     )
        #     print(
        #         f"   Heading:      "
        #         f"{result.get('heading', '')}"
        #     )
        #     print(
        #         f"   Heading Path: "
        #         f"{result.get('heading_path', '')}"
        #     )
        #     print(
        #         f"   Chunk Index:  "
        #         f"{result.get('chunk_index', '')}"
        #     )
        #     print(
        #         f"   Semantic:     "
        #         f"{result.get('distance', '')}"
        #     )

        # print("=" * 80)

        return reranked_results[:top_k]