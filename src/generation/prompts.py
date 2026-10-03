from typing import Any


SYSTEM_PROMPT = """You are a helpful Kubernetes documentation assistant.

Answer the user's question using only the provided context.

Rules:
- Use only information present in the context.
- Do not invent or assume information that is not in the context.
- If the context does not contain enough information to answer the question, say:
  "I don't have enough information in the provided context."
- Keep the answer concise and clear.
"""


class PromptBuilder:
    """Build grounded prompts for RAG generation."""

    def build(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> str:
        """Build a prompt using retrieved documents as context."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        context = self._build_context(results)

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n"
            f"{context}\n\n"
            f"User question:\n"
            f"{query}\n\n"
            f"Answer:"
        )

    @staticmethod
    def _build_context(
        results: list[dict[str, Any]],
    ) -> str:
        if not results:
            return "No context was retrieved."

        context_parts = []

        for index, result in enumerate(results, start=1):
            heading = result.get("heading", "")
            content = result.get("content", "")

            context_parts.append(
                f"[Document {index}]\n"
                f"Heading: {heading}\n"
                f"Content:\n{content}"
            )

        return "\n\n".join(context_parts)