from typing import Any

from src.generation.ollama_client import OllamaClient
from src.generation.prompts import PromptBuilder


class Generator:
    """Generate grounded answers from retrieved context."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        prompt_builder: PromptBuilder | None = None,
    ):
        self.ollama_client = ollama_client
        self.prompt_builder = prompt_builder or PromptBuilder()

    def generate(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> str:
        """Build a grounded prompt and generate an answer."""

        if not query.strip():
            raise ValueError("Query cannot be empty")

        prompt = self.prompt_builder.build(
            query=query,
            results=results,
        )

        return self.ollama_client.generate(prompt)