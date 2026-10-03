from typing import Any

import ollama


class OllamaClient:
    """Client wrapper for local Ollama text generation."""

    def __init__(
        self,
        model: str = "llama3.2:3b",
    ):
        self.model = model

    def generate(
        self,
        prompt: str,
    ) -> str:
        """Generate a response from Ollama."""

        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response: dict[str, Any] = ollama.generate(
            model=self.model,
            prompt=prompt,
        )

        return response["response"]