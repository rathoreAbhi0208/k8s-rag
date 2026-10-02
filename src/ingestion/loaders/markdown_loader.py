from pathlib import Path
from typing import Any

import yaml


class MarkdownLoader:
    """Load Markdown/MDX files and extract front matter."""

    SUPPORTED_EXTENSIONS = {".md", ".mdx"}

    def load(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}"
            )

        raw_content = path.read_text(encoding="utf-8")

        frontmatter, content = self._parse_frontmatter(
            raw_content
        )

        metadata = {
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "file_size": path.stat().st_size,
            **frontmatter,
        }

        return {
            "content": content,
            "metadata": metadata,
        }

    @staticmethod
    def _parse_frontmatter(
        content: str,
    ) -> tuple[dict[str, Any], str]:

        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)

        if len(parts) != 3:
            return {}, content

        frontmatter_text = parts[1]
        markdown_content = parts[2].strip()

        frontmatter = yaml.safe_load(frontmatter_text) or {}

        return frontmatter, markdown_content