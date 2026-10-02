from dataclasses import dataclass
from typing import Any
import re


@dataclass
class DocumentChunk:
    content: str
    metadata: dict[str, Any]


class MarkdownChunker:
    """Split Markdown documents while preserving heading context."""

    def __init__(self, max_chars: int = 2000):
        self.max_chars = max_chars

    def chunk(
        self,
        content: str,
        metadata: dict[str, Any],
    ) -> list[DocumentChunk]:

        sections = self._split_by_headings(content)

        chunks = []

        global_chunk_index = 0

        for section in sections:
            section_chunks = self._split_large_section(
                section["content"]
            )

            for chunk_content in section_chunks:
                chunk_metadata = metadata.copy()

                chunk_metadata.update(
                    {
                        "heading": section["heading"],
                        "chunk_index": global_chunk_index,
                    }
                )

                chunks.append(
                    DocumentChunk(
                        content=chunk_content.strip(),
                        metadata=chunk_metadata,
                    )
                )

                global_chunk_index += 1

        return chunks

    def _split_by_headings(self, content: str) -> list[dict[str, str]]:
        lines = content.splitlines()

        sections = []
        current_heading = "root"
        current_content = []

        for line in lines:

            if re.match(r"^#{1,6}\s+", line):
                section_content = "\n".join(current_content).strip()

                if section_content:
                    sections.append(
                        {
                            "heading": current_heading,
                            "content": section_content,
                        }
                    )

                current_heading = line.strip("# ").strip()
                current_content = []

            else:
                current_content.append(line)

        section_content = "\n".join(current_content).strip()

        if section_content:
            sections.append(
                {
                    "heading": current_heading,
                    "content": section_content,
                }
            )

        return sections

    def _split_large_section(self, content: str) -> list[str]:

        if len(content) <= self.max_chars:
            return [content]

        chunks = []

        for i in range(0, len(content), self.max_chars):
            chunks.append(
                content[i : i + self.max_chars]
            )

        return chunks