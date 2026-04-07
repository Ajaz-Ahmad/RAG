from dataclasses import dataclass, field
from typing import Any


@dataclass
class WikiChunk:
    """A raw paragraph extracted from a Wikipedia article, tagged with its section hierarchy."""
    page_title: str
    level: str
    section: str
    subsection: str
    context: str
    text: str

    def __post_init__(self) -> None:
        self.char_count: int = len(self.text)
        self.word_count: int = len(self.text.split())


@dataclass
class Chunk:
    """A merged, indexable unit of text ready for embedding and retrieval."""
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.word_count: int = len(self.text.split())
