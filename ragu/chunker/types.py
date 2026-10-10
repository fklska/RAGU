from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from ragu.utils.ragu_utils import compute_mdhash_id


@dataclass(slots=True)
class Chunk:
    id: str=field(init=False)
    content: str
    chunk_order_idx: int
    doc_id: str
    num_tokens: int | None = None
    metadata: dict[str: Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = compute_mdhash_id(self.content, prefix="chunk-")

    @abstractmethod
    def to_embed(self) -> str:
        pass

    @abstractmethod
    def to_llm(self) -> str:
        pass



@dataclass
class TextChunk(Chunk):
    def to_embed(self) -> str:
        return self.content

    def to_llm(self) -> str:
        content = [f"{key} - {value}\n" for key, value in self.metadata.items()]
        return "".join(content)


@dataclass
class TableChunk(Chunk):
    def to_embed(self) -> str:
        return self.metadata.get("description", "")

    def to_llm(self) -> str:
        return self.content


@dataclass
class FormulaChunk(Chunk):
    def to_embed(self) -> str:
        return self.metadata.get("description", "")

    def to_llm(self) -> str:
        return self.content


@dataclass
class ImageChunk(Chunk):
    def to_embed(self) -> str:
        return self.metadata.get("description", "")

    def to_llm(self) -> str:
        return self.metadata.get("description", "")