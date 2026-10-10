from collections.abc import Sequence
from typing import Any

from chonkie import BaseChunker as ChonkieChunker
from chonkie import MarkdownChef, RecursiveChunker, SemanticChunker, SentenceTransformerEmbeddings

from ragu.chunker.base_chunker import BaseChunker
from ragu.chunker.types import Chunk, CodeChunk, Document, FormulaChunk, ImageChunk, TableChunk, TextChunk
from ragu.chunker.utils import (
    FORMULA,
    HTML_TABLE,
    Span,
    cut,
    image_caption,
    mask,
    mineru_captions,
    normalize,
    resolve,
    sections,
)


class MarkdownChunker(BaseChunker):
    def __init__(self, text_chunker: ChonkieChunker) -> None:
        super().__init__()
        self.text_chunker = text_chunker
        self.chef = MarkdownChef()

    def split(self, documents: str | Document | Sequence[str | Document]) -> list[Chunk]:
        if isinstance(documents, (str, Document)):
            documents = [documents]
        return [
            chunk
            for document in documents
            for chunk in self._split(document if isinstance(document, Document) else Document(document))
        ]

    def _split(self, document: Document) -> list[Chunk]:
        text = document.content
        spans = self._find_objects(text)
        captions = mineru_captions(document.structured_content)
        chunks: list[Chunk] = []
        for section, start, body_start, end in sections(mask(text, spans)):
            metadata = {"document_title": document.title, "section": section}
            metadata = {key: value for key, value in metadata.items() if value}
            inner = [span for span in spans if start <= span.start < end]
            section_text = cut(text, body_start, end, inner)
            for piece in self.text_chunker.chunk(section_text) if section_text else []:
                if piece.text.strip():
                    chunks.append(TextChunk(piece.text.strip(), len(chunks), document.id, metadata=dict(metadata)))
            for span in inner:
                chunk = span.chunk_cls(span.content, len(chunks), document.id, metadata={**metadata, **span.metadata})
                if caption := captions.get(normalize(span.content)):
                    chunk.metadata["caption"] = caption
                chunk.describe()
                chunks.append(chunk)
        return chunks

    def _find_objects(self, text: str) -> list[Span]:
        parsed = self.chef.parse(text + "\n")
        size = len(text)
        code = resolve([
            Span(block.start_index, min(block.end_index, size), CodeChunk, block.content,
                 {"language": block.language} if block.language else {})
            for block in parsed.code
        ])
        blocks = resolve(
            code
            + [Span(table.start_index, min(table.end_index, size), TableChunk, table.content.strip())
               for table in parsed.tables]
            + [Span(match.start(), match.end(), TableChunk, match.group(0))
               for match in HTML_TABLE.finditer(mask(text, code))]
        )
        return resolve(
            blocks
            + [Span(match.start(), match.end(), FormulaChunk, match.group(1).strip())
               for match in FORMULA.finditer(mask(text, blocks))]
            + [Span(image.start_index, min(image.end_index, size), ImageChunk, image.content,
                    image_caption(text, image.start_index))
               for image in parsed.images]
        )


class RecursiveMarkdownChunker(MarkdownChunker):
    def __init__(self, max_chunk_size: int = 1000, tokenizer: Any = "character", **kwargs: Any) -> None:
        super().__init__(RecursiveChunker(tokenizer=tokenizer, chunk_size=max_chunk_size, **kwargs))


class SemanticMarkdownChunker(MarkdownChunker):
    def __init__(
        self,
        model_name: str,
        max_chunk_size: int = 512,
        threshold: float = 0.8,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        embeddings = SentenceTransformerEmbeddings(model_name, device=device)
        super().__init__(SemanticChunker(embedding_model=embeddings, chunk_size=max_chunk_size,
                                         threshold=threshold, **kwargs))
