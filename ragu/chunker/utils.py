import re
from dataclasses import dataclass, field
from typing import Any

from ragu.utils.ragu_utils import compute_mdhash_id

DESCRIPTION_LIMIT = 1000
FIRST_H1 = re.compile(r"^#[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
HEADING = re.compile(r"^#{1,2}[ \t]+(.+?)[ \t#]*$", re.MULTILINE)
HTML_TABLE = re.compile(r"<table\b.*?</table>", re.DOTALL | re.IGNORECASE)
FORMULA = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
IMAGE_ALT = re.compile(r"\[?!\[([^\]]*)\]")
TABLE_MARKUP = re.compile(r"<[^>]+>|\||-{3,}|\*\*")
EXTRA_NEWLINES = re.compile(r"\n{3,}")
MINERU_OBJECT_TYPES = frozenset({"table", "equation", "code", "image", "chart"})


@dataclass(slots=True)
class Span:
    start: int
    end: int
    chunk_cls: type
    content: str
    metadata: dict[str, str] = field(default_factory=dict)


def typed_id(chunk: Any) -> str:
    # NOTE: identical formulas (or any identical chunks of one class) within one document get the same id
    return compute_mdhash_id(type(chunk).__name__, chunk.doc_id, chunk.content, prefix="chunk-")


def join(*parts: str | None) -> str:
    return "\n\n".join(part for part in parts if part)


def context_header(metadata: dict[str, Any]) -> str:
    labels = (("document_title", "Document"), ("section", "Section"))
    return "\n".join(f"{label}: {metadata[key]}" for key, label in labels if metadata.get(key))


def shorten(text: str) -> str:
    return text if len(text) <= DESCRIPTION_LIMIT else text[:DESCRIPTION_LIMIT].rsplit(" ", 1)[0] + "…"


def table_text(table: str) -> str:
    return shorten(" ".join(TABLE_MARKUP.sub(" ", table).split()))


def document_title(markdown: str, structured_content: dict[str, Any] | None) -> str | None:
    title = (structured_content or {}).get("metadata", {}).get("document", {}).get("title")
    match = FIRST_H1.search(markdown)
    return title or (match.group(1).strip() if match else None)


def normalize(text: str) -> str:
    return " ".join(text.split())


def mineru_captions(structured_content: dict[str, Any] | None) -> dict[str, str]:
    captions = {}
    for page in (structured_content or {}).get("pages", []):
        for block in page.get("blocks", []):
            caption = " ".join(item.get("content", "").strip() for item in block.get("captions") or []).strip()
            if block.get("type") in MINERU_OBJECT_TYPES and caption:
                for key in (block.get("image_source"), block.get("content")):
                    if key:
                        captions[normalize(key)] = caption
    return captions


def image_caption(text: str, start: int) -> dict[str, str]:
    match = IMAGE_ALT.match(text, start)
    return {"caption": match.group(1).strip()} if match and match.group(1).strip() else {}


def resolve(spans: list[Span]) -> list[Span]:
    resolved: list[Span] = []
    for span in sorted(spans, key=lambda item: (item.start, -item.end)):
        if not resolved or span.start >= resolved[-1].end:
            resolved.append(span)
    return resolved


def mask(text: str, spans: list[Span]) -> str:
    for span in spans:
        text = text[:span.start] + " " * (span.end - span.start) + text[span.end:]
    return text


def cut(text: str, start: int, end: int, spans: list[Span]) -> str:
    parts, position = [], start
    for span in spans:
        # TODO: leave a reference to the object chunk in place of the cut-out object
        parts.append(text[position:span.start])
        position = max(position, span.end)
    parts.append(text[position:end])
    return EXTRA_NEWLINES.sub("\n\n", "".join(parts)).strip()


def sections(masked: str) -> list[tuple[str | None, int, int, int]]:
    headings = list(HEADING.finditer(masked))
    starts = [match.start() for match in headings] + [len(masked)]
    return [(None, 0, 0, starts[0])] + [
        (match.group(1).strip("*_ ") or None, match.start(), match.end(), end)
        for match, end in zip(headings, starts[1:], strict=True)
    ]
