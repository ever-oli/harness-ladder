"""Small dependency-free lexical retriever used by the P3 rung."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9_./-]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "what", "which", "with", "you", "your",
}


@dataclass(frozen=True)
class Chunk:
    """A corpus passage and its stable source label."""

    source: str
    text: str


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN_RE.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 1
    }


def load_corpus(path: str | Path) -> list[Chunk]:
    """Load markdown/text passages separated by blank lines."""
    corpus_path = Path(path)
    raw = corpus_path.read_text(encoding="utf-8")
    chunks: list[Chunk] = []
    for index, block in enumerate(re.split(r"\n\s*\n", raw)):
        text = block.strip()
        if not text or (text.startswith("#") and "source:" not in text.lower()):
            continue
        source = corpus_path.name + f"#{index + 1}"
        first, _, rest = text.partition("\n")
        if first.lower().startswith("source:"):
            source = first.split(":", 1)[1].strip() or source
            text = rest.strip()
        chunks.append(Chunk(source=source, text=text))
    return chunks

def retrieve(query: str, chunks: Iterable[Chunk], top_k: int = 3) -> list[Chunk]:
    """Return highest-overlap passages, deterministically breaking ties."""
    if top_k <= 0:
        return []
    query_tokens = _tokens(query)
    scored: list[tuple[int, int, Chunk]] = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = _tokens(chunk.source + " " + chunk.text)
        overlap = len(query_tokens & chunk_tokens)
        if overlap:
            scored.append((overlap, -index, chunk))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [chunk for _, _, chunk in scored[:top_k]]


def retrieve_from_path(query: str, path: str | Path, top_k: int = 3) -> list[Chunk]:
    return retrieve(query, load_corpus(path), top_k=top_k)


def render_context(chunks: Iterable[Chunk]) -> str:
    """Render retrieved text as private reference, not assistant output."""
    passages = list(chunks)
    if not passages:
        return ""
    lines = [
        "Private local reference passages. Use them to answer the user's request;",
        "do not mention this retrieval step, repeat headers, or reveal scratch work.",
    ]
    for chunk in passages:
        lines.extend((f"[{chunk.source}]", chunk.text))
    return "\n".join(lines)
