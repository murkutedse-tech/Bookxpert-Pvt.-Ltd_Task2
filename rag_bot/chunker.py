"""Step 2 - Sentence-based chunking with overlap.

Sentences are packed into a chunk until it reaches `chunk_size` characters.
The next chunk starts with the last few sentences of the previous one
(up to `overlap` characters), so an idea that spans a chunk boundary is
still fully present in at least one chunk. Chunks never cross a page or
section boundary, which keeps citations accurate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

from .loaders import Section

# Split after . ! ? when followed by whitespace and an upper-case letter,
# digit or quote. Simple, dependency-free and good enough for prose.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    page: int | None
    section: str | None

    @property
    def citation(self) -> str:
        if self.page is not None:
            return f"{self.source}, page {self.page}"
        if self.section:
            return f"{self.source}, section \"{self.section}\""
        return self.source

    def to_dict(self) -> dict:
        return asdict(self)


def split_sentences(text: str) -> list[str]:
    sentences = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if paragraph:
            sentences.extend(s.strip() for s in _SENTENCE_RE.split(paragraph) if s.strip())
    return sentences


def _split_long(sentence: str, size: int) -> list[str]:
    """Hard-split a single sentence that is longer than the chunk size."""
    words, parts, current = sentence.split(), [], ""
    for word in words:
        if current and len(current) + 1 + len(word) > size:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    sentences = []
    for s in split_sentences(text):
        sentences.extend(_split_long(s, chunk_size) if len(s) > chunk_size else [s])

    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for sentence in sentences:
        if current and length + 1 + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            # carry the tail of the previous chunk forward as overlap
            tail, tail_len = [], 0
            for prev in reversed(current):
                if tail_len + len(prev) > overlap:
                    break
                tail.insert(0, prev)
                tail_len += len(prev) + 1
            current, length = tail, tail_len
        current.append(sentence)
        length += len(sentence) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_sections(sections: list[Section], chunk_size: int, overlap: int,
                   start_id: int = 0) -> list[Chunk]:
    chunks = []
    next_id = start_id
    for section in sections:
        for piece in chunk_text(section.text, chunk_size, overlap):
            chunks.append(Chunk(next_id, piece, section.source, section.page, section.section))
            next_id += 1
    return chunks
