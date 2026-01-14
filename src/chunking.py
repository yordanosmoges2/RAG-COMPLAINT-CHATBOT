from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    text: str
    chunk_index: int
    total_chunks: int


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    """
    Character-based chunking that matches the assignment’s reference setup:
    chunk_size=500 chars, chunk_overlap=50 chars :contentReference[oaicite:12]{index=12}
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    total = len(chunks)
    return [Chunk(text=c, chunk_index=i, total_chunks=total) for i, c in enumerate(chunks)]
