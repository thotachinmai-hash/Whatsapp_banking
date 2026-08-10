"""Minimal, dependency-free retriever over the markdown files in
app/rag/documents/.

Deliberately not using a vector-embedding library (faiss/chroma/
sentence-transformers) — this repo has no ML/embedding dependencies today
(see requirements.txt) and the corpus here is small (a handful of FAQ
documents), so a TF-IDF + cosine-similarity index in pure Python is enough
to find the right chunk without adding heavy new dependencies. If the
corpus grows much larger or needs semantic (not just keyword) matching,
swap this module for an embedding-backed store — callers only depend on
`search()`'s return shape, not how it's computed.

SENSITIVE-DATA RULE: every file under app/rag/documents/ is bank product/
policy information only — never account-specific or customer-specific
data. This module has no access to the database, so it structurally
cannot leak anything beyond what's written in those files.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.logger import get_logger

logger = get_logger(__name__)

DOCUMENTS_DIR = Path(__file__).parent / "documents"

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "for", "and", "or", "in", "on", "at", "it", "this", "that", "i", "you",
    "do", "does", "did", "doing", "my", "me", "can", "will", "with", "as", "if",
    "what", "how", "why", "when", "where", "which", "who", "whom", "whose",
    "about", "tell", "know", "get", "need", "needed", "needs", "there",
}


@dataclass
class Chunk:
    title: str
    text: str
    source: str
    term_freq: Counter


def _tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    if not DOCUMENTS_DIR.exists():
        logger.warning(f"RAG documents directory not found: {DOCUMENTS_DIR}")
        return chunks

    for path in sorted(DOCUMENTS_DIR.glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        current_title = ""
        current_lines: list[str] = []

        def _flush():
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    chunks.append(Chunk(
                        title=current_title or path.stem,
                        text=text,
                        source=path.name,
                        term_freq=Counter(_tokenize(f"{current_title} {text}")),
                    ))

        for line in lines:
            if line.startswith("#"):
                _flush()
                current_title = line.lstrip("#").strip()
                current_lines = [line]
            else:
                current_lines.append(line)
        _flush()

    logger.info(f"RAG index built | chunks={len(chunks)} | files={len(list(DOCUMENTS_DIR.glob('*.md')))}")
    return chunks


class _Index:
    """Lazily built once per process; documents don't change at runtime."""

    def __init__(self):
        self.chunks: list[Chunk] = []
        self.idf: dict[str, float] = {}
        self._built = False

    def _build(self):
        self.chunks = _load_chunks()
        n_docs = len(self.chunks) or 1
        doc_freq: Counter = Counter()
        for chunk in self.chunks:
            for term in chunk.term_freq:
                doc_freq[term] += 1
        self.idf = {
            term: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freq.items()
        }
        self._built = True

    def ensure_built(self):
        if not self._built:
            self._build()

    def reload(self):
        """Rebuild from disk — call after documents/*.md change."""
        self._build()


_index = _Index()


def _vector(term_freq: Counter, idf: dict[str, float]) -> dict[str, float]:
    return {term: freq * idf.get(term, 0.0) for term, freq in term_freq.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return dot / (norm_a * norm_b)


def search(query: str, top_k: int = 3, min_score: float = 0.12) -> list[dict]:
    """Return up to `top_k` chunks relevant to `query`, each as
    {"title": ..., "text": ..., "source": ..., "score": ...}, sorted by
    score descending. Empty list if nothing clears `min_score` — callers
    must treat that as "no grounded answer available", not fall back to
    guessing (see the RAG tool in app/agent/tools.py)."""
    _index.ensure_built()
    if not _index.chunks:
        return []

    query_terms = Counter(_tokenize(query))
    if not query_terms:
        return []

    query_vec = _vector(query_terms, _index.idf)

    scored = []
    for chunk in _index.chunks:
        chunk_vec = _vector(chunk.term_freq, _index.idf)
        score = _cosine(query_vec, chunk_vec)
        if score >= min_score:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {"title": chunk.title, "text": chunk.text, "source": chunk.source, "score": round(score, 3)}
        for score, chunk in scored[:top_k]
    ]
