"""Repo awareness: find the few chunks of code relevant to a question.

Chunk the codebase, index the chunks, and answer ``search_codebase`` queries by
returning the most relevant chunks. The RAG role then answers questions over
those chunks, so the planner asks questions instead of dumping whole files into
its small context window.

Two ranking modes:

* **TF-IDF (default)** — dependency-free cosine ranker over identifier tokens;
  works offline and in tests. Read :meth:`Retriever.search` — it is ~20 lines.
* **Dense** — pass ``embed_fn`` (any function returning a vector for a string,
  e.g. a call to an embedding endpoint) and the same API ranks by dense cosine
  instead. Not wired to a served model by default: single-vector embeddings
  need an embedding model served with ``llama-server --embedding``.

The built index is persisted under ``config.index_dir`` keyed by a signature of
the indexed files (path + size + mtime); an unchanged repo reloads instantly
instead of re-chunking every run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import RetrievalConfig

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")
_INDEX_VERSION = 2

EmbedFn = Callable[[str], list[float]]


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class Chunk:
    path: str
    start_line: int
    end_line: int
    text: str
    tf: Counter = field(default_factory=Counter)
    vec: list[float] | None = None  # dense embedding, when embed_fn is used

    def render(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}\n{self.text}"


@dataclass
class SearchHit:
    chunk: Chunk
    score: float

    def render(self) -> str:
        return f"[score={self.score:.3f}] {self.chunk.render()}"


class Retriever:
    def __init__(
        self,
        config: RetrievalConfig,
        workdir: Path,
        embed_fn: EmbedFn | None = None,
    ):
        self.config = config
        self.workdir = Path(workdir).resolve()
        # When set, chunks/queries are embedded densely; otherwise TF-IDF.
        self.embed_fn = embed_fn
        self.chunks: list[Chunk] = []
        self._idf: dict[str, float] = {}

    @property
    def dense(self) -> bool:
        return self.embed_fn is not None

    # -- indexing ------------------------------------------------------------ #
    def _iter_files(self):
        excluded = set(self.config.exclude_dirs)
        seen: set[Path] = set()
        for pattern in self.config.include_globs:
            for path in self.workdir.glob(pattern):
                if not path.is_file():
                    continue
                if any(part in excluded for part in path.relative_to(self.workdir).parts):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                yield path

    def _signature(self, files: list[Path]) -> str:
        h = hashlib.sha256()
        h.update(f"v{_INDEX_VERSION}|dense={self.dense}|".encode())
        h.update(f"{self.config.chunk_lines},{self.config.chunk_overlap}|".encode())
        for path in sorted(files):
            st = path.stat()
            h.update(f"{path}:{st.st_size}:{int(st.st_mtime)}\n".encode())
        return h.hexdigest()

    def _index_path(self) -> Path:
        return self.workdir / self.config.index_dir / "index.json"

    def build(self, force: bool = False) -> "Retriever":
        files = list(self._iter_files())
        signature = self._signature(files)
        if not force and self._load(signature):
            return self

        self.chunks = []
        step = max(1, self.config.chunk_lines - self.config.chunk_overlap)
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            rel = str(path.relative_to(self.workdir))
            for start in range(0, max(1, len(lines)), step):
                window = lines[start : start + self.config.chunk_lines]
                if not any(line.strip() for line in window):
                    continue
                text = "\n".join(window)
                self.chunks.append(
                    Chunk(
                        path=rel,
                        start_line=start + 1,
                        end_line=min(start + self.config.chunk_lines, len(lines)),
                        text=text,
                        tf=Counter(tokenize(text)),
                    )
                )
                if start + self.config.chunk_lines >= len(lines):
                    break
        if self.dense:
            for chunk in self.chunks:
                chunk.vec = list(self.embed_fn(chunk.text))
        else:
            self._compute_idf()
        self._save(signature)
        return self

    def _compute_idf(self) -> None:
        n = len(self.chunks) or 1
        df: Counter = Counter()
        for chunk in self.chunks:
            df.update(chunk.tf.keys())
        self._idf = {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}

    # -- persistence --------------------------------------------------------- #
    def _save(self, signature: str) -> None:
        try:
            path = self._index_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "signature": signature,
                "dense": self.dense,
                "idf": self._idf,
                "chunks": [
                    {
                        "path": c.path,
                        "start_line": c.start_line,
                        "end_line": c.end_line,
                        "text": c.text,
                        "tf": dict(c.tf),
                        "vec": c.vec,
                    }
                    for c in self.chunks
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass  # a read-only workdir just means we rebuild next time

    def _load(self, signature: str) -> bool:
        path = self._index_path()
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if payload.get("signature") != signature or payload.get("dense") != self.dense:
            return False
        self._idf = payload.get("idf", {})
        self.chunks = [
            Chunk(
                path=c["path"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                text=c["text"],
                tf=Counter(c.get("tf", {})),
                vec=c.get("vec"),
            )
            for c in payload.get("chunks", [])
        ]
        return True

    # -- search -------------------------------------------------------------- #
    def _tfidf_vec(self, tf: Counter) -> dict[str, float]:
        return {t: c * self._idf.get(t, 0.0) for t, c in tf.items()}

    @staticmethod
    def _sparse_cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _dense_cosine(a: list[float], b: list[float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        top_k = top_k or self.config.top_k
        if not self.chunks:
            return []
        if self.dense:
            qvec = list(self.embed_fn(query))
            scored = [
                SearchHit(c, self._dense_cosine(qvec, c.vec or []))
                for c in self.chunks
            ]
        else:
            qvec = self._tfidf_vec(Counter(tokenize(query)))
            scored = [
                SearchHit(c, self._sparse_cosine(qvec, self._tfidf_vec(c.tf)))
                for c in self.chunks
            ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return [h for h in scored[:top_k] if h.score > 0.0]
