"""Step 4 - FAISS vector store, persisted to disk.

Files written to the index folder:
  index.faiss    - the FAISS index (exact inner-product search)
  chunks.json    - chunk text + metadata, in the same order as the vectors
  manifest.json  - embedding model, settings and document list used to build it
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from .chunker import Chunk

INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE = "index.faiss", "chunks.json", "manifest.json"


class FaissStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], manifest: dict):
        self.index = index
        self.chunks = chunks
        self.manifest = manifest

    @classmethod
    def build(cls, vectors: np.ndarray, chunks: list[Chunk], manifest: dict) -> "FaissStore":
        if len(vectors) != len(chunks):
            raise ValueError("number of vectors and chunks differ")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        manifest = {**manifest, "num_chunks": len(chunks), "dimension": int(vectors.shape[1]),
                    "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        return cls(index, chunks, manifest)

    def save(self, folder: Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(folder / INDEX_FILE))
        (folder / CHUNKS_FILE).write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1), encoding="utf-8")
        (folder / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @staticmethod
    def exists(folder: Path) -> bool:
        return all((Path(folder) / f).exists() for f in (INDEX_FILE, CHUNKS_FILE, MANIFEST_FILE))

    @classmethod
    def load(cls, folder: Path) -> "FaissStore":
        folder = Path(folder)
        if not cls.exists(folder):
            raise FileNotFoundError(
                f"No vector store found in '{folder}'. Run `python ingest.py` first to build it.")
        index = faiss.read_index(str(folder / INDEX_FILE))
        chunks = [Chunk(**c) for c in json.loads((folder / CHUNKS_FILE).read_text(encoding="utf-8"))]
        manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
        return cls(index, chunks, manifest)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        k = max(1, min(k, len(self.chunks)))
        scores, ids = self.index.search(query_vector.reshape(1, -1).astype("float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]
