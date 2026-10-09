"""Indexing step: load documents, chunk them, embed them and save the FAISS store.

Usage:
    python ingest.py                     # uses settings from .env / defaults
    python ingest.py --chunk-size 600 --overlap 120
"""

import argparse
import time

from rag_bot.config import Settings
from rag_bot.pipeline import build_index
from rag_bot.vector_store import FaissStore


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Build the vector index from the documents folder.")
    parser.add_argument("--data-dir", default=str(settings.data_dir))
    parser.add_argument("--index-dir", default=str(settings.index_dir))
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--force", action="store_true", help="rebuild even if an index already exists")
    args = parser.parse_args()

    settings.data_dir, settings.index_dir = type(settings.data_dir)(args.data_dir), type(settings.index_dir)(args.index_dir)
    settings.chunk_size, settings.chunk_overlap = args.chunk_size, args.overlap

    if FaissStore.exists(settings.index_dir) and not args.force:
        print(f"An index already exists in {settings.index_dir}. Use --force to rebuild it.")
        return

    start = time.perf_counter()
    build_index(settings)
    print(f"Done in {time.perf_counter() - start:.1f}s. Now run: python query.py")


if __name__ == "__main__":
    main()
