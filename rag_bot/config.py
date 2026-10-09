"""Central configuration. Every value can be overridden with an environment
variable (or a .env file in the project root)."""

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    # Paths
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("DATA_DIR", "data"))
    index_dir: Path = field(default_factory=lambda: PROJECT_ROOT / _env("INDEX_DIR", "vector_store"))

    # Chunking (sizes are in characters)
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "150")))

    # Embedding
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embedding_batch_size: int = field(default_factory=lambda: int(_env("EMBEDDING_BATCH_SIZE", "64")))

    # Retrieval
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    # Chunks scoring below this cosine similarity are treated as irrelevant.
    min_score: float = field(default_factory=lambda: float(_env("MIN_SCORE", "0.30")))

    # Generation
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama").lower())
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-opus-5-5"))
    request_timeout: int = field(default_factory=lambda: int(_env("LLM_TIMEOUT", "120")))
