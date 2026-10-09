"""Ties the steps together.

Indexing  (ingest.py):  load -> chunk -> embed (batched) -> save FAISS store
Querying  (query.py):   embed question -> top-k search -> prompt LLM -> answer + citations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .chunker import Chunk, chunk_sections
from .config import Settings
from .embedder import Embedder
from .llm import LLMError, generate
from .loaders import load_documents
from .vector_store import FaissStore

NOT_FOUND = "I could not find the answer to this question in the provided documents."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.
Answer the user's question using ONLY the numbered context passages provided.
Rules:
- Do not use any outside knowledge, even if you know the answer.
- If the passages do not contain the answer, reply with exactly: "{NOT_FOUND}"
- Cite the passages you used with their numbers in square brackets, e.g. [1] or [2][3].
- Be concise: a short paragraph or a few bullet points."""


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------
def build_index(settings: Settings, log=print) -> FaissStore:
    log(f"Loading documents from {settings.data_dir} ...")
    documents = load_documents(settings.data_dir)
    if not documents:
        raise RuntimeError(f"No PDF/DOCX/TXT files found in {settings.data_dir}")

    all_chunks: list[Chunk] = []
    for name, sections in documents.items():
        words = sum(len(s.text.split()) for s in sections)
        chunks = chunk_sections(sections, settings.chunk_size, settings.chunk_overlap,
                                start_id=len(all_chunks))
        all_chunks.extend(chunks)
        unit = "pages" if sections and sections[0].page is not None else "sections"
        log(f"  - {name}: {len(sections)} {unit}, {words} words -> {len(chunks)} chunks")

    log(f"Embedding {len(all_chunks)} chunks with '{settings.embedding_model}' "
        f"(batch size {settings.embedding_batch_size}) ...")
    embedder = Embedder(settings.embedding_model, settings.embedding_batch_size)
    vectors = embedder.embed([c.text for c in all_chunks], show_progress=True)

    store = FaissStore.build(vectors, all_chunks, {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "documents": sorted(documents),
    })
    store.save(settings.index_dir)
    log(f"Saved FAISS index with {len(all_chunks)} vectors (dim {vectors.shape[1]}) "
        f"to {settings.index_dir}")
    return store


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------
@dataclass
class Answer:
    question: str
    text: str
    grounded: bool
    retrieved: list[tuple[Chunk, float]] = field(default_factory=list)   # everything top-k returned
    context: list[tuple[Chunk, float]] = field(default_factory=list)     # passed to the LLM
    cited: list[Chunk] = field(default_factory=list)                     # cited in the answer

    @property
    def citations(self) -> list[str]:
        seen = []
        for chunk in self.cited:
            if chunk.citation not in seen:
                seen.append(chunk.citation)
        return seen


def build_user_prompt(question: str, context: list[tuple[Chunk, float]]) -> str:
    blocks = [f"[{i}] (source: {chunk.citation})\n{chunk.text}"
              for i, (chunk, _score) in enumerate(context, start=1)]
    return "Context passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


class QAEngine:
    """Loads the persisted index once, then answers questions."""

    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        self.store = FaissStore.load(settings.index_dir)
        built_with = self.store.manifest.get("embedding_model")
        if built_with != settings.embedding_model:
            raise RuntimeError(
                f"Index was built with '{built_with}' but EMBEDDING_MODEL is "
                f"'{settings.embedding_model}'. Re-run `python ingest.py`.")
        self.embedder = embedder or Embedder(settings.embedding_model, settings.embedding_batch_size)

    def retrieve(self, question: str, k: int) -> list[tuple[Chunk, float]]:
        return self.store.search(self.embedder.embed_query(question), k)

    def ask(self, question: str, k: int | None = None) -> Answer:
        k = k or self.settings.top_k
        retrieved = self.retrieve(question, k)
        context = [(c, s) for c, s in retrieved if s >= self.settings.min_score]

        # Guardrail 1: nothing relevant was retrieved -> refuse without calling the LLM.
        if not context:
            return Answer(question, NOT_FOUND, False, retrieved, [])

        if self.settings.llm_provider == "none":
            best = context[0][0]
            text = f"(LLM disabled - showing the most relevant passage)\n{best.text} [1]"
        else:
            try:
                text = generate(SYSTEM_PROMPT, build_user_prompt(question, context), self.settings)
            except LLMError as exc:
                return Answer(question, f"LLM error: {exc}", False, retrieved, context)

        # Guardrail 2: the LLM itself decided the context does not answer the question.
        if NOT_FOUND.lower().rstrip(".") in text.lower():
            return Answer(question, NOT_FOUND, False, retrieved, context)

        numbers = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
        cited = [context[n - 1][0] for n in dict.fromkeys(numbers) if 1 <= n <= len(context)]
        return Answer(question, text, True, retrieved, context, cited or [c for c, _ in context])
