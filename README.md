# Basic Document Q&A Bot (RAG Pipeline)

A command-line (and optional web) question-answering bot that answers natural-language questions **only** from a small collection of documents. It loads PDF, DOCX and TXT files, splits them into overlapping sentence-based chunks, embeds them with a local `sentence-transformers` model, stores them in a persistent FAISS index, and uses an LLM to write a grounded answer with citations (file name + page or section). If the answer is not in the documents, the bot says so instead of guessing.

Built for the Bookxpert Pvt. Ltd. AI Intern technical assignment (Stage 2).

---

## 1. Tech stack

| Component | Library / tool | Version |
|---|---|---|
| Language | Python | 3.11 or higher |
| PDF parsing | `pypdf` | >= 4.0 |
| DOCX parsing | `python-docx` | >= 1.1 |
| Embeddings | `sentence-transformers` (model `all-MiniLM-L6-v2`) | >= 3.0 |
| Vector database | `faiss-cpu` | >= 1.8 |
| Numerics | `numpy` | >= 1.26 |
| Config / secrets | `python-dotenv` | >= 1.0 |
| HTTP client for LLM APIs | `requests` | >= 2.31 |
| LLM (default, free, local) | Ollama with `llama3.2` | Ollama >= 0.3 |
| LLM (optional, API key) | Google Gemini `gemini-2.5-flash`, OpenAI `gpt-4o-mini`, Anthropic Claude via `anthropic` SDK | `anthropic` >= 0.40 |
| Web UI (bonus) | `streamlit` | >= 1.35 |
| Tests | `pytest` | >= 8.0 |

No framework such as LangChain is used: each RAG step is a small, readable module so every decision can be explained.

---

## 2. Project structure

```
.
├── data/                              # knowledge base (5 documents)
│   ├── accounting_fundamentals.pdf    # PDF  - double entry, statements, depreciation
│   ├── solar_energy_photovoltaics.pdf # PDF  - how solar PV works, India's solar programme
│   ├── gst_india_overview.docx        # DOCX - GST structure, ITC, registration, returns
│   ├── history_of_the_internet.txt    # TXT  - ARPANET, TCP/IP, the Web, governance
│   └── sleep_and_health.txt           # TXT  - sleep stages, circadian rhythm, disorders
├── rag_bot/
│   ├── config.py        # all settings, read from .env / environment variables
│   ├── loaders.py       # 1. ingestion + cleaning (headers, footers, page numbers)
│   ├── chunker.py       # 2. sentence-based chunking with overlap + metadata
│   ├── embedder.py      # 3. batched sentence-transformers embeddings
│   ├── vector_store.py  # 4. FAISS index persisted to disk
│   ├── llm.py           # 6. LLM providers (Ollama, Gemini, OpenAI, Claude)
│   └── pipeline.py      # indexing pipeline + QAEngine (5. retrieval, 6. generation)
├── ingest.py            # INDEXING step  (run once, or when documents change)
├── query.py             # QUERYING step  (interactive CLI loop)
├── app.py               # optional Streamlit web UI
├── tests/               # unit tests for loaders and chunker
├── requirements.txt
├── .env.example         # template for environment variables (no real keys)
└── vector_store/        # created by ingest.py (git-ignored)
```

---

## 3. Architecture overview

```
 INDEXING  (python ingest.py)                                       QUERYING  (python query.py)
 ───────────────────────────                                        ───────────────────────────
 data/*.pdf|docx|txt                                                user question
        │                                                                 │
        ▼                                                                 ▼
 [1] loaders.py  extract text, strip headers/                      [3] embed question (same model)
     footers/page numbers, tag page/section                               │
        │                                                                 ▼
        ▼                                                          [5] FAISS top-k similarity search
 [2] chunker.py  sentence-based chunks,                                   │   drop chunks below MIN_SCORE
     ~800 chars, ~150 chars overlap                                       │   (none left -> "not found", no LLM call)
        │                                                                 ▼
        ▼                                                          [6] prompt = rules + numbered chunks + question
 [3] embedder.py all-MiniLM-L6-v2,                                        │   LLM answers ONLY from context, cites [n]
     batched encode, L2-normalised                                        ▼
        │                                                          answer + citations (file, page/section)
        ▼                                                          + the retrieved source chunks and scores
 [4] vector_store.py  FAISS IndexFlatIP
     saved to vector_store/  ──────────── loaded from disk ─────────────►
```

1. **Ingestion** (`loaders.py`): PDFs are read page by page with `pypdf`. Lines that repeat on most pages (running headers and footers) and lines that are only a page number ("3", "Page 3", "3 of 10") are removed, hyphenated line breaks are repaired, and lines are re-flowed into paragraphs. DOCX files are split by their Heading styles and TXT/MD files by `#` heading lines, so each piece of text knows its **section**. PDFs keep their **page number**.
2. **Chunking** (`chunker.py`): see section 4.
3. **Embedding** (`embedder.py`): every chunk is embedded in one batched `model.encode(list_of_chunks, batch_size=64)` call. Vectors are L2-normalised.
4. **Vector store** (`vector_store.py`): vectors go into a FAISS `IndexFlatIP` (inner product = cosine similarity on normalised vectors). The index, the chunk texts with metadata, and a manifest (embedding model, chunk settings, document list) are saved to `vector_store/`. The query step only loads these files, it never re-indexes. If the embedding model in `.env` differs from the one the index was built with, the bot stops and asks you to re-run `ingest.py`.
5. **Retrieval** (`pipeline.py`): the question is embedded with the same model and the top-k chunks are retrieved (`TOP_K`, default 4, also `--top-k` or `:k 6` at runtime). Chunks below `MIN_SCORE` (default 0.30) are discarded as irrelevant.
6. **Generation** (`pipeline.py`, `llm.py`): the remaining chunks are numbered `[1]..[k]` with their source and sent to the LLM with a strict system prompt: answer only from the passages, cite them as `[n]`, and otherwise reply with a fixed "not found" sentence. The bot maps the `[n]` numbers back to `file, page/section` citations and prints them together with all retrieved chunks and their similarity scores.

**Two guardrails stop answers from the model's own training data:** (a) if no chunk passes the similarity threshold, the LLM is never called and the bot replies "I could not find the answer to this question in the provided documents."; (b) if chunks pass but do not contain the answer, the system prompt forces the same reply, which the bot detects and shows as "not found" with no sources.

---

## 4. Chunking strategy: sentence-based, ~800 characters, ~150 characters overlap

- Text is split into sentences, then whole sentences are packed into a chunk until it would exceed `CHUNK_SIZE` (800 characters, roughly 120 to 160 words).
- The next chunk starts with the last sentence(s) of the previous chunk, up to `CHUNK_OVERLAP` (150 characters), so a fact that sits on a chunk boundary is still complete in at least one chunk.
- A chunk never crosses a page (PDF) or section (DOCX/TXT) boundary, so every chunk has exactly one citation.
- Each chunk stores `chunk_id`, `source` (file name), `page` and/or `section`.

**Why:** fixed-size character windows often cut sentences in half, which hurts both embedding quality and the readability of the cited source. Paragraph-based chunks vary too much in size (some paragraphs are one line, some are half a page). Sentence packing keeps chunks semantically complete and of predictable size. 800 characters is small enough that one chunk is about one idea (good for precise retrieval and citations) and well within the 256-token input limit of `all-MiniLM-L6-v2`, while 4 chunks give the LLM enough context for multi-sentence answers.

---

## 5. Embedding model and vector database

**Embedding model: `sentence-transformers/all-MiniLM-L6-v2`**
- Free and runs locally on CPU, so no API key or cost is needed for indexing or retrieval.
- Small (about 80 MB, 384-dimensional vectors) and fast, yet strong on semantic-similarity benchmarks for English text.
- Trained for sentence and short-paragraph similarity, which matches the ~800-character chunks.
- Can be changed with `EMBEDDING_MODEL` in `.env` (for example `BAAI/bge-small-en-v1.5`); re-run `python ingest.py --force` afterwards.

**Vector database: FAISS (`faiss-cpu`)**
- Runs in-process with no server to install, and saves to a single file on disk.
- `IndexFlatIP` performs exact search; with a few hundred chunks an exact search takes well under a millisecond, so approximate indexes (IVF, HNSW) would add complexity with no benefit.
- Metadata (text, file, page, section) is kept in `chunks.json` next to the index, in the same order as the vectors.

---

## 6. Setup instructions

### 6.1 Clone and install

```bash
git clone https://github.com/murkutedse-tech/Bookxpert-Pvt.-Ltd_Task2.git
cd Bookxpert-Pvt.-Ltd_Task2

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Tip: on Linux, `pip install torch --index-url https://download.pytorch.org/whl/cpu` before the line above avoids downloading the large CUDA build of PyTorch.

### 6.2 Configure

```bash
# Windows: copy .env.example .env
cp .env.example .env
```

Then pick **one** LLM provider in `.env` (see section 7):

- **Ollama (free, local, default):** install from https://ollama.com, then run `ollama pull llama3.2`. Ollama runs in the background after installation (or start it with `ollama serve`).
- **Gemini (free API key):** create a key at https://aistudio.google.com/apikey, then set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY=...`.
- **OpenAI / Claude:** set `LLM_PROVIDER=openai` or `anthropic` and the matching key.

### 6.3 Indexing step (run once)

```bash
python ingest.py
```

Example output:

```
Loading documents from .../data ...
  - accounting_fundamentals.pdf: 3 pages, 1192 words -> 13 chunks
  - gst_india_overview.docx: 9 sections, 1065 words -> 13 chunks
  - history_of_the_internet.txt: 7 sections, 1025 words -> 12 chunks
  - sleep_and_health.txt: 7 sections, 942 words -> 9 chunks
  - solar_energy_photovoltaics.pdf: 3 pages, 1134 words -> 12 chunks
Embedding 59 chunks with 'sentence-transformers/all-MiniLM-L6-v2' (batch size 64) ...
Saved FAISS index with 59 vectors (dim 384) to .../vector_store
```

The first run downloads the embedding model (about 80 MB) from Hugging Face. Running `ingest.py` again does nothing unless you pass `--force` (use it after adding or changing documents).

### 6.4 Querying step

```bash
python query.py                  # interactive loop
python query.py --top-k 6        # retrieve more chunks
python query.py -p gemini        # override LLM provider for this run
python query.py -q "What is input tax credit?"   # one question, then exit
```

In the loop, type a question, `:k 5` to change top-k, or `:quit` to exit. Each answer shows the answer text, the **sources** (file + page/section), and every **retrieved chunk** with its similarity score.

### 6.5 Web UI (optional)

```bash
streamlit run app.py
```

### 6.6 Tests

```bash
pytest -q
```

---

## 7. Environment variables

All configuration lives in `.env` (copied from `.env.example`). `.env` is git-ignored; **never commit real API keys**.

| Variable | Required when | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | always | `ollama` | `ollama`, `gemini`, `openai`, `anthropic`, or `none` (no LLM; shows the best passage) |
| `GEMINI_API_KEY` | `LLM_PROVIDER=gemini` | - | Google AI Studio key |
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` | - | OpenAI key |
| `ANTHROPIC_API_KEY` | `LLM_PROVIDER=anthropic` | - | Anthropic key |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `LLM_PROVIDER=ollama` | `http://localhost:11434` / `llama3.2` | Local Ollama server and model |
| `GEMINI_MODEL`, `OPENAI_MODEL`, `ANTHROPIC_MODEL` | optional | `gemini-2.5-flash`, `gpt-4o-mini`, `claude-opus-5-5` | Model names |
| `EMBEDDING_MODEL` | optional | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | optional | `800`, `150` | Chunking (characters) |
| `TOP_K` | optional | `4` | Number of chunks retrieved |
| `MIN_SCORE` | optional | `0.30` | Similarity threshold below which chunks are ignored |

You can also set them in the shell instead of `.env`, e.g. `export GEMINI_API_KEY=...` (macOS/Linux) or `setx GEMINI_API_KEY "..."` (Windows, new terminal needed).

---

## 8. Example queries

| # | Question | Expected answer theme | Main source |
|---|---|---|---|
| 1 | What is the accounting equation and why must it always balance? | Assets = Liabilities + Equity; every transaction affects both sides equally | accounting_fundamentals.pdf, page 1 |
| 2 | How is straight-line depreciation calculated? | (Cost - residual value) / useful life; example machine -> 30,000 rupees a year | accounting_fundamentals.pdf, page 2 |
| 3 | What is the difference between CGST, SGST and IGST? | Intra-state = CGST + SGST; inter-state and imports = IGST; Pune/Mumbai vs Bengaluru example | gst_india_overview.docx, "The Dual GST Structure" |
| 4 | What conditions must be met to claim input tax credit? | Valid invoice, goods/services received, supplier paid tax, appears in GSTR-2B; blocked credits | gst_india_overview.docx, "Input Tax Credit" |
| 5 | How does a solar cell generate electricity? | Photovoltaic effect, p-n junction, electron-hole pairs, DC current | solar_energy_photovoltaics.pdf, page 1 |
| 6 | What is net metering? | Grid-tied export/import; billed for net energy | solar_energy_photovoltaics.pdf, page 2 |
| 7 | What was the first message sent over ARPANET? | "LO", UCLA to SRI, 29 October 1969 | history_of_the_internet.txt, "ARPANET" |
| 8 | Who invented the World Wide Web and what technologies did he create? | Tim Berners-Lee at CERN; HTML, HTTP, URLs | history_of_the_internet.txt, "The World Wide Web" |
| 9 | What happens during REM sleep? | Dreaming, rapid eye movement, muscle paralysis, memory and emotion | sleep_and_health.txt, "The Stages of Sleep" |
| 10 | Why does coffee in the afternoon affect sleep? | Caffeine blocks adenosine; half-life about 5 hours | sleep_and_health.txt, "The Body Clock and Melatonin" |
| 11 | Who won the 2011 Cricket World Cup? | **Not in the documents** -> "I could not find the answer ..." | none |

---

## 9. Known limitations

- **Small, fixed knowledge base.** The bot only knows these five documents; questions outside them are refused by design, even when the LLM "knows" the answer.
- **PDF structure is lost.** `pypdf` returns plain text, so PDF headings are merged into the surrounding text and tables or multi-column layouts may be extracted out of order. Scanned (image-only) PDFs would need OCR, which is not implemented. PDF citations are therefore by page, not by section.
- **Header/footer removal is heuristic.** It removes lines that repeat on at least half the pages; a single-page PDF keeps its header, and a genuine sentence that repeats on most pages would be removed.
- **Simple sentence splitter.** It is regex-based, so abbreviations like "Dr." or "e.g." can occasionally split a sentence.
- **Similarity threshold is a trade-off.** `MIN_SCORE=0.30` filters unrelated questions cheaply, but a very short or vaguely worded relevant question can score below it; lowering it lets more borderline questions reach the LLM (which still refuses if the context lacks the answer).
- **No conversation memory.** Each question is answered independently, so follow-ups like "and what about IGST?" work less well than a full question.
- **Questions that need many chunks** (e.g. "summarise every document") are limited by top-k; increase `--top-k` for broad questions.
- **English only.** `all-MiniLM-L6-v2` is an English model.
- **Answer quality depends on the LLM.** Small local models (e.g. `llama3.2` 3B) are slower on CPU and can occasionally forget to add `[n]` citations; the bot then lists all chunks it sent as sources. Hosted models (Gemini, GPT, Claude) follow the citation format more reliably.
- **Facts are a snapshot.** Figures such as GST rates and thresholds reflect the documents as written and may change over time.
