"""Step 1 - Document ingestion.

Loads PDF, DOCX and TXT/MD files from a folder and returns clean text
"sections", each tagged with its source file and page number (PDF) or
section heading (DOCX/TXT).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class Section:
    text: str
    source: str            # file name, e.g. "accounting_fundamentals.pdf"
    page: int | None = None
    section: str | None = None

    @property
    def location(self) -> str:
        if self.page is not None:
            return f"page {self.page}"
        if self.section:
            return f"section '{self.section}'"
        return "document"


# --------------------------------------------------------------------------
# Text cleaning helpers
# --------------------------------------------------------------------------
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d+(\s*(of|/)\s*\d+)?\s*$", re.IGNORECASE)


def _normalise_line(line: str) -> str:
    """Digits are replaced so 'Page 3' and 'Page 4' count as the same line."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _join_lines(lines: list[str]) -> str:
    """Re-flow PDF lines into paragraphs and repair hyphenated line breaks."""
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # hyphen-
    text = re.sub(r"[ \t]+", " ", text)               # ation
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newlines -> spaces
    return text.strip()


def strip_headers_footers(pages: list[list[str]]) -> list[list[str]]:
    """Remove lines that repeat on most pages (running headers/footers) and
    lines that are only a page number."""
    if len(pages) >= 2:
        counts = Counter(
            norm for page in pages for norm in {_normalise_line(l) for l in page if l.strip()}
        )
        threshold = max(2, int(len(pages) * 0.5))
        repeated = {line for line, n in counts.items() if n >= threshold}
    else:
        repeated = set()

    cleaned = []
    for page in pages:
        cleaned.append([
            line for line in page
            if line.strip()
            and _normalise_line(line) not in repeated
            and not _PAGE_NUMBER_RE.match(line)
        ])
    return cleaned


# --------------------------------------------------------------------------
# Format-specific loaders
# --------------------------------------------------------------------------
def load_pdf(path: Path) -> list[Section]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    raw_pages = [(page.extract_text() or "").splitlines() for page in reader.pages]
    sections = []
    for page_no, lines in enumerate(strip_headers_footers(raw_pages), start=1):
        text = _join_lines(lines)
        if text:
            sections.append(Section(text=text, source=path.name, page=page_no))
    return sections


def load_docx(path: Path) -> list[Section]:
    import docx

    document = docx.Document(str(path))
    sections: list[Section] = []
    heading, buffer = None, []

    def flush():
        if buffer:
            sections.append(Section("\n\n".join(buffer), path.name, section=heading))

    for para in document.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower()
        if style.startswith("heading") or style == "title":
            flush()
            heading, buffer = text, []
        else:
            buffer.append(text)
    flush()
    return sections


def load_text(path: Path) -> list[Section]:
    """Plain text / markdown. Lines starting with '#' are treated as headings."""
    sections: list[Section] = []
    heading, buffer = None, []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(re.sub(r"(?<!\n)\n(?!\n)", " ", body), path.name, section=heading))

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.lstrip().startswith("#"):
            flush()
            heading, buffer = line.lstrip("# ").strip(), []
        else:
            buffer.append(line)
    flush()
    return sections


LOADERS = {".pdf": load_pdf, ".docx": load_docx, ".txt": load_text, ".md": load_text}


def load_documents(folder: Path) -> dict[str, list[Section]]:
    """Load every supported file in `folder`. Returns {file name: sections}."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")
    documents = {}
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            documents[path.name] = LOADERS[path.suffix.lower()](path)
    return documents
