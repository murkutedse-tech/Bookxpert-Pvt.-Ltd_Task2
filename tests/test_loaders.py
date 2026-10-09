from pathlib import Path

from rag_bot.loaders import load_documents, strip_headers_footers

DATA = Path(__file__).resolve().parent.parent / "data"


def test_headers_footers_and_page_numbers_removed():
    bodies = ["Alpha text", "Beta text", "Gamma text"]
    pages = [["My Report", body, f"Page {i}"] for i, body in enumerate(bodies, start=1)]
    assert strip_headers_footers(pages) == [[b] for b in bodies]


def test_sample_documents_load():
    docs = load_documents(DATA)
    assert len(docs) >= 4
    assert any(name.endswith(".pdf") for name in docs)
    for name, sections in docs.items():
        words = sum(len(s.text.split()) for s in sections)
        assert words >= 500, f"{name} has only {words} words"


def test_pdf_has_page_metadata_and_no_footer():
    sections = load_documents(DATA)["accounting_fundamentals.pdf"]
    assert [s.page for s in sections] == list(range(1, len(sections) + 1))
    assert not any("Knowledge Base Series" in s.text for s in sections)
