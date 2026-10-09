from rag_bot.chunker import chunk_sections, chunk_text, split_sentences
from rag_bot.loaders import Section


def test_split_sentences():
    assert split_sentences("One. Two? Three!\n\nFour.") == ["One.", "Two?", "Three!", "Four."]


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=60)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # the last sentence of a chunk re-appears at the start of the next one
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(a)[-1] in b


def test_long_sentence_is_hard_split():
    chunks = chunk_text("word " * 500, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)


def test_metadata_is_kept():
    sections = [Section("Alpha. Beta.", "a.pdf", page=2), Section("Gamma.", "b.docx", section="Intro")]
    chunks = chunk_sections(sections, 800, 100)
    assert [c.citation for c in chunks] == ["a.pdf, page 2", 'b.docx, section "Intro"']
    assert [c.chunk_id for c in chunks] == [0, 1]
