"""Querying step: interactive command-line Q&A over the indexed documents.

Usage:
    python query.py                       # interactive loop
    python query.py --top-k 5
    python query.py -q "What is input tax credit?"   # single question

Inside the loop:
    :k 6      change top-k
    :quit     exit (also 'exit', 'quit' or Ctrl+C)
"""

import argparse
import textwrap

from rag_bot.config import Settings
from rag_bot.pipeline import QAEngine

WIDTH = 100


def show(answer, show_chunks: bool = True) -> None:
    print("\n" + "=" * WIDTH)
    print("ANSWER")
    print("-" * WIDTH)
    for para in answer.text.splitlines():
        print(textwrap.fill(para, WIDTH) if para.strip() else "")
    if answer.grounded:
        print("\nSOURCES")
        for citation in answer.citations:
            print(f"  * {citation}")

    if show_chunks:
        print("\nRETRIEVED CHUNKS (cosine similarity)")
        used = {id(c) for c, _ in answer.context}
        for rank, (chunk, score) in enumerate(answer.retrieved, start=1):
            tag = f"[{rank}]" if id(chunk) in used else "[--] below threshold, not sent to LLM"
            print("-" * WIDTH)
            print(f"{tag}  {chunk.citation}  |  score {score:.3f}  |  chunk #{chunk.chunk_id}")
            print(textwrap.fill(chunk.text, WIDTH, initial_indent="    ", subsequent_indent="    "))
    print("=" * WIDTH + "\n")


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Ask questions about the indexed documents.")
    parser.add_argument("-k", "--top-k", type=int, default=settings.top_k)
    parser.add_argument("-p", "--provider", default=settings.llm_provider,
                        help="ollama | gemini | openai | anthropic | none")
    parser.add_argument("-q", "--question", help="ask a single question and exit")
    parser.add_argument("--hide-chunks", action="store_true", help="do not print the retrieved chunks")
    args = parser.parse_args()
    settings.llm_provider = args.provider.lower()
    top_k = args.top_k

    print("Loading vector store and embedding model ...")
    engine = QAEngine(settings)
    m = engine.store.manifest
    print(f"Ready. {m['num_chunks']} chunks from {len(m['documents'])} documents: {', '.join(m['documents'])}")
    print(f"LLM provider: {settings.llm_provider} | top-k: {top_k} | min score: {settings.min_score}")

    if args.question:
        show(engine.ask(args.question, top_k), not args.hide_chunks)
        return

    print("Type a question, ':k <n>' to change top-k, or ':quit' to exit.\n")
    while True:
        try:
            question = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not question:
            continue
        if question.lower() in {":quit", ":q", "quit", "exit"}:
            print("Bye!")
            break
        if question.startswith(":k"):
            try:
                top_k = max(1, int(question.split()[1]))
                print(f"top-k set to {top_k}")
            except (IndexError, ValueError):
                print("Usage: :k <number>")
            continue
        show(engine.ask(question, top_k), not args.hide_chunks)


if __name__ == "__main__":
    main()
