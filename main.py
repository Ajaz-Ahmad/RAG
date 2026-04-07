"""Entry point for the Wikipedia RAG pipeline.

Usage
-----
  # Index one or more Wikipedia articles
  python main.py ingest https://en.wikipedia.org/wiki/Diabetes

  # Ask a question against the indexed documents
  python main.py query "What causes type 2 diabetes?"

  # Override settings via environment variables or a .env file
  OPENAI_API_KEY=sk-... python main.py query "What is insulin?"
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path regardless of where the script is run from
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from src.pipeline.orchestrator import ingest
from src.pipeline.query_pipeline import build_retriever, query as run_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    _, chunks = ingest(args.urls, settings)
    print(f"\nIndexed {len(chunks)} chunks from {len(args.urls)} URL(s).")
    print(f"Index saved to:  {settings.index_path}")
    print(f"Chunks saved to: {settings.chunks_path}")


def cmd_query(args: argparse.Namespace) -> None:
    retriever = build_retriever(settings)
    result = run_query(args.question, retriever, settings)

    print(f"\nAnswer:\n{result['answer']}\n")
    print("Sources:")
    for i, src in enumerate(result["sources"], 1):
        section = f"{src['section']} > {src['subsection']}".strip(" >")
        print(f"  [{i}] score={src['score']:.4f}  [{section}]")
        print(f"       {src['text_preview']}...")
        print()

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wikipedia RAG — ingest articles, then ask questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- ingest --
    ingest_p = sub.add_parser("ingest", help="Scrape and index Wikipedia URLs.")
    ingest_p.add_argument("urls", nargs="+", metavar="URL", help="Wikipedia article URLs.")
    ingest_p.set_defaults(func=cmd_ingest)

    # -- query --
    query_p = sub.add_parser("query", help="Ask a question against indexed documents.")
    query_p.add_argument("question", help="Natural language question.")
    query_p.add_argument("--json", action="store_true", help="Also print full JSON response.")
    query_p.set_defaults(func=cmd_query)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
