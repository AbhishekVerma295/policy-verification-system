"""
Ask the system a question from the command line.

Usage:
    python scripts/ask.py "What is the minimum attendance requirement?"
    python scripts/ask.py "Can a parent stay overnight?" --policy residence
    python scripts/ask.py "..." -k 8 --show-passages

Requires an index (python scripts/build_index.py) and a running Ollama.

This is the same code path the UI and the evaluation harness use - they all
call into src/policyverify/, so a result seen here is the result they get.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.generate import GenerationError, generate_answer_draft  # noqa: E402
from policyverify.indexing import IndexMismatchError  # noqa: E402
from policyverify.llm import LLMError  # noqa: E402
from policyverify.retrieve import retrieve  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a policy question.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("-k", type=int, default=None, help="passages to retrieve")
    parser.add_argument("--policy", default=None, help="restrict to one policy type")
    parser.add_argument("--university", default=None, help="restrict to one university")
    parser.add_argument(
        "--show-passages", action="store_true", help="print the retrieved passages"
    )
    args = parser.parse_args()

    config = get_config()

    try:
        t0 = time.time()
        chunks = retrieve(
            args.question,
            k=args.k,
            university=args.university,
            policy_type=args.policy,
            config=config,
        )
        retrieve_ms = (time.time() - t0) * 1000
    except IndexMismatchError as exc:
        print(f"\n{exc}\n")
        return 1

    print(f"\nQ: {args.question}")
    print(f"\nretrieved {len(chunks)} passages in {retrieve_ms:.0f}ms")
    for r in chunks:
        preview = " ".join(r.chunk.text.split())[:88]
        print(f"  [{r.score:.3f}] {r.chunk.citation_id}")
        if args.show_passages:
            print(f"          {preview}...")

    if not chunks:
        print("\nNo passages retrieved - nothing to ground an answer in.")
        return 0

    try:
        t0 = time.time()
        draft, fabricated = generate_answer_draft(args.question, chunks, config=config)
        generate_ms = (time.time() - t0) * 1000
    except (LLMError, GenerationError) as exc:
        print(f"\nGeneration failed: {exc}\n")
        return 1

    print(f"\n--- {len(draft.claims)} claims generated in {generate_ms:.0f}ms ---\n")
    for i, claim in enumerate(draft.claims, 1):
        print(f"  CLAIM {i}")
        print(f"    {claim.text}")
        for citation in claim.citation_ids:
            print(f"      -> {citation}")
        if not claim.citation_ids:
            print("      -> (no citation given)")
        print()

    if fabricated:
        print("  FABRICATED CITATIONS (cited but not among the retrieved passages):")
        for citation in fabricated:
            print(f"    !! {citation}")
        print()

    print("NOTE: these claims are NOT yet verified - that is Phase 4.")
    print("Nothing above has been checked against the passages it cites.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
