"""
Ask the system a question from the command line.

Usage:
    python scripts/ask.py "What is the minimum attendance requirement?"
    python scripts/ask.py "Can a parent stay overnight?" --policy residence
    python scripts/ask.py "..." -k 8 --json

Requires an index (python scripts/build_index.py) and a running Ollama.

This calls pipeline.answer_question(), the same entry point the UI and the
evaluation harness use, so what you see here is what they measure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.generate import GenerationError  # noqa: E402
from policyverify.indexing import IndexMismatchError  # noqa: E402
from policyverify.llm import LLMError  # noqa: E402
from policyverify.pipeline import answer_question  # noqa: E402
from policyverify.schema import Answer, ClaimVerdict, VerdictStatus  # noqa: E402

LABELS = {
    VerdictStatus.SUPPORTED: "SUPPORTED",
    VerdictStatus.REFUTED: "REFUTED",
    VerdictStatus.NEUTRAL: "UNVERIFIED",
}
WIDTH = 74


def print_claim(index: int, verdict: ClaimVerdict) -> None:
    label = LABELS[verdict.status]
    print(f"\n  [{index}] {label}  ({verdict.score:.2f})")
    print(f"      {verdict.claim.text}")
    for citation in verdict.claim.citation_ids:
        print(f"        -> {citation}")
    if not verdict.claim.citation_ids:
        print("        -> (no citation given)")
    print(f"      {verdict.explanation}")
    if verdict.checks.numeric_detail and verdict.checks.numeric_ok is False:
        print(f"      numbers: {verdict.checks.numeric_detail}")


def render(answer: Answer) -> None:
    print(f"\nQ: {answer.question}")
    print("=" * WIDTH)

    if answer.abstained:
        print("\n  NO ANSWER GIVEN")
        print(f"  {answer.reason}")
    elif answer.claims_kept:
        print(f"\nANSWER  ({len(answer.claims_kept)} verified claims)")
        for i, verdict in enumerate(answer.claims_kept, 1):
            print_claim(i, verdict)
        if answer.reason:
            print(f"\n  Note: {answer.reason}")

    if answer.claims_removed:
        print("\n" + "-" * WIDTH)
        print(f"\nREMOVED  ({len(answer.claims_removed)} claims the system would not stand behind)")
        print("  Shown rather than hidden, so you can see what was rejected and why.")
        for i, verdict in enumerate(answer.claims_removed, 1):
            print_claim(i, verdict)

    print("\n" + "=" * WIDTH)
    t = answer.timings
    print(
        f"  retrieve {t.retrieve_ms:.0f}ms | generate {t.generate_ms:.0f}ms "
        f"| verify {t.verify_ms:.0f}ms | total {t.total_ms:.0f}ms"
    )
    print(
        f"  {len(answer.claims_kept)} kept, {len(answer.claims_removed)} removed"
        f"{'  (ABSTAINED)' if answer.abstained else ''}"
    )
    if answer.claims_kept:
        print(f"  sources: {', '.join(answer.all_citations())}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a policy question.")
    parser.add_argument("question", help="the question to ask")
    parser.add_argument("-k", type=int, default=None, help="passages to retrieve")
    parser.add_argument("--policy", default=None, help="restrict to one policy type")
    parser.add_argument("--university", default=None, help="restrict to one university")
    parser.add_argument("--json", action="store_true", help="print the Answer as JSON")
    args = parser.parse_args()

    try:
        answer = answer_question(
            args.question,
            k=args.k,
            university=args.university,
            policy_type=args.policy,
            config=get_config(),
        )
    except IndexMismatchError as exc:
        print(f"\n{exc}\n")
        return 1
    except (LLMError, GenerationError) as exc:
        print(f"\nCould not produce an answer: {exc}\n")
        return 1

    if args.json:
        print(answer.model_dump_json(indent=2))
    else:
        render(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
