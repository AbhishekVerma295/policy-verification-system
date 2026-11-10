"""
A scripted terminal demo: a fixed set of questions, models loaded once.

Usage:
    python scripts/demo.py            # all the questions below
    python scripts/demo.py --pause    # wait for Enter between questions

Requires an index (python scripts/build_index.py) and a running Ollama.

scripts/ask.py loads every model afresh for each question, which is fine for
one question and painful for six. This loads them once and reuses them, and
picks questions that each show a different behaviour of the system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask import render  # noqa: E402  (scripts/ is on the path when run as a script)
from policyverify.config import get_config  # noqa: E402
from policyverify.generate import GenerationError  # noqa: E402
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402
from policyverify.llm import LLMError, get_llm  # noqa: E402
from policyverify.pipeline import answer_question  # noqa: E402
from policyverify.verify import NLIChecker  # noqa: E402

# (what this one shows, the question)
QUESTIONS = [
    (
        "A plain answerable question - claims verified against the cited rule",
        "What is the minimum attendance required to sit the end semester examination?",
    ),
    (
        "A different policy - the hostel rules",
        "How many days in advance must a parent inform the hostel office before staying?",
    ),
    (
        "A numeric trap - 10% condonation sits next to a 25% leave allowance",
        "How much attendance condonation can be granted at most?",
    ),
    (
        "A false premise - the question asserts a rule that does not exist",
        "Why does the university require 90% attendance in every course?",
    ),
    (
        "Not in the corpus - the right behaviour is to decline",
        "What is the campus wifi password?",
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Scripted demo.")
    parser.add_argument("--pause", action="store_true", help="wait for Enter between questions")
    args = parser.parse_args()

    config = get_config()
    print("loading models (once) ...")
    store = VectorStore(config)
    try:
        store.check_ready()
    except IndexMismatchError as exc:
        print(f"\n{exc}\n")
        return 1
    nli = NLIChecker(config)
    llm = get_llm(config)

    for n, (shows, question) in enumerate(QUESTIONS, 1):
        print(f"\n\n### {n}/{len(QUESTIONS)}  {shows}")
        try:
            answer = answer_question(question, config=config, store=store, llm=llm, nli=nli)
        except (LLMError, GenerationError) as exc:
            print(f"\nCould not produce an answer: {exc}\n")
            return 1
        render(answer)
        if args.pause and n < len(QUESTIONS):
            input("press Enter for the next question ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
