"""
Validate the adversarial question set.

Usage:
    python scripts/validate_evalset.py

Two levels of checking:

  SCHEMA   every line parses, ids are unique, and the labels are internally
           coherent (an unanswerable item cannot expect a citation)

  CORPUS   every expected citation resolves to a real section in the index

The corpus check matters more than it looks. A dataset that cites sections
which do not exist would silently corrupt every number computed from it - the
system would be scored wrong for failing to cite something that was never
there, and the failure would look like a model problem rather than a data
problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.evalset import load_evalset, summarise, validate_against_corpus  # noqa: E402
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402


def main() -> int:
    print("=== Adversarial set validation ===\n")

    # --- schema ----------------------------------------------------------
    try:
        items = load_evalset()
    except (ValueError, FileNotFoundError) as exc:
        print(f"SCHEMA FAILED\n  {exc}")
        return 1

    print(f"[1/2] schema: OK - {len(items)} items parsed, ids unique")

    stats = summarise(items)
    print("\n  by category:")
    for name, count in sorted(stats["by_category"].items()):
        print(f"    {name:<22} {count:>3}")
    print("\n  by split:")
    for name, count in sorted(stats["by_split"].items()):
        print(f"    {name:<22} {count:>3}")
    print("\n  answerable:")
    print(f"    {'yes':<22} {stats['answerable']['yes']:>3}")
    print(f"    {'no (should abstain)':<22} {stats['answerable']['no']:>3}")

    # Both halves should cover every trap type, or a category could be tuned
    # on and never reported, or reported on and never tuned.
    categories = {i.category for i in items}
    for split_name in ("tuning", "heldout"):
        present = {i.category for i in items if i.split.value == split_name}
        missing = categories - present
        if missing:
            print(
                f"\n  WARNING: {split_name} half is missing categories: "
                f"{', '.join(sorted(c.value for c in missing))}"
            )

    # --- corpus ----------------------------------------------------------
    print("\n[2/2] corpus: checking every expected citation resolves ...")
    try:
        store = VectorStore()
        store.check_ready()
    except IndexMismatchError as exc:
        print(f"\n  SKIPPED - no usable index:\n  {exc}")
        return 1

    problems = validate_against_corpus(items, store)
    if problems:
        print(f"\n  FAILED - {len(problems)} bad citation(s):")
        for problem in problems:
            print(f"    {problem}")
        return 1

    cited = {c for i in items for c in i.expected_citations}
    print(f"  OK - {len(cited)} distinct citations all resolve")

    print("\nAdversarial set is valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
