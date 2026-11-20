"""
Validate the adversarial question set.

Usage:
    python scripts/validate_evalset.py
    python scripts/validate_evalset.py --update-fingerprints

Four levels of checking, each catching a failure the previous one cannot:

  SCHEMA       every line parses, ids are unique, labels are internally
               coherent (an unanswerable item cannot expect a citation)

  EXISTENCE    every expected citation resolves to a real section

  CONTENT      the cited sections actually contain what the labels claim -
               required values are present, and numeric forbidden values are
               NOT present (if the wrong answer is sitting in the evidence,
               the item is not a trap)

  DRIFT        the cited sections are the same text the labels were written
               against, compared to eval/citation_fingerprints.json

WHY DRIFT MATTERS MOST
    Existence checking is not enough, and the gap is not theoretical. An
    ingestion change can re-split a document so a citation still resolves
    while the fact it was labelled for has moved into a different section.
    The ID is valid, the existence check passes, and the item is quietly
    scoring against the wrong passage. Drift is the only check here with
    full coverage, because it needs no label to compare against.

    After a deliberate ingestion change, re-check the reported items by hand,
    then re-run with --update-fingerprints to accept the new state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.evalset import (  # noqa: E402
    FINGERPRINT_PATH,
    check_expected_content,
    compare_fingerprints,
    content_check_coverage,
    fingerprint_citations,
    load_evalset,
    load_fingerprints,
    save_fingerprints,
    summarise,
    validate_against_corpus,
)
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the adversarial set.")
    parser.add_argument(
        "--update-fingerprints",
        action="store_true",
        help="record the current cited-section hashes as the accepted baseline",
    )
    args = parser.parse_args()

    print("=== Adversarial set validation ===\n")

    # --- 1. schema -------------------------------------------------------
    try:
        items = load_evalset()
    except (ValueError, FileNotFoundError) as exc:
        print(f"SCHEMA FAILED\n  {exc}")
        return 1

    print(f"[1/4] schema: OK - {len(items)} items parsed, ids unique")

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

    categories = {i.category for i in items}
    for split_name in ("tuning", "heldout"):
        present = {i.category for i in items if i.split.value == split_name}
        missing = categories - present
        if missing:
            print(
                f"\n  WARNING: {split_name} half is missing categories: "
                f"{', '.join(sorted(c.value for c in missing))}"
            )

    # --- index needed from here on ---------------------------------------
    try:
        store = VectorStore()
        store.check_ready()
    except IndexMismatchError as exc:
        print(f"\n  SKIPPED checks 2-4 - no usable index:\n  {exc}")
        return 1

    failed = False

    # --- 2. existence ----------------------------------------------------
    print("\n[2/4] existence: every expected citation resolves ...")
    problems = validate_against_corpus(items, store)
    if problems:
        failed = True
        print(f"  FAILED - {len(problems)} bad citation(s):")
        for problem in problems:
            print(f"    {problem}")
    else:
        cited = {c for i in items for c in i.expected_citations}
        print(f"  OK - {len(cited)} distinct citations all resolve")

    # --- 3. content ------------------------------------------------------
    cov = content_check_coverage(items)
    print("\n[3/4] content: cited sections hold what the labels claim ...")
    print(
        f"  coverage: {cov['required_checkable']} item(s) checkable by required value, "
        f"{cov['forbidden_checkable']} by numeric forbidden value, "
        f"{cov['uncheckable']} of {cov['with_citations']} cited items not checkable this way"
    )
    errors, warnings = check_expected_content(items, store)
    if errors:
        failed = True
        print(f"  FAILED - {len(errors)} stale label(s):")
        for problem in errors:
            print(f"    {problem}")
    else:
        print("  OK - every required value is present in its cited section")
    if warnings:
        print(f"\n  {len(warnings)} warning(s) - not failures, but worth confirming:")
        for warning in warnings:
            print(f"    {warning}")

    # --- 4. drift --------------------------------------------------------
    print("\n[4/4] drift: cited sections unchanged since labels were written ...")
    current = fingerprint_citations(items, store)

    if args.update_fingerprints:
        path = save_fingerprints(current)
        print(f"  RECORDED - {len(current)} citation fingerprints written to")
        print(f"    {path}")
        print("  Only do this after checking the reported items by hand.")
        return 1 if failed else 0

    recorded = load_fingerprints()
    if recorded is None:
        print(f"  no baseline yet ({FINGERPRINT_PATH.name} does not exist)")
        print("  run with --update-fingerprints to record the current state")
    else:
        drift = compare_fingerprints(recorded, current)
        if drift:
            failed = True
            print(f"  FAILED - {len(drift)} cited section(s) changed:")
            for problem in drift:
                print(f"    {problem}")
            print("\n  A citation can still resolve while the fact it was labelled for")
            print("  has moved. Re-check the listed items, then re-run with")
            print("  --update-fingerprints to accept the new state.")
        else:
            print(f"  OK - all {len(recorded)} cited sections unchanged")

    print("\n" + ("Adversarial set has problems - see above." if failed
                  else "Adversarial set is valid."))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
