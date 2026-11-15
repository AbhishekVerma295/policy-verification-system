"""
Score the verifier on its own, against the labelled claim set.

Usage:
    python scripts/evaluate_verifier.py                   # every claim
    python scripts/evaluate_verifier.py --split tuning    # the half to tune on
    python scripts/evaluate_verifier.py --split heldout   # the half to report on
    python scripts/evaluate_verifier.py --kind role       # one kind of claim

Requires an index (python scripts/build_index.py). Does NOT need Ollama: the
claims are fixed in eval/claims.jsonl, so no language model is involved and
the whole run is the NLI model on CPU - about two minutes.

This is the number scripts/evaluate.py cannot give. That script scores whole
answers end to end; this one hands the verifier a claim and the section it
cites, and checks the verdict against a label. See claimset.py.

A summary is saved to eval/results/verifier-<run-id>.summary.json.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.claimset import (  # noqa: E402
    ClaimKind,
    evidence_for,
    format_claim_report,
    load_claimset,
    missing_citations,
    score_claims,
)
from policyverify.config import get_config  # noqa: E402
from policyverify.evalset import Split  # noqa: E402
from policyverify.evaluate import new_run_id, save_summary  # noqa: E402
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402
from policyverify.schema import Claim  # noqa: E402
from policyverify.verify import NLIChecker  # noqa: E402
from policyverify.verify.verifier import verify_claim  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the verifier on labelled claims.")
    parser.add_argument("--split", choices=["tuning", "heldout", "all"], default="all")
    parser.add_argument("--kind", choices=[k.value for k in ClaimKind], default=None)
    parser.add_argument("--no-save", action="store_true", help="do not write a summary file")
    args = parser.parse_args()

    config = get_config()
    items = load_claimset()
    if args.split != "all":
        items = [i for i in items if i.split is Split(args.split)]
    if args.kind:
        items = [i for i in items if i.kind.value == args.kind]
    if not items:
        print("No claims match those filters.")
        return 1

    print("loading models ...")
    store = VectorStore(config)
    try:
        store.check_ready()
    except IndexMismatchError as exc:
        print(f"\n{exc}\n")
        return 1

    problems = missing_citations(items, store)
    if problems:
        # A label pointing at a section that no longer exists cannot be scored.
        # Stop rather than report numbers over a silently smaller set.
        print("\nThe claim set cites sections that are not in the index:")
        for line in problems:
            print(f"  {line}")
        print("\nRe-check those labels (the index may have been rebuilt), then run again.")
        return 1

    nli = NLIChecker(config)
    statuses = {}
    started = time.perf_counter()
    print(f"{len(items)} claims\n")
    for n, item in enumerate(items, 1):
        verdict = verify_claim(
            Claim(text=item.claim, citation_ids=item.citation_ids),
            evidence_for(item, store),
            nli=nli,
            store=store,
            config=config,
        )
        statuses[item.id] = verdict.status
        if n % 20 == 0 or n == len(items):
            print(f"  {n}/{len(items)}", flush=True)
    seconds = time.perf_counter() - started

    summary = score_claims(items, statuses)
    print("\n" + "=" * 74)
    print(format_claim_report(summary, items, statuses))
    print("=" * 74)
    print(f"\n{seconds:.0f}s  ({seconds / len(items) * 1000:.0f} ms per claim)")

    if not args.no_save:
        run_id = new_run_id().replace("run-", "verifier-")
        meta = {
            "run_id": run_id,
            "split": args.split,
            "kind": args.kind,
            "wall_seconds": round(seconds, 1),
            "config": config.fingerprint(),
            "entailment_threshold": config.nli.entailment_threshold,
        }
        path = save_summary(
            config.paths.resolve("runs") / f"{run_id}.summary.json", summary, meta
        )
        print(f"summary {path}")
    if args.split != "heldout" or args.kind:
        print(
            "\nNote: tune on --split tuning; report final numbers on "
            "--split heldout with no other filters."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
