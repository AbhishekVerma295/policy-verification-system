"""
Run the adversarial set through the system and report the metrics.

Usage:
    python scripts/evaluate.py                     # every item (slow)
    python scripts/evaluate.py --split heldout     # the half to report on
    python scripts/evaluate.py --sample 10         # 10 items, spread across categories
    python scripts/evaluate.py --category numeric
    python scripts/evaluate.py --ids num-001 fp-003
    python scripts/evaluate.py --report eval/results/run-....jsonl   # re-print a saved run
    python scripts/evaluate.py --report <log> --split heldout       # ... one half of it
    python scripts/evaluate.py --report <log> --baseline            # ... verification off

Requires an index (python scripts/build_index.py) and a running Ollama.

Every item goes through pipeline.answer_question(), the same entry point the
CLI and the UI use, so the numbers describe the system people actually see.
Results are appended to eval/results/<run-id>.jsonl one item at a time, so an
interrupted run keeps what it finished.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.config import get_config  # noqa: E402
from policyverify.evalset import Split, load_evalset, split_items  # noqa: E402
from policyverify.evaluate import (  # noqa: E402
    append_result,
    format_report,
    load_results,
    new_run_id,
    rescore_without_verification,
    run_item,
    save_summary,
    stratified_sample,
    summarise_results,
)
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402
from policyverify.llm import get_llm  # noqa: E402
from policyverify.pipeline import answer_question  # noqa: E402
from policyverify.verify import NLIChecker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate on the adversarial set.")
    parser.add_argument("--split", choices=["tuning", "heldout", "all"], default="all")
    parser.add_argument("--category", default=None, help="only this trap category")
    parser.add_argument("--ids", nargs="+", default=None, help="only these item ids")
    parser.add_argument(
        "--sample", type=int, default=0, help="this many items, spread across categories"
    )
    parser.add_argument("--report", default=None, help="re-print a saved run log and exit")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="with --report: score the same run as if verification were switched off",
    )
    args = parser.parse_args()

    if args.report:
        results = load_results(args.report)
        if args.split != "all":
            results = [r for r in results if r.split == args.split]
        if args.category:
            results = [r for r in results if r.category == args.category]
        if args.baseline:
            results = rescore_without_verification(results, load_evalset())
            print("BASELINE - the same generations with verification switched off\n")
        print(format_report(summarise_results(results), results))
        return 0

    config = get_config()
    items = load_evalset()
    if args.split != "all":
        items = split_items(items, Split(args.split))
    if args.category:
        items = [i for i in items if i.category.value == args.category]
    if args.ids:
        wanted = set(args.ids)
        items = [i for i in items if i.id in wanted]
    if args.sample:
        items = stratified_sample(items, args.sample)
    if not items:
        print("No items match those filters.")
        return 1

    # Load everything once. Loading per question would spend most of the run
    # re-reading model weights.
    print("loading models ...")
    store = VectorStore(config)
    try:
        store.check_ready()
    except IndexMismatchError as exc:
        print(f"\n{exc}\n")
        return 1
    nli = NLIChecker(config)
    llm = get_llm(config)

    def answer_fn(question: str):
        return answer_question(question, config=config, store=store, llm=llm, nli=nli)

    run_id = new_run_id()
    results_dir = config.paths.resolve("runs")
    log_path = results_dir / f"{run_id}.jsonl"
    print(f"{len(items)} items -> {log_path}\n")

    results = []
    started = time.perf_counter()
    for n, item in enumerate(items, 1):
        result = run_item(item, answer_fn)
        append_result(log_path, result)
        results.append(result)
        mark = "ok  " if result.passed else "FAIL"
        print(
            f"[{n:>2}/{len(items)}] {mark} {item.id:<10} {result.outcome.value:<22} "
            f"{result.total_ms / 1000:5.1f}s",
            flush=True,
        )

    summary = summarise_results(results)
    meta = {
        "run_id": run_id,
        "split": args.split,
        "category": args.category,
        "sample": args.sample or None,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "config": config.fingerprint(),
    }
    summary_path = save_summary(results_dir / f"{run_id}.summary.json", summary, meta)

    print("\n" + "=" * 74)
    print(format_report(summary, results))
    print("=" * 74)
    print(f"\nlog     {log_path}\nsummary {summary_path}")
    if args.split != "heldout" or args.sample or args.category or args.ids:
        print(
            "\nNote: report final numbers on the held-out half only "
            "(--split heldout, no other filters)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
