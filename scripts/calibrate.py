"""
Calibration and pass marks, measured on the TUNING half only.

Usage:
    python scripts/calibrate.py               # calibration + risk-coverage on tuning claims
    python scripts/calibrate.py --retrieval   # also: best retrieval score per question
    python scripts/calibrate.py --bins 4

Needs the index (python scripts/build_index.py). Does NOT need Ollama: the
claims are fixed in eval/claims.jsonl and retrieval uses only the embedding
model and BM25.

What it prints (see src/policyverify/calibration.py for the definitions):

  1. Calibration of the NLI support score (the best entailment probability
     over the cited section, the number entailment_threshold cuts) and of
     the confidence a user is shown for an accepted claim: ECE with every
     bin's count, and Brier score, before and after one fitted temperature.
     The temperature is fitted on the very claims it is then measured on,
     so the "after" numbers are optimistic; they say what one parameter can
     do on these claims, not what it would do on new ones.
  2. Risk-coverage tables: the verifier is re-run at each candidate
     entailment_threshold (the NLI results are cached, so this is cheap),
     and the claims kept at the current threshold are re-filtered at each
     candidate min_claim_score.
  3. With --retrieval: the best retrieval score (cosine similarity, which is
     what abstention.min_retrieval_score is compared with) for each tuning
     question, answerable vs unanswerable, and for a fixed list of
     off-topic and gibberish questions written for this script.

The held-out half is never loaded: there is deliberately no --split option.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from policyverify.calibration import (  # noqa: E402
    apply_temperature,
    brier,
    calibration_bins,
    ece,
    fit_temperature,
    format_bins,
    format_risk_coverage,
    risk_coverage,
)
from policyverify.claimset import ClaimLabel, evidence_for, load_claimset  # noqa: E402
from policyverify.config import Config, get_config  # noqa: E402
from policyverify.evalset import Split, load_evalset  # noqa: E402
from policyverify.indexing import IndexMismatchError, VectorStore  # noqa: E402
from policyverify.schema import Claim, VerdictStatus  # noqa: E402
from policyverify.verify import NLIChecker  # noqa: E402
from policyverify.verify.citation import check_citations  # noqa: E402
from policyverify.verify.verifier import premise_with_heading, verify_claim  # noqa: E402

ENTAILMENT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
CLAIM_SCORE_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]

# Questions the corpus cannot answer, written for this script (not taken
# from either eval split). Off-topic first, then gibberish. They exist only
# to show where such questions land on the retrieval-score scale.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the football world cup in 2018?",
    "What is the boiling point of water at sea level?",
    "Recommend a good laptop for gaming.",
    "How tall is Mount Everest?",
    "Write a poem about the ocean.",
    "What is the exchange rate between the dollar and the euro?",
]
GIBBERISH = [
    "asdf qwer zxcv",
    "blorp fizzle wum tango quux",
    "xkcd lorem ipsum dolor sit amet",
    "?",
    "purple monkey dishwasher seventeen",
    "qqqq wwww eeee rrrr",
]


class CachingNLIChecker(NLIChecker):
    """The real NLI checker, remembering every (unit, claim) pair it scored.

    Re-running the verifier at several thresholds would otherwise score the
    same sentences again each time. The model is deterministic, so a cached
    result is exactly the result a fresh call would give.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self._cache: dict[tuple[str, str], object] = {}

    def _score_many(self, premises, hypothesis):
        todo = [p for p in dict.fromkeys(premises) if (p, hypothesis) not in self._cache]
        if todo:
            for p, r in zip(todo, super()._score_many(todo, hypothesis), strict=True):
                self._cache[(p, hypothesis)] = r
        return [self._cache[(p, hypothesis)] for p in premises]


def _with_threshold(config: Config, threshold: float) -> Config:
    copy = config.model_copy(deep=True)
    copy.nli.entailment_threshold = threshold
    return copy


def _support_score(claim: Claim, evidence, nli, store) -> float:
    """Best entailment probability over every cited chunk, as the verifier
    computes it. This is the raw number entailment_threshold is applied to."""
    checks = check_citations(claim.citation_ids, evidence, store)
    scores = [
        nli.check(premise_with_heading(chunk), claim.text).entailment
        for check in checks
        if check.exists
        for chunk in check.chunks
    ]
    return max(scores, default=0.0)


def _calibration_block(title: str, scores, labels, bins: int) -> float:
    print(f"\n--- {title} ({len(scores)} claims, measured) ---")
    t = fit_temperature(scores, labels)
    scaled = [apply_temperature(s, t) for s in scores]
    print(f"before:  ECE {ece(scores, labels, bins):.3f}   Brier {brier(scores, labels):.3f}")
    print(format_bins(calibration_bins(scores, labels, bins)))
    print(f"fitted temperature T = {t:.3f}")
    print(f"after:   ECE {ece(scaled, labels, bins):.3f}   Brier {brier(scaled, labels):.3f}"
          "   (OPTIMISTIC: T fitted and measured on these same claims)")
    print(format_bins(calibration_bins(scaled, labels, bins)))
    return t


def calibrate_claims(config: Config, store, bins: int) -> None:
    items = [i for i in load_claimset() if i.split is Split.TUNING]
    labels = [i.label is ClaimLabel.SUPPORTED for i in items]
    print(f"{len(items)} tuning claims ({sum(labels)} truly supported, "
          f"{len(labels) - sum(labels)} not)")

    nli = CachingNLIChecker(config)
    started = time.perf_counter()
    claims = [Claim(text=i.claim, citation_ids=i.citation_ids) for i in items]
    evidence = [evidence_for(i, store) for i in items]

    # Verdicts at every candidate entailment threshold. The current value is
    # always included so the min_claim_score sweep below starts from it.
    thresholds = sorted(set(ENTAILMENT_THRESHOLDS) | {config.nli.entailment_threshold})
    verdicts = {}
    for t in thresholds:
        cfg = _with_threshold(config, t)
        verdicts[t] = [
            verify_claim(c, e, nli=nli, store=store, config=cfg)
            for c, e in zip(claims, evidence, strict=True)
        ]
    support = [_support_score(c, e, nli, store) for c, e in zip(claims, evidence, strict=True)]
    print(f"verifier run at {len(thresholds)} thresholds in {time.perf_counter() - started:.0f}s")

    current = verdicts[config.nli.entailment_threshold]

    # 1. Calibration.
    _calibration_block("NLI support score: best entailment over the cited section",
                       support, labels, bins)
    kept = [i for i, v in enumerate(current) if v.status is VerdictStatus.SUPPORTED]
    shown = [current[i].score for i in kept]
    if shown:
        _calibration_block(
            "confidence shown for ACCEPTED claims (verdict.score, current thresholds)",
            shown, [labels[i] for i in kept], bins,
        )

    # 2. Risk-coverage.
    print("\n--- risk-coverage: entailment_threshold (verifier re-run at each, measured) ---")
    print(format_risk_coverage(
        risk_coverage(
            thresholds, labels,
            lambda t: [v.status is VerdictStatus.SUPPORTED for v in verdicts[t]],
        ),
        "entail_t",
    ))
    print(f"\n--- risk-coverage: min_claim_score (entailment_threshold "
          f"{config.nli.entailment_threshold}, measured) ---")
    print(format_risk_coverage(
        risk_coverage(
            CLAIM_SCORE_THRESHOLDS, labels,
            lambda m: [v.status is VerdictStatus.SUPPORTED and v.score >= m for v in current],
        ),
        "min_score",
    ))

    wrong = [
        (items[i], current[i]) for i in range(len(items))
        if (current[i].status is VerdictStatus.SUPPORTED) != labels[i]
    ]
    if wrong:
        print("\nwrong verdicts at the current thresholds (id, kind, verdict, score, support score):")
        for item, v in wrong:
            j = items.index(item)
            print(f"  {item.id:<9} {item.kind.value:<14} {v.status.value:<9} "
                  f"{v.score:.3f}  {support[j]:.3f}")


def _best_score(question: str, config: Config, store) -> float:
    from policyverify.retrieve import retrieve

    results = retrieve(question, config=config, store=store)
    return max((r.score for r in results), default=0.0)


def _describe(name: str, scores: list[float]) -> str:
    if not scores:
        return f"  {name:<28} none"
    return (f"  {name:<28} n={len(scores):<3} min {min(scores):.3f}  "
            f"median {statistics.median(scores):.3f}  max {max(scores):.3f}")


def retrieval_scores(config: Config, store) -> None:
    items = [i for i in load_evalset() if i.split is Split.TUNING]
    answerable = [(i.id, _best_score(i.question, config, store)) for i in items if i.answerable]
    unanswerable = [(i.id, _best_score(i.question, config, store)) for i in items if not i.answerable]
    off = [(q, _best_score(q, config, store)) for q in OFF_TOPIC]
    gib = [(q, _best_score(q, config, store)) for q in GIBBERISH]

    print("\n--- best retrieval score per question (cosine similarity, measured) ---")
    print(f"current abstention.min_retrieval_score = {config.abstention.min_retrieval_score}")
    for name, rows in [("tuning answerable", answerable), ("tuning unanswerable", unanswerable),
                       ("off-topic (script list)", off), ("gibberish (script list)", gib)]:
        print(_describe(name, [s for _, s in rows]))
    print("\n  per question, lowest first:")
    tagged = ([("answerable", q, s) for q, s in answerable]
              + [("unanswerable", q, s) for q, s in unanswerable]
              + [("off-topic", q, s) for q, s in off]
              + [("gibberish", q, s) for q, s in gib])
    for kind, q, s in sorted(tagged, key=lambda r: r[2]):
        print(f"    {s:.3f}  {kind:<12} {q[:60]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration and thresholds on tuning data.")
    parser.add_argument("--bins", type=int, default=5, help="calibration bins (default 5)")
    parser.add_argument("--retrieval", action="store_true",
                        help="also measure best retrieval scores per question")
    parser.add_argument("--no-claims", action="store_true", help="skip the claim calibration")
    args = parser.parse_args()

    config = get_config()
    print("loading models ...")
    store = VectorStore(config)
    try:
        store.check_ready()
    except IndexMismatchError as exc:
        print(f"\n{exc}\n")
        return 1

    if not args.no_claims:
        calibrate_claims(config, store, args.bins)
    if args.retrieval:
        retrieval_scores(config, store)
    print("\nAll numbers above are from the TUNING half only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
