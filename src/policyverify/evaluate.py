"""
evaluate.py - scoring the system against the adversarial set (Phase 8).

WHAT THIS MEASURES
    Each adversarial item carries labels fixed in advance (see evalset.py):
    whether the corpus covers the question, which sections a correct answer
    must rest on, and which figures must or must not appear. This module runs
    a question through `pipeline.answer_question()` - the same entry point the
    CLI and the UI use - and compares the `Answer` to those labels.

THE METRICS, AND WHAT EACH ONE CAN HONESTLY CLAIM
    abstention quality   Did the system decline exactly the questions the
                         corpus does not cover? Reported as two numbers,
                         because they fail in opposite directions: answering
                         an unanswerable question, and refusing an answerable
                         one. A system that refuses everything is perfect on
                         the first and useless on the second.

    citation accuracy    Of the answerable questions it answered, how often a
                         kept claim rests on a section the labels expect.

    hallucination rate   How often the system STOOD BEHIND something it should
                         not have: a forbidden figure in a kept claim, or any
                         answer at all to an unanswerable question. Removed
                         claims do not count - catching them is the system
                         working.

    claim precision /    Claims are generated fresh each run, so they cannot be
    recall (by citation) labelled in advance. What can be scored automatically
                         is where they point: precision is the share of kept
                         claims citing an expected section, recall the share of
                         expected sections that a kept claim cites. This is a
                         proxy. True claim-level precision needs a human to
                         read each claim, and nothing here pretends otherwise.

    latency              Per stage, so "slow" comes with a location.

Nothing in this file loads a model. `run_item()` takes the pipeline function
as an argument, so the scoring logic is tested with canned answers.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from policyverify.evalset import EvalItem, _mentions
from policyverify.schema import Answer


class Outcome(StrEnum):
    """What happened on one item, most serious failure first."""

    # --- unanswerable items ---------------------------------------------
    CORRECT_ABSTENTION = "correct_abstention"
    ANSWERED_UNANSWERABLE = "answered_unanswerable"

    # --- answerable items -----------------------------------------------
    CORRECT = "correct"
    # A figure the labels mark as wrong appears in a claim the system kept.
    WRONG_VALUE = "wrong_value"
    # Answered, but no kept claim rests on an expected section.
    WRONG_CITATION = "wrong_citation"
    # Right section, but a figure a correct answer must contain is missing.
    INCOMPLETE = "incomplete"
    FALSE_ABSTENTION = "false_abstention"

    # The pipeline raised - model unreachable, unparseable output, ...
    ERROR = "error"


PASSING = {Outcome.CORRECT, Outcome.CORRECT_ABSTENTION}
# Outcomes where the system stood behind something it should not have.
HALLUCINATIONS = {Outcome.WRONG_VALUE, Outcome.ANSWERED_UNANSWERABLE}


class ItemResult(BaseModel):
    """One adversarial item, scored. One line of the JSONL run log."""

    id: str
    question: str
    category: str
    split: str
    answerable: bool
    outcome: Outcome

    abstained: bool = False
    claims_kept: int = 0
    claims_removed: int = 0

    # Citation scoring. Only meaningful for answerable items that were answered.
    cited: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)
    kept_claims_on_expected: int = 0

    required_missing: list[str] = Field(default_factory=list)
    forbidden_found: list[str] = Field(default_factory=list)

    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    verify_ms: float = 0.0

    error: str | None = None
    # The full answer, so a surprising score can be read rather than re-run.
    answer: Answer | None = None

    @property
    def passed(self) -> bool:
        return self.outcome in PASSING

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.generate_ms + self.verify_ms


def score_item(item: EvalItem, answer: Answer) -> ItemResult:
    """Compare one `Answer` to the labels on its item. Pure - no I/O."""
    kept_texts = [v.claim.text for v in answer.claims_kept]
    combined = "\n".join(kept_texts)
    cited = answer.all_citations()
    expected = set(item.expected_citations)

    on_expected = sum(
        1 for v in answer.claims_kept if expected.intersection(v.claim.citation_ids)
    )
    # Only what the system kept is scored. A forbidden figure in a REMOVED
    # claim is the verifier doing its job, not a hallucination.
    forbidden_found = [v for v in item.forbidden_values if _mentions(v, combined)]
    required_missing = [v for v in item.required_values if not _mentions(v, combined)]

    if answer.abstained:
        outcome = Outcome.FALSE_ABSTENTION if item.answerable else Outcome.CORRECT_ABSTENTION
        # Nothing was asserted, so there is nothing to find or to miss.
        forbidden_found, required_missing = [], []
    elif not item.answerable:
        outcome = Outcome.ANSWERED_UNANSWERABLE
    elif forbidden_found:
        outcome = Outcome.WRONG_VALUE
    elif on_expected == 0:
        outcome = Outcome.WRONG_CITATION
    elif required_missing:
        outcome = Outcome.INCOMPLETE
    else:
        outcome = Outcome.CORRECT

    return ItemResult(
        id=item.id,
        question=item.question,
        category=item.category.value,
        split=item.split.value,
        answerable=item.answerable,
        outcome=outcome,
        abstained=answer.abstained,
        claims_kept=len(answer.claims_kept),
        claims_removed=len(answer.claims_removed),
        cited=cited,
        expected_citations=list(item.expected_citations),
        kept_claims_on_expected=on_expected,
        required_missing=required_missing,
        forbidden_found=forbidden_found,
        retrieve_ms=answer.timings.retrieve_ms,
        generate_ms=answer.timings.generate_ms,
        verify_ms=answer.timings.verify_ms,
        answer=answer,
    )


def run_item(item: EvalItem, answer_fn: Callable[[str], Answer]) -> ItemResult:
    """Run one item through the pipeline and score it.

    A pipeline failure is recorded as an ERROR result rather than raised, so
    one bad question cannot throw away an hour-long run.
    """
    try:
        answer = answer_fn(item.question)
    except Exception as exc:  # noqa: BLE001 - any failure is a result here
        return ItemResult(
            id=item.id,
            question=item.question,
            category=item.category.value,
            split=item.split.value,
            answerable=item.answerable,
            outcome=Outcome.ERROR,
            expected_citations=list(item.expected_citations),
            error=f"{type(exc).__name__}: {exc}",
        )
    return score_item(item, answer)


# ---------------------------------------------------------------------------
# The baseline: the same answers with verification switched off
# ---------------------------------------------------------------------------


def without_verification(answer: Answer) -> Answer:
    """`answer` as it would have been if nothing had been checked.

    Every claim the model generated is kept, and the system declines only
    when the model itself produced no claims. This is what a plain
    retrieve-then-generate system would have shown the user.

    It costs nothing to compute, because removed claims are kept in the
    `Answer` rather than deleted - so one run yields both the system's score
    and the score of the same generations without the verifier. The two are
    therefore compared on identical model output, not on two separate runs.
    """
    everything = list(answer.claims_kept) + list(answer.claims_removed)
    return answer.model_copy(
        update={
            "claims_kept": everything,
            "claims_removed": [],
            "abstained": not everything,
            "reason": None if everything else "The model made no claims.",
        }
    )


def rescore_without_verification(
    results: list[ItemResult], items: list[EvalItem]
) -> list[ItemResult]:
    """Re-score a finished run as the no-verification baseline.

    Items that errored, or whose labels are no longer in the set, are passed
    through unchanged.
    """
    by_id = {item.id: item for item in items}
    rescored: list[ItemResult] = []
    for result in results:
        item = by_id.get(result.id)
        if result.answer is None or item is None:
            rescored.append(result)
        else:
            rescored.append(score_item(item, without_verification(result.answer)))
    return rescored


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _rate(numerator: int, denominator: int) -> float | None:
    """A proportion, or None when there is nothing to take a proportion of.

    None rather than 0.0 on purpose: "0% hallucination over zero items" reads
    as a result and is not one.
    """
    return round(numerator / denominator, 4) if denominator else None


def _latency(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {
        "mean_ms": round(statistics.fmean(ordered), 1),
        "median_ms": round(statistics.median(ordered), 1),
        "p95_ms": round(p95, 1),
    }


def summarise_results(results: list[ItemResult]) -> dict:
    """The headline numbers for one run."""
    scored = [r for r in results if r.outcome is not Outcome.ERROR]
    answerable = [r for r in scored if r.answerable]
    unanswerable = [r for r in scored if not r.answerable]
    answered = [r for r in answerable if not r.abstained]

    kept = sum(r.claims_kept for r in scored)
    removed = sum(r.claims_removed for r in scored)

    kept_on_answered = sum(r.claims_kept for r in answered)
    kept_on_expected = sum(r.kept_claims_on_expected for r in answered)
    expected_total = sum(len(r.expected_citations) for r in answerable)
    expected_cited = sum(
        len(set(r.expected_citations).intersection(r.cited)) for r in answered
    )

    by_category: dict[str, dict] = {}
    for category in sorted({r.category for r in scored}):
        rows = [r for r in scored if r.category == category]
        by_category[category] = {
            "items": len(rows),
            "passed": sum(1 for r in rows if r.passed),
            "accuracy": _rate(sum(1 for r in rows if r.passed), len(rows)),
        }

    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r.outcome.value] = outcomes.get(r.outcome.value, 0) + 1

    return {
        "items": len(results),
        "scored": len(scored),
        "errors": len(results) - len(scored),
        "accuracy": _rate(sum(1 for r in scored if r.passed), len(scored)),
        "abstention": {
            "unanswerable_items": len(unanswerable),
            "correct_abstention_rate": _rate(
                sum(1 for r in unanswerable if r.abstained), len(unanswerable)
            ),
            "answerable_items": len(answerable),
            "false_abstention_rate": _rate(
                sum(1 for r in answerable if r.abstained), len(answerable)
            ),
        },
        "citation_accuracy": _rate(
            sum(1 for r in answered if r.kept_claims_on_expected > 0), len(answered)
        ),
        "hallucination_rate": _rate(
            sum(1 for r in scored if r.outcome in HALLUCINATIONS), len(scored)
        ),
        "claims": {
            "generated": kept + removed,
            "kept": kept,
            "removed": removed,
            "removal_rate": _rate(removed, kept + removed),
            # Proxies by citation - see the module docstring.
            "citation_precision": _rate(kept_on_expected, kept_on_answered),
            "citation_recall": _rate(expected_cited, expected_total),
        },
        "latency": {
            "retrieve": _latency([r.retrieve_ms for r in scored]),
            "generate": _latency([r.generate_ms for r in scored]),
            "verify": _latency([r.verify_ms for r in scored]),
            "total": _latency([r.total_ms for r in scored]),
        },
        "outcomes": outcomes,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# Choosing items
# ---------------------------------------------------------------------------


def stratified_sample(items: list[EvalItem], n: int) -> list[EvalItem]:
    """The first `n` items taken round-robin across categories.

    A plain `items[:n]` would be all one category, because the file is
    grouped by category. Deterministic - no randomness - so two runs of the
    same sample size score the same questions.
    """
    if n <= 0 or n >= len(items):
        return list(items)

    queues: dict[str, list[EvalItem]] = {}
    for item in items:
        queues.setdefault(item.category.value, []).append(item)

    picked: list[EvalItem] = []
    while len(picked) < n and any(queues.values()):
        for queue in queues.values():
            if queue and len(picked) < n:
                picked.append(queue.pop(0))
    return picked


# ---------------------------------------------------------------------------
# Run logs
# ---------------------------------------------------------------------------


def new_run_id() -> str:
    return datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")


def append_result(path: Path, result: ItemResult) -> None:
    """Append one result to a JSONL log.

    Written item by item rather than at the end, so an interrupted run keeps
    everything it had finished.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(result.model_dump_json() + "\n")


def load_results(path: Path | str) -> list[ItemResult]:
    """Read a JSONL run log back."""
    results: list[ItemResult] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            results.append(ItemResult.model_validate_json(line))
    return results


def save_summary(path: Path, summary: dict, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"meta": meta or {}, "summary": summary}, indent=2), encoding="utf-8"
    )
    return path


def latest_run(results_dir: Path | str) -> Path | None:
    """The most recent JSONL run log in `results_dir`, if any."""
    logs = sorted(Path(results_dir).glob("run-*.jsonl"))
    return logs[-1] if logs else None


def format_report(summary: dict, results: Iterable[ItemResult] = ()) -> str:
    """A plain-text report for the terminal."""

    def pct(value: float | None) -> str:
        return "   n/a" if value is None else f"{value * 100:5.1f}%"

    a, c = summary["abstention"], summary["claims"]
    lines = [
        f"items scored          {summary['scored']} of {summary['items']}"
        + (f"  ({summary['errors']} errored)" if summary["errors"] else ""),
        f"overall accuracy      {pct(summary['accuracy'])}",
        "",
        f"hallucination rate    {pct(summary['hallucination_rate'])}   "
        "(stood behind a wrong figure, or answered an unanswerable question)",
        f"citation accuracy     {pct(summary['citation_accuracy'])}   "
        "(answered questions resting on an expected section)",
        f"correct abstention    {pct(a['correct_abstention_rate'])}   "
        f"(of {a['unanswerable_items']} unanswerable)",
        f"false abstention      {pct(a['false_abstention_rate'])}   "
        f"(of {a['answerable_items']} answerable)",
        "",
        f"claims generated      {c['generated']}  ->  {c['kept']} kept, {c['removed']} removed "
        f"({pct(c['removal_rate']).strip()} removed by verification)",
        f"citation precision    {pct(c['citation_precision'])}   (kept claims citing an expected section)",
        f"citation recall       {pct(c['citation_recall'])}   (expected sections cited by a kept claim)",
        "",
        "by category",
    ]
    for name, row in summary["by_category"].items():
        lines.append(
            f"  {name:<20} {row['passed']:>2}/{row['items']:<2}  {pct(row['accuracy'])}"
        )

    lines += ["", "latency (ms)          mean    median       p95"]
    for stage in ("retrieve", "generate", "verify", "total"):
        row = summary["latency"][stage]
        if row:
            lines.append(
                f"  {stage:<12} {row['mean_ms']:>12.0f} {row['median_ms']:>9.0f} "
                f"{row['p95_ms']:>9.0f}"
            )

    failures = [r for r in results if not r.passed]
    if failures:
        lines += ["", "failures"]
        for r in failures:
            detail = r.error or ", ".join(
                part
                for part in (
                    f"forbidden {r.forbidden_found}" if r.forbidden_found else "",
                    f"missing {r.required_missing}" if r.required_missing else "",
                    f"cited {r.cited}" if r.outcome is Outcome.WRONG_CITATION else "",
                )
                if part
            )
            lines.append(f"  {r.id:<10} {r.outcome.value:<22} {detail}")
    return "\n".join(lines)
