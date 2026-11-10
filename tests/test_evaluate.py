"""
Tests for the evaluation harness.

These check the scoring and aggregation logic against hand-built answers. No
model runs here: whether the system scores well is what the harness reports,
not something a unit test can assert.
"""

from __future__ import annotations

import pytest

from policyverify.evalset import EvalItem
from policyverify.evaluate import (
    ItemResult,
    Outcome,
    append_result,
    format_report,
    latest_run,
    load_results,
    rescore_without_verification,
    run_item,
    score_item,
    stratified_sample,
    summarise_results,
    without_verification,
)
from policyverify.schema import Answer, Claim, ClaimVerdict, Timings, VerdictStatus

RULE = "srm/attendance/7.3-minimum_attendance"
OTHER = "srm/scholarship/16-school_of_law"


def verdict(text: str, citation: str = RULE, status=VerdictStatus.SUPPORTED) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(text=text, citation_ids=[citation]),
        status=status,
        score=0.9 if status is VerdictStatus.SUPPORTED else 0.1,
    )


def answer(kept=(), removed=(), abstained=False) -> Answer:
    return Answer(
        question="q",
        claims_kept=list(kept),
        claims_removed=list(removed),
        abstained=abstained,
        reason="declined" if abstained else None,
        timings=Timings(retrieve_ms=100, generate_ms=2000, verify_ms=900),
    )


def item(**overrides) -> EvalItem:
    base = {
        "id": "num-001",
        "question": "What is the minimum attendance?",
        "category": "numeric",
        "split": "tuning",
        "answerable": True,
        "expected_citations": [RULE],
        "required_values": ["75"],
        "forbidden_values": ["80%"],
    }
    base.update(overrides)
    return EvalItem.model_validate(base)


def unanswerable(**overrides) -> EvalItem:
    fields = {
        "id": "una-001",
        "category": "unanswerable",
        "answerable": False,
        "expected_citations": [],
        "required_values": [],
        "forbidden_values": [],
    }
    fields.update(overrides)
    return item(**fields)


# --- score_item -------------------------------------------------------------


def test_correct_answer():
    result = score_item(item(), answer([verdict("Students need 75% attendance.")]))
    assert result.outcome is Outcome.CORRECT
    assert result.passed
    assert result.kept_claims_on_expected == 1


def test_forbidden_value_in_kept_claim_is_wrong_value():
    result = score_item(item(), answer([verdict("Students need 80% attendance.")]))
    assert result.outcome is Outcome.WRONG_VALUE
    assert result.forbidden_found == ["80%"]


def test_forbidden_value_in_removed_claim_is_not_counted():
    # The verifier caught it. That is the system working, not a hallucination.
    result = score_item(
        item(),
        answer(
            kept=[verdict("Students need 75% attendance.")],
            removed=[verdict("Students need 80% attendance.", status=VerdictStatus.REFUTED)],
        ),
    )
    assert result.outcome is Outcome.CORRECT
    assert result.forbidden_found == []


def test_forbidden_value_outranks_a_wrong_citation():
    result = score_item(item(), answer([verdict("Students need 80% attendance.", OTHER)]))
    assert result.outcome is Outcome.WRONG_VALUE


def test_answer_resting_on_the_wrong_section():
    result = score_item(item(), answer([verdict("A 75% tuition waiver applies.", OTHER)]))
    assert result.outcome is Outcome.WRONG_CITATION
    assert result.cited == [OTHER]


def test_right_section_but_required_value_missing():
    result = score_item(item(), answer([verdict("Students must attend classes.")]))
    assert result.outcome is Outcome.INCOMPLETE
    assert result.required_missing == ["75"]


def test_required_value_is_matched_as_a_whole_token():
    # "7" must not be satisfied by the 7 in "75".
    result = score_item(
        item(required_values=["7"], forbidden_values=[]),
        answer([verdict("Students need 75% attendance.")]),
    )
    assert result.outcome is Outcome.INCOMPLETE


def test_abstaining_on_an_answerable_item():
    result = score_item(item(), answer(abstained=True))
    assert result.outcome is Outcome.FALSE_ABSTENTION
    # Nothing was asserted, so nothing is reported missing.
    assert result.required_missing == []


def test_abstaining_on_an_unanswerable_item():
    result = score_item(unanswerable(), answer(abstained=True))
    assert result.outcome is Outcome.CORRECT_ABSTENTION
    assert result.passed


def test_answering_an_unanswerable_item():
    result = score_item(unanswerable(), answer([verdict("The mess serves food.")]))
    assert result.outcome is Outcome.ANSWERED_UNANSWERABLE
    assert not result.passed


def test_timings_are_carried_over():
    result = score_item(item(), answer([verdict("Students need 75% attendance.")]))
    assert (result.retrieve_ms, result.generate_ms, result.verify_ms) == (100, 2000, 900)
    assert result.total_ms == 3000


# --- run_item ---------------------------------------------------------------


def test_run_item_scores_what_the_pipeline_returned():
    result = run_item(item(), lambda q: answer([verdict("Students need 75% attendance.")]))
    assert result.outcome is Outcome.CORRECT


def test_run_item_records_a_failure_instead_of_raising():
    def broken(question: str) -> Answer:
        raise RuntimeError("ollama is not running")

    result = run_item(item(), broken)
    assert result.outcome is Outcome.ERROR
    assert "ollama is not running" in result.error
    assert not result.passed


# --- summarise_results ------------------------------------------------------


def results_fixture() -> list[ItemResult]:
    return [
        # answerable, correct
        score_item(item(id="a"), answer([verdict("Students need 75% attendance.")])),
        # answerable, hallucinated a forbidden figure
        score_item(item(id="b"), answer([verdict("Students need 80% attendance.")])),
        # answerable, refused
        score_item(item(id="c"), answer(abstained=True)),
        # unanswerable, correctly refused
        score_item(unanswerable(id="d"), answer(abstained=True)),
        # unanswerable, answered anyway
        score_item(unanswerable(id="e"), answer([verdict("The mess serves food.")])),
    ]


def test_summary_headline_rates():
    summary = summarise_results(results_fixture())
    assert summary["items"] == 5
    assert summary["accuracy"] == pytest.approx(2 / 5)
    # b (wrong value) and e (answered unanswerable)
    assert summary["hallucination_rate"] == pytest.approx(2 / 5)
    assert summary["abstention"]["correct_abstention_rate"] == pytest.approx(1 / 2)
    assert summary["abstention"]["false_abstention_rate"] == pytest.approx(1 / 3, abs=1e-4)
    # a and b were answered; both rest on the expected section
    assert summary["citation_accuracy"] == pytest.approx(1.0)


def test_summary_claim_counts():
    results = [
        score_item(
            item(id="a"),
            answer(
                kept=[verdict("Students need 75% attendance."), verdict("Waiver.", OTHER)],
                removed=[verdict("Fine of 500.", status=VerdictStatus.NEUTRAL)],
            ),
        )
    ]
    claims = summarise_results(results)["claims"]
    assert (claims["generated"], claims["kept"], claims["removed"]) == (3, 2, 1)
    assert claims["removal_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert claims["citation_precision"] == pytest.approx(1 / 2)
    assert claims["citation_recall"] == pytest.approx(1.0)


def test_summary_by_category():
    by_category = summarise_results(results_fixture())["by_category"]
    assert by_category["numeric"] == {"items": 3, "passed": 1, "accuracy": pytest.approx(1 / 3, abs=1e-4)}
    assert by_category["unanswerable"]["passed"] == 1


def test_errors_are_excluded_from_rates_but_counted():
    results = results_fixture() + [run_item(item(id="f"), lambda q: 1 / 0)]
    summary = summarise_results(results)
    assert summary["items"] == 6
    assert summary["scored"] == 5
    assert summary["errors"] == 1
    assert summary["accuracy"] == pytest.approx(2 / 5)


def test_empty_denominators_are_none_not_zero():
    # Only answerable items: there is no abstention-on-unanswerable rate to report.
    summary = summarise_results(
        [score_item(item(), answer([verdict("Students need 75% attendance.")]))]
    )
    assert summary["abstention"]["correct_abstention_rate"] is None
    assert summarise_results([])["accuracy"] is None


def test_latency_summary():
    latency = summarise_results(results_fixture())["latency"]
    assert latency["total"]["mean_ms"] == 3000
    assert latency["generate"]["median_ms"] == 2000


# --- sampling, logs, report -------------------------------------------------


def test_stratified_sample_spreads_across_categories():
    items = [item(id=f"n{i}") for i in range(5)] + [unanswerable(id=f"u{i}") for i in range(5)]
    picked = stratified_sample(items, 4)
    assert [i.id for i in picked] == ["n0", "u0", "n1", "u1"]


def test_stratified_sample_returns_everything_when_n_is_large_or_zero():
    items = [item(id="a"), unanswerable(id="b")]
    assert stratified_sample(items, 0) == items
    assert stratified_sample(items, 99) == items


def test_run_log_round_trips(tmp_path):
    log = tmp_path / "results" / "run-20260101-000000.jsonl"
    for result in results_fixture():
        append_result(log, result)

    loaded = load_results(log)
    assert [r.id for r in loaded] == ["a", "b", "c", "d", "e"]
    assert loaded[1].outcome is Outcome.WRONG_VALUE
    assert loaded[0].answer.claims_kept[0].claim.text == "Students need 75% attendance."
    assert summarise_results(loaded) == summarise_results(results_fixture())


def test_latest_run_picks_the_newest_log(tmp_path):
    assert latest_run(tmp_path) is None
    (tmp_path / "run-20260101-000000.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "run-20260301-000000.jsonl").write_text("", encoding="utf-8")
    assert latest_run(tmp_path).name == "run-20260301-000000.jsonl"


def test_report_lists_failures_and_handles_missing_rates():
    results = results_fixture()
    report = format_report(summarise_results(results), results)
    assert "hallucination rate" in report
    assert "wrong_value" in report
    assert "n/a" not in report

    answerable_only = [results[0]]
    assert "n/a" in format_report(summarise_results(answerable_only), answerable_only)


# --- the no-verification baseline -------------------------------------------


def test_without_verification_keeps_every_generated_claim():
    original = answer(
        kept=[verdict("Students need 75% attendance.")],
        removed=[verdict("Students need 80% attendance.", status=VerdictStatus.REFUTED)],
    )
    baseline = without_verification(original)
    assert [v.claim.text for v in baseline.claims_kept] == [
        "Students need 75% attendance.",
        "Students need 80% attendance.",
    ]
    assert baseline.claims_removed == []
    assert not baseline.abstained
    # The original is untouched.
    assert len(original.claims_removed) == 1


def test_without_verification_un_abstains_when_claims_were_generated():
    declined = answer(
        removed=[verdict("The mess costs 5000.", status=VerdictStatus.NEUTRAL)], abstained=True
    )
    assert not without_verification(declined).abstained


def test_without_verification_still_declines_when_the_model_said_nothing():
    assert without_verification(answer(abstained=True)).abstained


def test_rescoring_shows_what_verification_caught():
    labelled = item()
    # The verifier removed the wrong figure, so the system was correct ...
    results = [
        score_item(
            labelled,
            answer(
                kept=[verdict("Students need 75% attendance.")],
                removed=[verdict("Students need 80% attendance.", status=VerdictStatus.REFUTED)],
            ),
        )
    ]
    assert results[0].outcome is Outcome.CORRECT
    # ... and without it, the same generations would have been a hallucination.
    baseline = rescore_without_verification(results, [labelled])
    assert baseline[0].outcome is Outcome.WRONG_VALUE


def test_rescoring_passes_errors_and_unknown_items_through():
    errored = run_item(item(id="x"), lambda q: 1 / 0)
    orphan = score_item(item(id="gone"), answer([verdict("Students need 75% attendance.")]))
    assert rescore_without_verification([errored, orphan], []) == [errored, orphan]
