"""
Tests for the labelled claim set and its scoring.

Like every test here, these need no model, no index and no network: scoring
takes the verifier's verdicts as plain input, so canned verdicts are enough.
"""

from __future__ import annotations

import json

import pytest

from policyverify.claimset import (
    ClaimItem,
    ClaimKind,
    evidence_for,
    format_claim_report,
    load_claimset,
    missing_citations,
    score_claims,
)
from policyverify.evaluate import format_comparison, wilson_interval
from policyverify.schema import Chunk, PolicyType, VerdictStatus

CITE = "srm/attendance/7.3-minimum_attendance"
S, N, R = VerdictStatus.SUPPORTED, VerdictStatus.NEUTRAL, VerdictStatus.REFUTED


def make(id: str, kind: str, split: str = "tuning", citation: str = CITE) -> ClaimItem:
    supported = kind in ("paraphrase", "verbatim")
    return ClaimItem(
        id=id,
        claim=f"claim {id}",
        citation_ids=[citation],
        label="supported" if supported else "not_supported",
        kind=kind,
        split=split,
    )


class StubStore:
    """Knows about exactly one section."""

    def get_by_citation(self, citation_id: str) -> list[Chunk]:
        if citation_id != CITE:
            return []
        return [
            Chunk(
                chunk_id=f"{CITE}#{i}",
                text=f"piece {i}",
                citation_id=CITE,
                university="srm",
                university_name="SRM",
                policy_type=PolicyType.ATTENDANCE,
                section_path="Minimum Attendance",
                source_url="https://example.invalid/",
            )
            for i in range(2)
        ]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_kind_must_agree_with_label():
    with pytest.raises(ValueError, match="implies label"):
        ClaimItem(
            id="x", claim="c", citation_ids=[CITE], label="supported", kind="number",
            split="tuning",
        )


def test_malformed_citation_is_rejected():
    with pytest.raises(ValueError):
        ClaimItem(
            id="x", claim="c", citation_ids=["not a citation"], label="supported",
            kind="paraphrase", split="tuning",
        )


def test_blank_claim_and_missing_citation_are_rejected():
    with pytest.raises(ValueError):
        ClaimItem(id="x", claim="  ", citation_ids=[CITE], label="supported",
                  kind="paraphrase", split="tuning")
    with pytest.raises(ValueError):
        ClaimItem(id="x", claim="c", citation_ids=[], label="supported",
                  kind="paraphrase", split="tuning")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _line(**overrides) -> str:
    item = {"id": "par-001", "claim": "c", "citation_ids": [CITE], "label": "supported",
            "kind": "paraphrase", "split": "tuning"}
    item.update(overrides)
    return json.dumps(item)


def test_loader_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text(f"# a comment\n\n{_line()}\n{_line(id='par-002')}\n", encoding="utf-8")
    assert [i.id for i in load_claimset(path)] == ["par-001", "par-002"]


def test_loader_reports_the_line_of_a_bad_item(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text(f"{_line()}\n{_line(id='x', kind='number')}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_claimset(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "claims.jsonl"
    path.write_text(f"{_line()}\n{_line()}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_claimset(path)


def test_the_shipped_claim_set_is_valid_and_balanced():
    """The real file loads, and both halves cover every kind of claim."""
    items = load_claimset()
    assert len(items) >= 80
    for split in ("tuning", "heldout"):
        kinds = {i.kind for i in items if i.split.value == split}
        assert kinds == set(ClaimKind), f"{split} half is missing a kind"


# ---------------------------------------------------------------------------
# Evidence and existence
# ---------------------------------------------------------------------------


def test_evidence_is_every_chunk_of_the_cited_section_in_order():
    retrieved = evidence_for(make("a", "paraphrase"), StubStore())
    assert [r.chunk.chunk_id for r in retrieved] == [f"{CITE}#0", f"{CITE}#1"]
    assert [r.rank for r in retrieved] == [0, 1]


def test_missing_citations_are_named():
    gone = "srm/attendance/99-removed"
    problems = missing_citations(
        [make("a", "paraphrase"), make("b", "number", citation=gone)], StubStore()
    )
    assert problems == [f"b: cited section {gone} is not in the index"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_precision_recall_and_error_rates():
    items = [
        make("t1", "paraphrase"), make("t2", "paraphrase"), make("t3", "verbatim"),
        make("t4", "verbatim"),
        make("f1", "number"), make("f2", "role"), make("f3", "unsupported"),
        make("f4", "negation"),
    ]
    statuses = {
        "t1": S, "t2": S, "t3": S, "t4": R,   # one true claim thrown away
        "f1": R, "f2": S, "f3": N, "f4": R,   # one false claim let through
    }
    summary = score_claims(items, statuses)

    assert summary["confusion"] == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}
    assert summary["precision"] == pytest.approx(0.75)
    assert summary["recall"] == pytest.approx(0.75)
    assert summary["f1"] == pytest.approx(0.75)
    assert summary["false_accept_rate"] == pytest.approx(0.25)
    assert summary["false_reject_rate"] == pytest.approx(0.25)
    assert summary["wrong"] == ["f2", "t4"]


def test_neutral_and_refuted_both_count_as_not_accepted():
    items = [make("f1", "unsupported"), make("f2", "number")]
    summary = score_claims(items, {"f1": N, "f2": R})
    assert summary["confusion"]["tn"] == 2
    assert summary["false_accept_rate"] == 0.0


def test_results_are_broken_down_by_kind():
    items = [make("r1", "role"), make("r2", "role"), make("v1", "verbatim")]
    summary = score_claims(items, {"r1": S, "r2": R, "v1": N})
    assert summary["by_kind"]["role"]["correct"] == 1
    assert summary["by_kind"]["role"]["verdicts"] == {"SUPPORTED": 1, "NEUTRAL": 0, "REFUTED": 1}
    # A rejected verbatim quote is a wrong verdict, not a correct rejection.
    assert summary["by_kind"]["verbatim"]["correct"] == 0
    assert "number" not in summary["by_kind"]


def test_contradiction_detection_counts_only_conflicting_kinds():
    items = [make("n1", "number"), make("g1", "negation"), make("u1", "unsupported")]
    summary = score_claims(items, {"n1": R, "g1": N, "u1": N})
    # "unsupported" is excluded: the section is silent, so NEUTRAL is right.
    assert summary["contradiction_detection"] == {"items": 2, "refuted": 1, "rate": 0.5}


def test_claims_without_a_verdict_are_left_out_not_counted_wrong():
    items = [make("t1", "paraphrase"), make("t2", "paraphrase")]
    summary = score_claims(items, {"t1": S})
    assert summary["items"] == 1
    assert summary["recall"] == 1.0


def test_nothing_accepted_gives_no_precision_rather_than_zero():
    summary = score_claims([make("f1", "number")], {"f1": R})
    assert summary["precision"] is None
    assert summary["f1"] is None


def test_report_names_the_wrong_verdicts():
    items = [make("t1", "verbatim"), make("f1", "role")]
    statuses = {"t1": R, "f1": S}
    report = format_claim_report(score_claims(items, statuses), items, statuses)
    assert "precision" in report and "recall" in report
    assert "wrong verdicts" in report
    assert "t1" in report and "f1" in report


# ---------------------------------------------------------------------------
# Confidence intervals and the on/off comparison (evaluate.py)
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_a_known_value():
    # 20 of 31, the first held-out result: the interval is wide, which is
    # the whole reason it is printed.
    low, high = wilson_interval(20, 31)
    assert low == pytest.approx(0.469, abs=0.005)
    assert high == pytest.approx(0.789, abs=0.005)


def test_wilson_interval_stays_inside_zero_and_one():
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0
    assert 0.0 < wilson_interval(0, 10)[1] < 0.5


def test_wilson_interval_of_nothing_is_none():
    assert wilson_interval(0, 0) is None


def test_comparison_shows_both_columns_and_the_change():
    on = {"accuracy": 0.6, "hallucination_rate": 0.2, "citation_accuracy": 0.9,
          "abstention": {"correct_abstention_rate": 0.5, "false_abstention_rate": 0.1}}
    off = {"accuracy": 0.6, "hallucination_rate": 0.3, "citation_accuracy": None,
           "abstention": {"correct_abstention_rate": 0.25, "false_abstention_rate": 0.0}}
    text = format_comparison(on, off)
    assert "+0.0 pts" in text          # accuracy unchanged is stated, not hidden
    assert "-10.0 pts" in text         # hallucination fell
    assert "+25.0 pts" in text
    assert "n/a" in text               # a missing value does not crash the table
