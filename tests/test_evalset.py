"""
Tests for the adversarial question set: its schema, and the real file.

Two kinds of test here. Most check the schema rejects incoherent labels. The
last group loads the actual `eval/adversarial.jsonl` and asserts properties
the set must hold to produce meaningful numbers - enough items, both halves
covering every trap, and enough unanswerable questions to measure false
abstention. Those would otherwise only be caught by noticing a result looked
odd, long after the fact.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from policyverify.evalset import (
    DEFAULT_EVALSET_PATH,
    EvalItem,
    Split,
    TrapCategory,
    _mentions,
    check_expected_content,
    compare_fingerprints,
    content_check_coverage,
    fingerprint_citations,
    load_evalset,
    load_fingerprints,
    save_fingerprints,
    split_items,
    summarise,
    validate_against_corpus,
)

VALID_CITATION = "srm/attendance/7.3-minimum_attendance"


def _item(**overrides) -> dict:
    base = {
        "id": "num-999",
        "question": "What is the minimum attendance requirement?",
        "category": "numeric",
        "split": "tuning",
        "answerable": True,
        "expected_citations": [VALID_CITATION],
        "required_values": ["75"],
        "forbidden_values": ["80%"],
        "note": "test item",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_a_well_formed_item_validates():
    item = EvalItem.model_validate(_item())
    assert item.category is TrapCategory.NUMERIC
    assert item.split is Split.TUNING
    assert item.answerable


def test_malformed_citations_are_rejected():
    """A citation the system could never produce would make the item
    unscoreable."""
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(expected_citations=["not-a-citation"]))


def test_unanswerable_item_may_not_expect_citations():
    """If the corpus does not cover it, there is nothing correct to cite -
    an item claiming otherwise is mislabelled."""
    with pytest.raises(ValidationError, match="cannot expect citations"):
        EvalItem.model_validate(
            _item(answerable=False, expected_citations=[VALID_CITATION])
        )


def test_unanswerable_item_may_not_require_values():
    with pytest.raises(ValidationError, match="cannot require values"):
        EvalItem.model_validate(
            _item(answerable=False, expected_citations=[], required_values=["75"])
        )


def test_answerable_item_needs_at_least_one_expected_citation():
    """Otherwise citation accuracy cannot be scored for it."""
    with pytest.raises(ValidationError, match="at least one expected citation"):
        EvalItem.model_validate(_item(expected_citations=[]))


def test_unanswerable_item_is_valid_with_forbidden_values():
    """Forbidden values still make sense when nothing is answerable - they
    name the figures the system must not invent."""
    item = EvalItem.model_validate(
        _item(
            id="fp-999",
            category="false_premise",
            answerable=False,
            expected_citations=[],
            required_values=[],
            forbidden_values=["5000"],
        )
    )
    assert item.forbidden_values == ["5000"]


def test_blank_question_is_rejected():
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(question="   "))


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        EvalItem.model_validate(_item(category="made_up_category"))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        "# a comment\n\n" + json.dumps(_item()) + "\n\n# another\n",
        encoding="utf-8",
    )
    assert len(load_evalset(path)) == 1


def test_load_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "set.jsonl"
    line = json.dumps(_item())
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate id"):
        load_evalset(path)


def test_load_reports_the_offending_line_number(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps(_item()) + "\n" + json.dumps(_item(id="x", question="")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        load_evalset(path)


def test_load_raises_a_clear_error_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_evalset(tmp_path / "nope.jsonl")


def test_split_items_filters_by_half(tmp_path):
    path = tmp_path / "set.jsonl"
    path.write_text(
        json.dumps(_item(id="a", split="tuning"))
        + "\n"
        + json.dumps(_item(id="b", split="heldout"))
        + "\n",
        encoding="utf-8",
    )
    items = load_evalset(path)
    assert [i.id for i in split_items(items, Split.TUNING)] == ["a"]
    assert [i.id for i in split_items(items, Split.HELDOUT)] == ["b"]


# ---------------------------------------------------------------------------
# Corpus validation
# ---------------------------------------------------------------------------


def test_validate_against_corpus_reports_missing_citations():
    class EmptyStore:
        def get_by_citation(self, citation_id):
            return []

    items = [EvalItem.model_validate(_item())]
    problems = validate_against_corpus(items, EmptyStore())
    assert len(problems) == 1
    assert "does not exist" in problems[0]


def test_validate_against_corpus_passes_when_citations_resolve():
    class FullStore:
        def get_by_citation(self, citation_id):
            return ["a chunk"]

    items = [EvalItem.model_validate(_item())]
    assert validate_against_corpus(items, FullStore()) == []


def test_store_failure_is_reported_once_not_per_item():
    class BrokenStore:
        def get_by_citation(self, citation_id):
            raise RuntimeError("index unavailable")

    items = [
        EvalItem.model_validate(_item(id="a")),
        EvalItem.model_validate(_item(id="b")),
    ]
    problems = validate_against_corpus(items, BrokenStore())
    assert len(problems) == 1, "the same broken citation should not be reported twice"


# ---------------------------------------------------------------------------
# The real adversarial set
# ---------------------------------------------------------------------------


def test_the_real_set_loads():
    items = load_evalset()
    assert len(items) >= 60, "the set should have at least 60 items to be meaningful"


def test_the_real_set_covers_every_trap_category():
    items = load_evalset()
    present = {i.category for i in items}
    assert present == set(TrapCategory), f"missing: {set(TrapCategory) - present}"


def test_both_halves_cover_every_trap_category():
    """Otherwise a category could be tuned on and never reported, or reported
    on and never tuned - either makes its numbers meaningless."""
    items = load_evalset()
    for split in Split:
        present = {i.category for i in items if i.split is split}
        assert present == set(TrapCategory), f"{split.value} is missing {set(TrapCategory) - present}"


def test_the_real_set_has_enough_unanswerable_questions():
    """False abstention can only be measured against questions that genuinely
    have no answer. Too few and the metric is noise."""
    items = load_evalset()
    unanswerable = [i for i in items if not i.answerable]
    assert len(unanswerable) >= 10
    # and it must not be so lopsided that abstaining always looks good
    assert len(unanswerable) < len(items) / 2


def test_the_real_set_is_roughly_evenly_split():
    items = load_evalset()
    tuning = len(split_items(items, Split.TUNING))
    heldout = len(split_items(items, Split.HELDOUT))
    assert abs(tuning - heldout) <= max(4, len(items) // 10)


def test_every_answerable_item_in_the_real_set_names_a_citation():
    for item in load_evalset():
        if item.answerable:
            assert item.expected_citations, f"{item.id} has no expected citation"


def test_every_item_in_the_real_set_explains_its_trap():
    """The note is what makes a failure report readable months later."""
    for item in load_evalset():
        assert item.note.strip(), f"{item.id} has no note"


def test_the_real_set_file_is_where_the_code_expects_it():
    assert DEFAULT_EVALSET_PATH.exists()


def test_summarise_counts_the_real_set():
    stats = summarise(load_evalset())
    assert sum(stats["by_category"].values()) == sum(stats["by_split"].values())


# ---------------------------------------------------------------------------
# Content checks - does the cited section still hold the labelled fact?
# ---------------------------------------------------------------------------


class _Store:
    """Returns fixed section text per citation."""

    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    def get_by_citation(self, citation_id):
        text = self.texts.get(citation_id)
        if text is None:
            return []
        return [type("C", (), {"text": text})()]


def test_required_value_present_is_not_an_error():
    store = _Store({VALID_CITATION: "A student must maintain at least 75% attendance."})
    errors, _ = check_expected_content([EvalItem.model_validate(_item())], store)
    assert errors == []


def test_missing_required_value_is_an_error():
    """The fact the item was labelled for is not in the cited section - either
    the label is stale or the fact moved. Either way it cannot be scored."""
    store = _Store({VALID_CITATION: "Attendance is compulsory for all students."})
    errors, _ = check_expected_content([EvalItem.model_validate(_item())], store)
    assert len(errors) == 1
    assert "required value '75'" in errors[0]


def test_required_value_matches_whole_tokens_only():
    """'7' must not be satisfied by the 7 inside '75'. That substring
    confusion is exactly what the numeric traps exist to detect."""
    store = _Store({VALID_CITATION: "A minimum of 75% attendance is required."})
    item = EvalItem.model_validate(_item(required_values=["7"]))
    errors, _ = check_expected_content([item], store)
    assert len(errors) == 1


def test_forbidden_value_in_evidence_is_a_warning_not_an_error():
    """num-006 forbids '7 days' precisely because the cited passage contains
    7 days in a different role. A figure in the wrong role is a legitimate
    trap, so this warns rather than fails."""
    store = _Store({VALID_CITATION: "Intimate the office at least 7 days before."})
    item = EvalItem.model_validate(
        _item(required_values=[], forbidden_values=["7 days"])
    )
    errors, warnings = check_expected_content([item], store)
    assert errors == []
    assert len(warnings) == 1
    assert "7 days" in warnings[0]


def test_word_forbidden_values_are_not_flagged():
    """'fine' and 'always' appear in ordinary prose constantly; flagging them
    would be noise rather than signal."""
    store = _Store({VALID_CITATION: "A fine may always be levied in some cases."})
    item = EvalItem.model_validate(
        _item(required_values=[], forbidden_values=["fine", "always"])
    )
    errors, warnings = check_expected_content([item], store)
    assert errors == [] and warnings == []


def test_content_check_skips_items_with_no_citation():
    store = _Store({})
    item = EvalItem.model_validate(
        _item(id="un-999", category="unanswerable", answerable=False,
              expected_citations=[], required_values=[], forbidden_values=[])
    )
    errors, warnings = check_expected_content([item], store)
    assert errors == [] and warnings == []


def test_coverage_reports_what_cannot_be_checked():
    """Partial coverage must be stated, not implied away."""
    items = [
        EvalItem.model_validate(_item(id="a", required_values=["75"], forbidden_values=[])),
        EvalItem.model_validate(_item(id="b", required_values=[], forbidden_values=["80%"])),
        EvalItem.model_validate(_item(id="c", required_values=[], forbidden_values=["always"])),
    ]
    cov = content_check_coverage(items)
    assert cov["required_checkable"] == 1
    assert cov["forbidden_checkable"] == 1
    assert cov["uncheckable"] == 1


# ---------------------------------------------------------------------------
# Fingerprints - the full-coverage drift check
# ---------------------------------------------------------------------------


def test_fingerprints_record_every_cited_section():
    store = _Store({VALID_CITATION: "some policy text"})
    fp = fingerprint_citations([EvalItem.model_validate(_item())], store)
    assert VALID_CITATION in fp
    assert fp[VALID_CITATION]["chars"] == len("some policy text")
    assert fp[VALID_CITATION]["used_by"] == ["num-999"]


def test_unchanged_content_produces_no_drift():
    store = _Store({VALID_CITATION: "some policy text"})
    items = [EvalItem.model_validate(_item())]
    fp = fingerprint_citations(items, store)
    assert compare_fingerprints(fp, fp) == []


def test_changed_content_is_detected_even_though_the_citation_still_exists():
    """The exact failure this check exists for: an ingestion change re-splits
    a document, the citation still resolves, and the labelled fact has moved.
    Existence checking passes; this must not."""
    items = [EvalItem.model_validate(_item())]
    before = fingerprint_citations(items, _Store({VALID_CITATION: "x" * 40000}))
    after = fingerprint_citations(items, _Store({VALID_CITATION: "x" * 1600}))
    drift = compare_fingerprints(before, after)
    assert len(drift) == 1
    assert "CHANGED" in drift[0]
    assert "num-999" in drift[0], "must name the item to re-check"


def test_vanished_citation_is_detected():
    items = [EvalItem.model_validate(_item())]
    before = fingerprint_citations(items, _Store({VALID_CITATION: "text"}))
    drift = compare_fingerprints(before, {})
    assert len(drift) == 1
    assert "no longer resolves" in drift[0]


def test_newly_cited_section_is_reported():
    items = [EvalItem.model_validate(_item())]
    after = fingerprint_citations(items, _Store({VALID_CITATION: "text"}))
    drift = compare_fingerprints({}, after)
    assert len(drift) == 1
    assert "newly cited" in drift[0]


def test_fingerprints_round_trip_through_disk(tmp_path):
    store = _Store({VALID_CITATION: "some policy text"})
    fp = fingerprint_citations([EvalItem.model_validate(_item())], store)
    path = save_fingerprints(fp, tmp_path / "fp.json")
    assert load_fingerprints(path) == fp


def test_load_fingerprints_returns_none_when_absent(tmp_path):
    assert load_fingerprints(tmp_path / "nope.json") is None


def test_the_real_baseline_matches_the_real_corpus_labels():
    """The committed baseline should cover exactly the citations the set uses."""
    from policyverify.evalset import FINGERPRINT_PATH

    recorded = load_fingerprints(FINGERPRINT_PATH)
    if recorded is None:
        import pytest

        pytest.skip("no fingerprint baseline recorded yet")
    cited = {c for i in load_evalset() for c in i.expected_citations}
    assert set(recorded) == cited


# ---------------------------------------------------------------------------
# _mentions - token boundaries around numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,text,expected",
    [
        # a sentence-final full stop is not part of the number
        ("75%", "attendance of 75%.", True),
        ("7 days", "at least 7 days.", True),
        # PDF extraction routinely splits the percent sign off its number
        ("75%", "a minimum of 75 % in", True),
        ("75 %", "written as 75% here", True),
        # but a "." followed by a digit IS part of the number
        ("7", "see section 7.3", False),
        ("75", "the figure 0.75 here", False),
        # and a value must never be satisfied by a longer number containing it
        ("7", "a minimum of 75% here", False),
        ("10", "a total of 100 marks", False),
        # ordinary cases still work
        ("75%", "75% at the very start", True),
        ("7", "within 7 days", True),
        ("two", "two representatives per floor", True),
        ("7 days", "at least  7   days later", True),
    ],
)
def test_mentions_token_boundaries(value, text, expected):
    assert _mentions(value, text) is expected
