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
    load_evalset,
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
