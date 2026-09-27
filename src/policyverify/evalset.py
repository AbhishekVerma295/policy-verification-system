"""
evalset.py - the adversarial question set: schema, loading, validation.

WHAT THIS IS FOR
    Claiming "the system works" is not evidence. This is a set of questions
    written specifically to trip it up, each labelled with what the correct
    behaviour is, so the five metrics can be computed rather than asserted.

WHAT CAN AND CANNOT BE LABELLED IN ADVANCE
    Claims are generated fresh on every run, so they cannot be labelled ahead
    of time. What *can* be fixed in advance is everything about the question:

        answerable           does the corpus cover this at all?
        expected_citations   which sections a correct answer must rest on
        required_values      figures a correct answer must contain
        forbidden_values     figures that would mean the system got it wrong

    `forbidden_values` is what makes most of this set automatically scorable.
    For a numeric trap we know the wrong answer in advance - if "80%" appears
    in a claim the system stood behind, that is a hallucination, and no human
    has to read it to find out.

TUNING VERSUS HELD-OUT
    Every item belongs to one half. Thresholds are tuned on the tuning half
    and results reported on the held-out half. Tuning and reporting on the
    same questions would only prove the system works on the questions it was
    tuned for - a result that sounds impressive and means nothing.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from policyverify.config import PROJECT_ROOT
from policyverify.schema import CitationID

DEFAULT_EVALSET_PATH = PROJECT_ROOT / "eval" / "adversarial.jsonl"


class TrapCategory(StrEnum):
    """The five ways these questions are designed to cause failure."""

    # The same figure means different things in different policies - 75% is
    # the minimum attendance rule and also a tuition-waiver tier. Answering
    # with the right number from the wrong policy is a realistic error.
    CROSS_POLICY = "cross_policy"

    # Thresholds, deadlines and percentages, where being wrong by one digit
    # changes the outcome for a student completely.
    NUMERIC = "numeric"

    # Plausible, well-formed questions the corpus simply does not cover. The
    # correct behaviour is to abstain.
    UNANSWERABLE = "unanswerable"

    # Rules with carve-outs. Reporting the rule while dropping the exception
    # produces an answer that is confidently and dangerously incomplete.
    NEGATION_EXCEPTION = "negation_exception"

    # Questions that assert something untrue in the asking ("why does the
    # university require 90% attendance?"). The correct behaviour is to
    # correct the premise rather than elaborate on it.
    FALSE_PREMISE = "false_premise"


class Split(StrEnum):
    TUNING = "tuning"
    HELDOUT = "heldout"


class EvalItem(BaseModel):
    """One adversarial question with its expected behaviour."""

    id: str
    question: str
    category: TrapCategory
    split: Split
    # Does the corpus contain an answer? Drives abstention scoring: for a
    # False item the system SHOULD abstain, and answering is the failure.
    answerable: bool

    expected_citations: list[str] = Field(default_factory=list)
    required_values: list[str] = Field(default_factory=list)
    forbidden_values: list[str] = Field(default_factory=list)

    # Why this is a trap, and what a correct answer looks like. For a human
    # reading a failure report, not used in scoring.
    note: str = ""

    @field_validator("question", "id")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @field_validator("expected_citations")
    @classmethod
    def _citations_well_formed(cls, v: list[str]) -> list[str]:
        for citation in v:
            CitationID.parse(citation)  # raises if malformed
        return v

    @model_validator(mode="after")
    def _coherent(self) -> EvalItem:
        if not self.answerable:
            if self.expected_citations:
                raise ValueError(
                    f"{self.id}: an unanswerable item cannot expect citations"
                )
            if self.required_values:
                raise ValueError(
                    f"{self.id}: an unanswerable item cannot require values"
                )
        elif not self.expected_citations:
            raise ValueError(
                f"{self.id}: an answerable item needs at least one expected citation, "
                f"otherwise citation accuracy cannot be scored"
            )
        return self


def load_evalset(path: Path | str | None = None) -> list[EvalItem]:
    """Read the adversarial set from JSONL.

    One JSON object per line. Blank lines and `#` comments are skipped, so
    the file can be annotated while remaining machine-readable.
    """
    path = Path(path) if path else DEFAULT_EVALSET_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No adversarial set at {path}. Expected one JSON object per line."
        )

    items: list[EvalItem] = []
    seen_ids: set[str] = set()

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = EvalItem.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        if item.id in seen_ids:
            raise ValueError(f"{path.name} line {number}: duplicate id {item.id!r}")
        seen_ids.add(item.id)
        items.append(item)

    return items


def split_items(items: list[EvalItem], split: Split) -> list[EvalItem]:
    """Just one half of the set."""
    return [i for i in items if i.split is split]


def summarise(items: list[EvalItem]) -> dict[str, dict[str, int]]:
    """Counts by category and split, for checking the set is balanced."""
    by_category: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for item in items:
        by_category[item.category.value] = by_category.get(item.category.value, 0) + 1
        by_split[item.split.value] = by_split.get(item.split.value, 0) + 1
    answerable = sum(1 for i in items if i.answerable)
    return {
        "by_category": by_category,
        "by_split": by_split,
        "answerable": {"yes": answerable, "no": len(items) - answerable},
    }


def validate_against_corpus(items: list[EvalItem], store) -> list[str]:
    """Check every expected citation actually resolves to a real section.

    A dataset that cites sections which do not exist would silently corrupt
    every result computed from it - the system would be marked wrong for not
    citing something that was never there. Worth checking explicitly rather
    than trusting the labels.
    """
    problems: list[str] = []
    cache: dict[str, bool] = {}

    for item in items:
        for citation in item.expected_citations:
            if citation not in cache:
                try:
                    cache[citation] = bool(store.get_by_citation(citation))
                except Exception as exc:
                    problems.append(f"{item.id}: could not check {citation} ({exc})")
                    cache[citation] = True  # do not report it twice
            if not cache[citation]:
                problems.append(f"{item.id}: expected citation does not exist: {citation}")

    return problems
