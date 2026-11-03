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

import hashlib
import json
import re
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

    EXISTENCE IS NOT ENOUGH. A citation can survive an ingestion change while
    the fact it was labelled for moves into a different section. The ID still
    resolves, so this check passes, and the label is quietly wrong. See
    check_expected_content() and fingerprint_citations() for the two checks
    that catch that.
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


# ---------------------------------------------------------------------------
# Content checks - does the cited section still hold the labelled fact?
# ---------------------------------------------------------------------------


def _mentions(value: str, text: str) -> bool:
    """Is `value` present in `text` as a whole token?

    Whole-token matching matters for the numbers these labels mostly contain:
    a plain substring test would let the required value "7" be satisfied by
    the "75" in "75% attendance", which is the exact confusion the numeric
    traps exist to detect.

    Three things the boundaries have to get right at once:

      "75%" must match "attendance of 75%."   - a sentence-final full stop is
          not part of the number, so the lookahead must not reject every "."
      "75%" must match "a minimum of 75 % in" - PDF extraction routinely
          splits the percent sign off, so whitespace before "%" is optional
      "7"   must NOT match "see section 7.3"  - here the "." IS part of the
          number, so a "." followed by a digit still blocks the match

    So the rule is: block a following word character, and block a following
    "." only when a digit comes after it.
    """
    # Normalise the label first: collapse whitespace runs, and drop any space
    # before a percent sign so "75 %" and "75%" are written the same way
    # before the pattern is built.
    normalised = re.sub(r"\s+", " ", value.strip())
    normalised = re.sub(r"\s+%", "%", normalised)
    pattern = re.escape(normalised)
    # Remaining spaces must be present in the text; a percent sign may be
    # separated from its number by whitespace that the label does not have.
    pattern = pattern.replace(r"\ ", r"\s+").replace(" ", r"\s+")
    pattern = pattern.replace("%", r"\s*%")
    return (
        re.search(rf"(?<![\w.]){pattern}(?!\w)(?!\.\d)", text, re.I) is not None
    )


def _citation_text(store, citation_id: str, cache: dict[str, str | None]) -> str | None:
    """All the text of one cited section, or None if it cannot be read."""
    if citation_id not in cache:
        try:
            chunks = store.get_by_citation(citation_id)
            cache[citation_id] = "\n".join(c.text for c in chunks) if chunks else None
        except Exception:
            cache[citation_id] = None
    return cache[citation_id]


def check_expected_content(items: list[EvalItem], store) -> tuple[list[str], list[str]]:
    """Check the cited sections actually contain what the labels claim.

    Returns (errors, warnings). The split is deliberate, because the two
    findings differ in how certain they are:

      ERROR - MISSING REQUIRED VALUE
          The item says a correct answer must contain "75", but no cited
          section mentions 75. Either the label is wrong or the fact has
          moved. Unambiguous: the item cannot be scored as written.

      WARNING - FORBIDDEN VALUE PRESENT
          The item treats "80%" as the wrong answer, but "80%" is sitting in
          the cited section. Usually that means the item is not a trap at
          all. But not always: num-006 forbids "7 days" precisely because
          the cited passage contains 7 days as a *notice period* while the
          question asks about *stay length*. A figure in the wrong role is a
          legitimate and valuable trap, so this is worth a human look rather
          than an automatic failure.

    Only items carrying those labels can be checked this way, so coverage is
    partial - callers should report how many items were actually checked
    rather than implying the whole set passed.
    """
    errors: list[str] = []
    warnings: list[str] = []
    cache: dict[str, str | None] = {}

    for item in items:
        if not item.expected_citations:
            continue
        texts = [_citation_text(store, c, cache) for c in item.expected_citations]
        combined = "\n".join(t for t in texts if t)
        if not combined:
            continue  # existence is validate_against_corpus()'s job

        for value in item.required_values:
            if not _mentions(value, combined):
                errors.append(
                    f"{item.id}: required value {value!r} is not in any cited section "
                    f"({', '.join(item.expected_citations)}) - the label may be stale"
                )

        for value in item.forbidden_values:
            # Only consider numeric-looking forbidden values. Word values
            # ("fine", "always") routinely appear in nearby prose without
            # making the item a bad trap, so flagging them would be noise.
            if any(ch.isdigit() for ch in value) and _mentions(value, combined):
                warnings.append(
                    f"{item.id}: forbidden value {value!r} is present in a cited section "
                    f"- confirm this is a role-confusion trap and not a mislabel"
                )

    return errors, warnings


def content_check_coverage(items: list[EvalItem]) -> dict[str, int]:
    """How much of the set the content check can actually speak to."""
    cited = [i for i in items if i.expected_citations]
    return {
        "total": len(items),
        "with_citations": len(cited),
        "required_checkable": len([i for i in cited if i.required_values]),
        "forbidden_checkable": len(
            [i for i in cited if any(any(c.isdigit() for c in v) for v in i.forbidden_values)]
        ),
        "uncheckable": len(
            [
                i
                for i in cited
                if not i.required_values
                and not any(any(c.isdigit() for c in v) for v in i.forbidden_values)
            ]
        ),
    }


# ---------------------------------------------------------------------------
# Fingerprints - detect a cited section changing under the labels
# ---------------------------------------------------------------------------

FINGERPRINT_PATH = PROJECT_ROOT / "eval" / "citation_fingerprints.json"


def fingerprint_citations(items: list[EvalItem], store) -> dict[str, dict]:
    """A content hash for every cited section, so drift becomes visible.

    This is the check with full coverage. The others need a label to compare
    against; this one needs nothing, because it only asks whether the cited
    text is the same text the labels were written against.

    The failure it exists for is real and silent: an ingestion change can
    re-split a document so that a citation still resolves while the fact it
    was labelled for has moved elsewhere. Existence checks pass, content
    checks pass if the label has no required value, and the item is quietly
    scoring against the wrong passage.
    """
    out: dict[str, dict] = {}
    cache: dict[str, str | None] = {}

    for citation in sorted({c for i in items for c in i.expected_citations}):
        text = _citation_text(store, citation, cache)
        if text is None:
            continue
        out[citation] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "chars": len(text),
            "used_by": sorted(i.id for i in items if citation in i.expected_citations),
        }
    return out


def load_fingerprints(path: Path | str | None = None) -> dict[str, dict] | None:
    path = Path(path) if path else FINGERPRINT_PATH
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_fingerprints(data: dict[str, dict], path: Path | str | None = None) -> Path:
    path = Path(path) if path else FINGERPRINT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def compare_fingerprints(recorded: dict[str, dict], current: dict[str, dict]) -> list[str]:
    """Report citations whose content changed, vanished or appeared."""
    problems: list[str] = []

    for citation, old in sorted(recorded.items()):
        new = current.get(citation)
        if new is None:
            problems.append(
                f"{citation}: no longer resolves (was {old['chars']} chars, "
                f"used by {', '.join(old['used_by'])})"
            )
            continue
        if new["sha256"] != old["sha256"]:
            delta = new["chars"] - old["chars"]
            problems.append(
                f"{citation}: content CHANGED ({old['chars']} -> {new['chars']} chars, "
                f"{delta:+d}) - re-check labels for {', '.join(old['used_by'])}"
            )

    for citation in sorted(set(current) - set(recorded)):
        problems.append(f"{citation}: newly cited, no fingerprint recorded yet")

    return problems
