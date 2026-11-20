"""
situation.py - applying a verified rule to the figure the asker gave.

WHY THIS IS CODE AND NOT THE LANGUAGE MODEL
"I attended 74 classes out of 100 - will I be detained?" needs one sum and
one comparison: 74 / 100 = 74%, and 74% is below the 75% minimum. A claim
saying "74% is below 75%" cannot be verified against the policy - the
policy never mentions 74 - so if the model wrote it, the checker would
rightly reject it. And a small model's arithmetic is not something to
stand behind anyway.

So the model states the rule (verified as usual), and this module does the
comparison deterministically, only from:
  - a figure the asker stated in the question, and
  - a threshold inside a claim that PASSED verification.
It never adds a rule, and it says nothing when the comparison is not
clear-cut: no figure in the question, no verified threshold, or more than
one threshold that could apply. Silence is the safe failure here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from policyverify.schema import ClaimVerdict

# "74 classes out of 100", "74 out of 100 classes", "74/100 lectures"
_OUT_OF_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(?:[a-z]+\s+){0,2}(?:out\s+of|/|of)\s*(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
# "I have 72% attendance", "my attendance is 72 percent"
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:%|per\s?cent|percent)", re.IGNORECASE)

# A threshold in a verified claim: the qualifier decides which side passes.
_MIN_WORDS = r"at\s+least|minimum(?:\s+of)?|not\s+less\s+than|no\s+less\s+than"
_BELOW_WORDS = r"less\s+than|below|under"
_MAX_WORDS = r"cannot\s+exceed|not\s+exceed|at\s+most|maximum(?:\s+of)?|up\s+to"
_THRESHOLD_RE = re.compile(
    rf"\b(?P<kind>{_MIN_WORDS}|{_BELOW_WORDS}|{_MAX_WORDS})\b[^.%\d]{{0,40}}?"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:%|per\s?cent|percent)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SituationNote:
    """The asker's figure set against one verified threshold."""

    user_value: float
    user_text: str
    threshold: float
    kind: str  # "minimum" (must reach it) or "maximum" (must not pass it)
    meets: bool
    citation: str
    message: str


def user_percentage(question: str) -> tuple[float, str] | None:
    """The asker's own percentage: stated, or worked out from "X out of Y"."""
    match = _OUT_OF_RE.search(question)
    if match:
        part, whole = float(match.group(1)), float(match.group(2))
        if whole > 0 and part <= whole:
            value = round(part / whole * 100, 2)
            return value, f"{match.group(0).strip()} = {_fmt(value)}%"
    match = _PERCENT_RE.search(question)
    if match:
        value = float(match.group(1))
        return value, f"{_fmt(value)}%"
    return None


def _fmt(value: float) -> str:
    return f"{value:g}"


def _kind(word: str) -> str:
    word = " ".join(word.lower().split())
    if re.fullmatch(_MAX_WORDS, word):
        return "maximum"
    if re.fullmatch(_BELOW_WORDS, word):
        return "below"
    return "minimum"


def apply_to_situation(question: str, kept: list[ClaimVerdict]) -> SituationNote | None:
    """Compare the asker's percentage with the one verified threshold, or None."""
    found = user_percentage(question)
    if found is None:
        return None
    user_value, user_text = found

    thresholds: dict[tuple[str, float], str] = {}
    for verdict in kept:
        for match in _THRESHOLD_RE.finditer(verdict.claim.text):
            kind = _kind(match.group("kind"))
            # "less than 75%" states the same minimum as "at least 75%".
            kind = "minimum" if kind == "below" else kind
            value = float(match.group("value"))
            citation = verdict.claim.citation_ids[0] if verdict.claim.citation_ids else ""
            thresholds.setdefault((kind, value), citation)
    # The asker's figure is compared with a MINIMUM it must reach. A maximum
    # ("condonation cannot exceed 10%") limits something else - the size of
    # a remedy - and comparing the asker's attendance with it would be wrong.
    minimums = {key: c for key, c in thresholds.items() if key[0] == "minimum"}
    if len({value for _, value in minimums}) != 1:
        return None  # none, or two different minimums: not clear-cut
    (kind, threshold), citation = next(iter(minimums.items()))

    meets = user_value >= threshold
    gap = round(abs(threshold - user_value), 2)
    if meets:
        verdict_text = f"which meets the minimum of {_fmt(threshold)}%"
    else:
        verdict_text = (
            f"which is below the minimum of {_fmt(threshold)}% "
            f"(short by {_fmt(gap)} percentage point{'s' if gap != 1 else ''})"
        )
    message = f"Your figure: {user_text}, {verdict_text}."
    return SituationNote(
        user_value=user_value,
        user_text=user_text,
        threshold=threshold,
        kind=kind,
        meets=meets,
        citation=citation,
        message=message,
    )


__all__ = ["SituationNote", "apply_to_situation", "user_percentage"]
