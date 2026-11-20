"""
polarity.py - does the evidence decide between a claim and its opposite?

THE FAILURE
The NLI model scores how well a passage supports a claim, and it leans on
shared words. A NEGATIVE claim that reuses the passage's words can score as
supported while saying the opposite of it. Measured: asked whether 74 of
100 classes means detention, the model wrote "The number of classes attended
does not affect the detention policy", and DeBERTa called it ENTAILED (0.76)
by "If a candidate does not earn the required attendance ... he/she will be
allowed to sit for the examinations after satisfying the required
attendance" - a passage that says attendance DOES matter. Read as a yes/no
answer ("No, ...") it is the most harmful kind of wrong.

THE TEST
For a supported claim that contains a negation, score the same claim with
the negation removed ("... does affect ..."). Real support for a negative
claim CONTRADICTS its positive twin: "Fees once paid will be refunded
normally" is contradicted by "Fees once paid will not be refunded normally".
If the evidence instead supports the twin about as well as the claim, it is
matching words, not meaning, and does not settle the claim either way.
"""

from __future__ import annotations

import re

# Ordered: longer forms first, so "does not" is handled before "not".
_FLIPS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bcannot\b", re.IGNORECASE), "can"),
    (re.compile(r"\bcan\s+not\b", re.IGNORECASE), "can"),
    (re.compile(r"\b(do|does|did|is|are|was|were|will|shall|should|must|may|can|could|would|has|have|had)\s+not\b", re.IGNORECASE), r"\1"),
    (re.compile(r"\b(do|does|did|is|are|was|were|wo|sha|should|must|ca|could|would|has|have|had)n['’]t\b", re.IGNORECASE), r"\1"),
    (re.compile(r"\bnever\b", re.IGNORECASE), "always"),
    (re.compile(r"\bnot\b", re.IGNORECASE), ""),
]
_LEADING_NO_RE = re.compile(r"^\s*no\s+(?=[a-z])", re.IGNORECASE)
_CONTRACTION_BASE = {"wo": "will", "sha": "shall", "ca": "can"}


def has_negation(text: str) -> bool:
    return flip_negation(text) != text


def flip_negation(text: str) -> str:
    """The claim with its negation removed, or the text unchanged."""
    out = _LEADING_NO_RE.sub("A ", text, count=1)
    for pattern, repl in _FLIPS:
        new = pattern.sub(repl, out, count=1)
        if new != out:
            out = new
            break
    out = re.sub(
        r"\b(wo|sha|ca)\b(?=\s)",
        lambda m: _CONTRACTION_BASE[m.group(1).lower()],
        out,
    )
    return " ".join(out.split())


def polarity_undecided(
    claim_entailment: float, twin_entailment: float, threshold: float, margin: float
) -> bool:
    """True when the evidence does not prefer the claim over its opposite.

    The twin must be clearly LESS supported - by `margin` - and not itself
    supported; otherwise the passage does not settle which is true.
    """
    return twin_entailment >= threshold or twin_entailment > claim_entailment - margin


__all__ = ["flip_negation", "has_negation", "polarity_undecided"]
