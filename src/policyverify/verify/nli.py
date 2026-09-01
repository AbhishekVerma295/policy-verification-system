"""
nli.py - does this passage actually prove this claim?

Natural Language Inference. The model is given two texts and answers one
narrow question:

    premise:    "A student must maintain a minimum attendance record of at
                 least 75% in individual courses."
    hypothesis: "The attendance requirement is 75%."
    answer:     ENTAILMENT

Three possible answers, and all three are useful:
    ENTAILMENT     the passage proves the claim
    CONTRADICTION  the passage disproves it
    NEUTRAL        the passage neither proves nor disproves it

NEUTRAL is not a failure of the checker. It is the honest answer when a
passage simply does not address a claim, and it is exactly the situation in
which the system should consider staying quiet rather than answering.

WHY NOT JUST ASK QWEN TO CHECK ITS OWN WORK
    A model grading itself tends to repeat its own mistakes: the same
    misreading that produced the bad claim produces a confident bad grade.
    A separate NLI model answers one narrow question independently, runs on
    CPU, and is deterministic.

WHY LONG PASSAGES ARE SPLIT INTO WINDOWS
    The model has a fixed input limit (512 tokens for DeBERTa). Our chunks
    run to ~1,950 characters, which can exceed it. Letting the tokenizer
    truncate would silently discard the tail of the passage - and the
    sentence that supports the claim may be exactly what gets cut. That is a
    silent wrong answer, the failure mode this project exists to prevent.

    Instead the passage is split into overlapping windows and each is scored.
    A claim supported by any part of the passage is supported by the passage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from policyverify.config import Config, get_config

# Split on sentence ends. Deliberately simple: over-splitting is harmless
# here because windows overlap and are scored independently.
_SENTENCE_RE = re.compile(r"(?<=[.!?;:])\s+")

# Characters per window. Comfortably inside a 512-token budget once the
# hypothesis is added, with room for the long words in policy prose.
_WINDOW_CHARS = 900
_WINDOW_OVERLAP = 200

ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"


@dataclass
class NLIResult:
    """One verdict from the NLI model."""

    label: str
    entailment: float
    neutral: float
    contradiction: float
    # Which window produced this, kept for debugging a surprising verdict.
    window: str = ""

    @property
    def score(self) -> float:
        """Confidence in whichever label was chosen."""
        return {
            ENTAILMENT: self.entailment,
            NEUTRAL: self.neutral,
            CONTRADICTION: self.contradiction,
        }.get(self.label, 0.0)


def split_windows(
    text: str, window_chars: int = _WINDOW_CHARS, overlap: int = _WINDOW_OVERLAP
) -> list[str]:
    """Split a passage into overlapping windows on sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= window_chars:
        return [text]

    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    windows: list[str] = []
    current = ""

    for sentence in sentences:
        # A single sentence longer than the window: hard-split it, since
        # there is no smaller natural unit left to break on.
        if len(sentence) > window_chars:
            if current:
                windows.append(current)
                current = ""
            for i in range(0, len(sentence), window_chars):
                windows.append(sentence[i : i + window_chars])
            continue

        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= window_chars:
            current = candidate
        else:
            windows.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail} {sentence}".strip()

    if current:
        windows.append(current)
    return windows


def normalise_label(raw: str) -> str:
    """Different checkpoints spell the labels differently. Map to ours."""
    value = raw.lower().replace("-", "_")
    if "entail" in value or value in ("label_0", "0"):
        return ENTAILMENT
    if "contradict" in value or value in ("label_2", "2"):
        return CONTRADICTION
    if "neutral" in value or value in ("label_1", "1"):
        return NEUTRAL
    return value


class NLIChecker:
    """The fact-checking model, loaded once and reused."""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            cfg = self.config.nli
            self._pipeline = pipeline(
                "text-classification",
                model=cfg.model_name,
                # -1 means CPU. Deliberate: this model is small enough to be
                # fast there, and it keeps the whole GPU free for Qwen.
                device=-1 if cfg.device == "cpu" else 0,
                top_k=None,
                truncation=True,
            )
        return self._pipeline

    def _score_pair(self, premise: str, hypothesis: str) -> NLIResult:
        raw = self.pipeline({"text": premise, "text_pair": hypothesis})
        scores = raw[0] if isinstance(raw[0], list) else raw
        by_label = {normalise_label(s["label"]): float(s["score"]) for s in scores}
        entail = by_label.get(ENTAILMENT, 0.0)
        neutral = by_label.get(NEUTRAL, 0.0)
        contra = by_label.get(CONTRADICTION, 0.0)
        label = max(
            ((ENTAILMENT, entail), (NEUTRAL, neutral), (CONTRADICTION, contra)),
            key=lambda pair: pair[1],
        )[0]
        return NLIResult(
            label=label,
            entailment=entail,
            neutral=neutral,
            contradiction=contra,
            window=premise[:120],
        )

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        """Does `premise` prove `hypothesis`?

        Scores every window of the premise and returns the most decisive
        result. Support anywhere in the passage counts as support; a
        contradiction only wins if nothing else entailed, so a passage that
        both states a rule and lists an exception is not written off.
        """
        windows = split_windows(premise)
        if not windows:
            return NLIResult(label=NEUTRAL, entailment=0.0, neutral=1.0, contradiction=0.0)

        results = [self._score_pair(window, hypothesis) for window in windows]
        threshold = self.config.nli.entailment_threshold

        best_entail = max(results, key=lambda r: r.entailment)
        if best_entail.entailment >= threshold:
            return best_entail

        best_contra = max(results, key=lambda r: r.contradiction)
        if best_contra.contradiction >= threshold:
            return best_contra

        # Nothing decisive either way - report the strongest signal we saw.
        return max(results, key=lambda r: max(r.entailment, r.contradiction))
