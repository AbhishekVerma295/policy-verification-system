"""
claimset.py - a labelled set of claims, for measuring the verifier itself.

WHY THIS EXISTS ALONGSIDE THE ADVERSARIAL SET
    eval/adversarial.jsonl scores whole answers: was the right section cited,
    did a forbidden figure appear, did the system decline when it should. It
    cannot say whether the VERIFIER is right about an individual claim,
    because the claims are written fresh by the language model on every run
    and nobody has labelled them.

    This set fixes the claims in advance. Each one is paired with the section
    it cites and a label - does that section support it or not - so the
    verifier can be scored on its own, with retrieval and generation taken
    out of the picture. That is what "claim verification precision and
    recall" means, and it could not be reported before this file existed.

HOW TO READ THE TWO HEADLINE NUMBERS
    precision   of the claims the verifier ACCEPTED, how many were really
                supported. Low precision = false claims reach the user.
    recall      of the claims that really WERE supported, how many the
                verifier accepted. Low recall = true claims are thrown away,
                which shows up to the user as needless refusals.

THE KINDS, AND WHY THEY ARE KEPT SEPARATE
    A single precision figure hides where the verifier is weak. A false claim
    can be false in several different ways, and the verifier is good at some
    and poor at others - so every claim records its kind and results are
    always broken down by it.

      supported claims
        paraphrase     the section says this, in other words
        verbatim       a sentence copied from the section, true by construction

      unsupported claims
        number         a figure changed to one that is NOT in the section
        role           a figure changed to one that IS in the section, but
                       means something else there (the wrong table row)
        negation       the meaning flipped
        entity         the person or thing swapped for another
        unsupported    plausible, but the section says nothing about it
        wrong_section  a true fact, cited to a section that does not state it

Nothing here loads a model. Scoring takes the verifier's verdicts as input,
so it is tested with canned verdicts and runs in milliseconds.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from policyverify.config import PROJECT_ROOT
from policyverify.evalset import Split
from policyverify.evaluate import wilson_interval
from policyverify.schema import CitationID, RetrievedChunk, VerdictStatus

DEFAULT_CLAIMSET_PATH = PROJECT_ROOT / "eval" / "claims.jsonl"


class ClaimLabel(StrEnum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"


class ClaimKind(StrEnum):
    PARAPHRASE = "paraphrase"
    VERBATIM = "verbatim"
    NUMBER = "number"
    ROLE = "role"
    NEGATION = "negation"
    ENTITY = "entity"
    UNSUPPORTED = "unsupported"
    WRONG_SECTION = "wrong_section"


SUPPORTED_KINDS = {ClaimKind.PARAPHRASE, ClaimKind.VERBATIM}
# Kinds where the section does not merely fail to support the claim but says
# something that conflicts with it, so REFUTED is the ideal verdict rather
# than NEUTRAL. Reported separately: telling "wrong" from "unknown" is what
# lets the interface show a correction instead of a shrug.
CONTRADICTED_KINDS = {ClaimKind.NUMBER, ClaimKind.ROLE, ClaimKind.NEGATION, ClaimKind.ENTITY}


class ClaimItem(BaseModel):
    """One claim, the section it cites, and whether that section supports it."""

    id: str
    claim: str
    citation_ids: list[str] = Field(min_length=1)
    label: ClaimLabel
    kind: ClaimKind
    split: Split
    note: str = ""

    @field_validator("claim")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim text is empty")
        return v.strip()

    @field_validator("citation_ids")
    @classmethod
    def _well_formed(cls, v: list[str]) -> list[str]:
        for citation in v:
            CitationID.parse(citation)  # raises on a malformed ID
        return v

    @model_validator(mode="after")
    def _label_matches_kind(self) -> ClaimItem:
        # The kind already implies the label. Storing both and checking they
        # agree catches a mislabelled line when it is written, not when it
        # quietly drags a metric down.
        expected = (
            ClaimLabel.SUPPORTED if self.kind in SUPPORTED_KINDS else ClaimLabel.NOT_SUPPORTED
        )
        if self.label is not expected:
            raise ValueError(
                f"kind {self.kind.value!r} implies label {expected.value!r}, "
                f"but the item says {self.label.value!r}"
            )
        return self


def load_claimset(path: Path | str | None = None) -> list[ClaimItem]:
    """Read the claim set from JSONL. Blank lines and `#` comments are skipped."""
    path = Path(path) if path else DEFAULT_CLAIMSET_PATH
    if not path.exists():
        raise FileNotFoundError(f"No claim set at {path}. Expected one JSON object per line.")

    items: list[ClaimItem] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = ClaimItem.model_validate(json.loads(line))
        except Exception as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
        if item.id in seen:
            raise ValueError(f"{path.name} line {number}: duplicate id {item.id!r}")
        seen.add(item.id)
        items.append(item)
    return items


def missing_citations(items: list[ClaimItem], store) -> list[str]:
    """Citations in the set that no longer exist in the index.

    A label is only meaningful while the section it points at exists, so this
    is checked before any scoring is trusted.
    """
    problems = []
    cache: dict[str, bool] = {}
    for item in items:
        for citation in item.citation_ids:
            if citation not in cache:
                cache[citation] = bool(store.get_by_citation(citation))
            if not cache[citation]:
                problems.append(f"{item.id}: cited section {citation} is not in the index")
    return problems


def evidence_for(item: ClaimItem, store) -> list[RetrievedChunk]:
    """Every chunk of the cited section(s), as if retrieval had returned them.

    The verifier normally checks a claim against the chunks retrieval found.
    Handing it the whole cited section instead removes retrieval from the
    measurement: a wrong verdict here is the verifier's alone.
    """
    chunks = [c for citation in item.citation_ids for c in store.get_by_citation(citation)]
    return [RetrievedChunk(chunk=c, score=1.0, rank=i) for i, c in enumerate(chunks)]


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def score_claims(items: list[ClaimItem], statuses: dict[str, VerdictStatus]) -> dict:
    """Compare the verifier's verdicts with the labels. Pure - no I/O.

    `statuses` maps item id -> the verdict the verifier returned. Accepting a
    claim means SUPPORTED; NEUTRAL and REFUTED both mean it was not accepted.
    """
    scored = [i for i in items if i.id in statuses]

    def accepted(item: ClaimItem) -> bool:
        return statuses[item.id] is VerdictStatus.SUPPORTED

    true_items = [i for i in scored if i.label is ClaimLabel.SUPPORTED]
    false_items = [i for i in scored if i.label is ClaimLabel.NOT_SUPPORTED]
    tp = sum(1 for i in true_items if accepted(i))
    fn = len(true_items) - tp
    fp = sum(1 for i in false_items if accepted(i))
    tn = len(false_items) - fp

    precision, recall = _rate(tp, tp + fp), _rate(tp, tp + fn)
    f1 = (
        round(2 * precision * recall / (precision + recall), 4)
        if precision and recall
        else None
    )

    by_kind: dict[str, dict] = {}
    for kind in ClaimKind:
        rows = [i for i in scored if i.kind is kind]
        if not rows:
            continue
        verdicts = {status.value: 0 for status in VerdictStatus}
        for row in rows:
            verdicts[statuses[row.id].value] += 1
        # "Correct" means accepted for a supported kind, not accepted otherwise.
        want_accept = kind in SUPPORTED_KINDS
        correct = sum(1 for r in rows if accepted(r) == want_accept)
        by_kind[kind.value] = {
            "items": len(rows),
            "correct": correct,
            "accuracy": _rate(correct, len(rows)),
            "verdicts": verdicts,
        }

    contradicted = [i for i in false_items if i.kind in CONTRADICTED_KINDS]
    refuted = sum(1 for i in contradicted if statuses[i.id] is VerdictStatus.REFUTED)

    return {
        "items": len(scored),
        "supported_items": len(true_items),
        "unsupported_items": len(false_items),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "precision_ci": wilson_interval(tp, tp + fp),
        "recall": recall,
        "recall_ci": wilson_interval(tp, tp + fn),
        "f1": f1,
        # Share of false claims the verifier let through.
        "false_accept_rate": _rate(fp, fp + tn),
        "false_accept_ci": wilson_interval(fp, fp + tn),
        # Share of true claims the verifier threw away.
        "false_reject_rate": _rate(fn, tp + fn),
        "false_reject_ci": wilson_interval(fn, tp + fn),
        "contradiction_detection": {
            "items": len(contradicted),
            "refuted": refuted,
            "rate": _rate(refuted, len(contradicted)),
        },
        "by_kind": by_kind,
        "wrong": sorted(
            i.id for i in scored if accepted(i) != (i.label is ClaimLabel.SUPPORTED)
        ),
    }


def format_claim_report(summary: dict, items: list[ClaimItem] = (), statuses=None) -> str:
    """A plain-text report for the terminal."""

    def pct(value: float | None) -> str:
        return "   n/a" if value is None else f"{value * 100:5.1f}%"

    def ci(interval) -> str:
        if not interval:
            return ""
        return f"  [95% CI {interval[0] * 100:.0f}-{interval[1] * 100:.0f}%]"

    c = summary["confusion"]
    d = summary["contradiction_detection"]
    lines = [
        f"claims scored         {summary['items']}  "
        f"({summary['supported_items']} supported, {summary['unsupported_items']} not)",
        "",
        f"precision             {pct(summary['precision'])}{ci(summary['precision_ci'])}",
        "                      (of the claims accepted, how many were really supported)",
        f"recall                {pct(summary['recall'])}{ci(summary['recall_ci'])}",
        "                      (of the claims really supported, how many were accepted)",
        f"F1                    {pct(summary['f1'])}",
        f"false-accept rate     {pct(summary['false_accept_rate'])}{ci(summary['false_accept_ci'])}",
        f"false-reject rate     {pct(summary['false_reject_rate'])}{ci(summary['false_reject_ci'])}",
        "",
        f"confusion             accepted & true {c['tp']}   accepted & false {c['fp']}   "
        f"rejected & true {c['fn']}   rejected & false {c['tn']}",
        f"contradictions named  {d['refuted']} of {d['items']} conflicting claims were marked "
        f"REFUTED rather than merely unverified",
        "",
        f"by kind               {'correct':>9}   supported / unverified / refuted",
    ]
    for kind, row in summary["by_kind"].items():
        v = row["verdicts"]
        lines.append(
            f"  {kind:<18}  {row['correct']:>3}/{row['items']:<3}      "
            f"{v[VerdictStatus.SUPPORTED.value]:>3} / {v[VerdictStatus.NEUTRAL.value]:>3} / "
            f"{v[VerdictStatus.REFUTED.value]:>3}"
        )

    wrong = set(summary["wrong"])
    if wrong and items and statuses:
        lines += ["", "wrong verdicts"]
        for item in items:
            if item.id in wrong:
                lines.append(
                    f"  {item.id:<9} {item.kind.value:<14} -> {statuses[item.id].value:<9} "
                    f"{item.claim[:70]}"
                )
    return "\n".join(lines)
