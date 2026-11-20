"""
verifier.py - combining the three checks into one verdict per claim.

Each claim goes through:
    1. citation.py  is the cited section real, and was it shown to the model?
    2. nli.py       does that section actually prove the claim?
    3. numeric.py   do the numbers in the claim appear in the evidence?

HOW THE CHECKS COMBINE
    The rule is deliberately conservative: a claim is SUPPORTED only when
    every check agrees. Any one objection is enough to withhold support.

    That asymmetry is the point. Wrongly marking a claim supported puts a
    false statement in front of a student with a citation attached, which is
    worse than wrongly withholding a true one - the second is visible and
    recoverable, the first is not. Phase 5 turns "not supported" into either
    a removed claim or an abstention, both of which the user can see.

    The numeric guard can VETO but never GRANT support. It is a cheap
    signal - "the digits appear somewhere in the passage" is not proof that
    the passage means what the claim says. Only the NLI model can grant
    support, and only the numeric check can veto it on numbers.

ONE AGGREGATION RULE, WHATEVER ORDER THE CITATIONS COME IN
    A claim may cite several sections, each split into chunks, and each
    chunk gets its own NLI result. They are combined by one rule:
        best support anywhere >= threshold       -> SUPPORTED (numbers permitting)
        otherwise strongest contradiction >= it  -> REFUTED
        otherwise                                -> NEUTRAL
    Taking maxima over everything means listing the citations in a
    different order can never change the verdict.

WHEN TWO CITED SECTIONS DISAGREE
    If one cited section supports the claim and a DIFFERENT cited section
    contradicts it strongly (nli.conflict_threshold, stricter than the
    normal threshold), the claim is not marked SUPPORTED. Picking the
    agreeable source and hiding the other would be exactly the kind of
    confident half-truth this project exists to prevent. The verdict is
    NEUTRAL with checks.conflict = True, and the explanation quotes both
    sentences so the reader can see the disagreement for themselves.
    Contradictions inside ONE section never count - a section that states
    a rule and then its exception is normal policy writing, not a conflict.
    Nor do contradictions from TABLE ROWS: a row of another table is about
    another category (another exam, another score band), and counting it
    withheld true scholarship claims as "conflicts" once tables were stored
    one row per line (see nli.NLIResult.conflict_strength).

THE SECTION HEADING IS PART OF THE PREMISE
    The first chunk of a section is checked with its own heading in front
    (premise_with_heading), because some sections only say who they apply
    to in the heading ("For all I Year Students:").
"""

from __future__ import annotations

from dataclasses import dataclass

from policyverify.config import Config, get_config
from policyverify.schema import (
    CheckResults,
    Claim,
    ClaimVerdict,
    RetrievedChunk,
    VerdictStatus,
)
from policyverify.verify.citation import CitationCheck, check_citations
from policyverify.verify.nli import NLIChecker, NLIResult, is_table_row
from policyverify.verify.numeric import check_numbers
from policyverify.verify.polarity import flip_negation, has_negation, polarity_undecided


@dataclass
class _Scored:
    """One cited chunk's NLI result, labelled with where it came from."""

    citation_id: str
    chunk_id: str
    result: NLIResult



# How much LESS the opposite of a negative claim must be supported than the
# claim itself (polarity.py). Measured on real claims: the two misleading
# ones had gaps of 0.10 and 0.17; correct negative paraphrases had gaps of
# 0.92 and more.
_POLARITY_MARGIN = 0.2

def _strongest(scored: list[_Scored]) -> tuple[_Scored, _Scored]:
    """The best-supporting and the most-contradicting result, across ALL
    cited passages.

    Taking a plain maximum over everything is what makes the verdict
    independent of the order the citations were listed in. (The earlier
    version updated a running "best" one premise at a time, and a strong
    contradiction from one chunk could be overwritten by a neutral from the
    next.) Ties are broken on chunk ID, so even the evidence sentence shown
    does not depend on order.
    """
    best_entail = max(scored, key=lambda s: (s.result.entailment, s.chunk_id))
    best_contra = max(scored, key=lambda s: (s.result.max_contradiction, s.chunk_id))
    return best_entail, best_contra


def _find_conflict(
    scored: list[_Scored], threshold: float, conflict_threshold: float
) -> tuple[_Scored, _Scored] | None:
    """Does one cited SECTION support the claim while another contradicts it?

    Returns (supporting result, contradicting result), or None.

    Results are grouped by citation ID first, and a section counts as
    contradicting only if it does NOT itself support the claim. That is
    what keeps a normal section safe: "students need 75% attendance ... a
    student with a medical certificate may be exempted" contains both a
    rule and its exception, and the model may well find one sentence that
    "contradicts" a claim another sentence supports. Within one section
    that is how policy is written; it only becomes a real conflict when two
    DIFFERENT sections pull in opposite directions.
    """
    by_citation: dict[str, list[_Scored]] = {}
    for s in scored:
        by_citation.setdefault(s.citation_id, []).append(s)
    if len(by_citation) < 2:
        return None

    # The contradiction used here is `conflict_strength`, which leaves out
    # table rows: a row about a different category (another exam, another
    # score band) is not a source disagreeing with the claim. See
    # nli.NLIResult.conflict_strength for the measured case.
    supporters: list[_Scored] = []
    against: list[_Scored] = []
    for group in by_citation.values():
        top_entail = max(group, key=lambda s: (s.result.entailment, s.chunk_id))
        top_contra = max(group, key=lambda s: (s.result.conflict_strength, s.chunk_id))
        if top_entail.result.entailment >= threshold:
            supporters.append(top_entail)
        elif top_contra.result.conflict_strength >= conflict_threshold:
            against.append(top_contra)

    if not supporters or not against:
        return None
    support = max(supporters, key=lambda s: (s.result.entailment, s.chunk_id))
    contra = max(against, key=lambda s: (s.result.conflict_strength, s.chunk_id))
    return support, contra


def _quote(text: str, limit: int = 300) -> str:
    """A sentence ready to drop into an explanation, shortened if very long."""
    text = " ".join(text.split())
    if not text:
        return "(text not recorded)."
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return f'"{text}"'


# Said after every quoted contradiction so nobody mistakes it for a rewrite:
# the sentence is copied from the policy, not produced by a model.
_QUOTE_NOTE = (
    "This is the sentence the checker found in conflict with the claim, "
    "quoted word for word from the policy."
)


def _section_of(chunk_id: str | None, fallback: str = "") -> str:
    """The section a chunk belongs to. Chunk IDs look like "<citation>#<n>"."""
    return (chunk_id or "").split("#")[0] or fallback


def policy_says(verdict: ClaimVerdict) -> list[str]:
    """Plain-text lines quoting the policy sentence(s) behind a wrong claim.

    Built from the recorded evidence fields, so it also works for answers
    saved before the explanation text carried the sentence. Returns an empty
    list when the verdict is not REFUTED or a conflict, when the fields are
    missing or None (old saved answers), or when the explanation already
    quotes the sentence - so callers can show the result unconditionally
    without printing it twice. Nothing is generated: every line is a verbatim
    sentence from the policy plus the section it came from.
    """
    checks = verdict.checks
    evidence = getattr(checks, "evidence_text", None)
    if not evidence or _quote(evidence) in verdict.explanation:
        return []
    evidence_section = _section_of(getattr(checks, "evidence_chunk_id", None))
    if getattr(checks, "conflict", False):
        against = getattr(checks, "contradicting_text", None)
        if not against:
            return []
        against_section = _section_of(getattr(checks, "contradicting_chunk_id", None))
        return [
            f"One section says: {_quote(evidence)} ({evidence_section})",
            f"Another section says: {_quote(against)} ({against_section})",
        ]
    if verdict.status is VerdictStatus.REFUTED:
        return [f"The policy says: {_quote(evidence)} ({evidence_section}). {_QUOTE_NOTE}"]
    return []


def premise_with_heading(chunk) -> str:
    """The chunk's text with its own section heading as a first line.

    Some sections only make sense under their heading. "For all I Year
    Students:" is the heading of a section whose items say "Full fee (except
    the cancellation fee of Rs.5000/-) will be refunded if a student wishes to
    withdraw ..." - the text never says WHO, so the true claim "a first-year
    student who withdraws ... loses Rs.5000" scored 0.00 against the item
    alone and 0.98 with the heading in front. The heading is part of the
    policy, not something we made up, so letting the checker read it is fair.

    Only the FIRST chunk of a section gets it, because that is where the
    heading really stands. A later chunk can start mid-section, or carry on
    into a following subsection that ingestion did not split off: chunk #1
    of the summer-semester section is mostly "3.7 Adding and dropping
    courses", and with "Summer Semester:" in front of it the false claim
    "students may withdraw from summer courses in the first two weeks" was
    supported. Only the last heading is added (the document title adds
    nothing), it is not repeated if the text already starts with it, and
    sections that are table rows are left alone because each row already
    carries its context.
    The NLI unit splitter treats the heading as the lead-in of any list that
    follows it (nli._lead_in_units).
    """
    text = chunk.text
    heading = (getattr(chunk, "section_path", "") or "").split(" > ")[-1].strip()
    if not heading:
        return text
    part = (getattr(chunk, "chunk_id", "") or "").rpartition("#")[2]
    if part.isdigit() and part != "0":
        return text
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    if is_table_row(first_line):
        return text
    if text.lstrip().lower().startswith(heading.rstrip(":").lower()):
        return text
    if not heading.endswith(":"):
        heading += ":"
    return f"{heading}\n{text}"


def _no_citation_verdict(claim: Claim) -> ClaimVerdict:
    """A claim with no citation at all cannot be checked, so it is not kept."""
    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.NEUTRAL,
        score=0.0,
        evidence_chunk_ids=[],
        checks=CheckResults(citation_exists=False),
        explanation="No citation given, so there is nothing to check this against.",
    )


def _fabricated_verdict(claim: Claim, checks: list[CitationCheck]) -> ClaimVerdict:
    """Every citation on this claim points at something that does not exist."""
    detail = "; ".join(c.describe() for c in checks)
    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.NEUTRAL,
        score=0.0,
        evidence_chunk_ids=[],
        checks=CheckResults(citation_exists=False, citation_supports=False),
        explanation=f"Citation could not be resolved. {detail}",
    )


def verify_claim(
    claim: Claim,
    retrieved: list[RetrievedChunk],
    nli: NLIChecker | None = None,
    store=None,
    config: Config | None = None,
) -> ClaimVerdict:
    """Check one claim against the passages it cites."""
    config = config or get_config()
    nli = nli or NLIChecker(config)

    if not claim.citation_ids:
        return _no_citation_verdict(claim)

    citation_checks = check_citations(claim.citation_ids, retrieved, store)
    resolved = [c for c in citation_checks if c.exists and c.chunks]
    if not resolved:
        return _fabricated_verdict(claim, citation_checks)

    evidence_chunks = [chunk for check in resolved for chunk in check.chunks]
    premises = [chunk.text for chunk in evidence_chunks]

    # --- NLI: the only check that can grant support ---------------------
    # Every cited chunk is scored once. The citation ID travels with each
    # result because the conflict rule below compares SECTIONS, not chunks.
    scored = [
        _Scored(check.citation_id, chunk.chunk_id, nli.check(premise_with_heading(chunk), claim.text))
        for check in resolved
        for chunk in check.chunks
    ]
    threshold = config.nli.entailment_threshold
    best_entail, best_contra = _strongest(scored)

    # --- numeric guard: can veto, never grant ---------------------------
    numeric = check_numbers(claim.text, premises)

    fabricated = [c for c in citation_checks if c.fabricated]
    chunk_ids = [chunk.chunk_id for chunk in evidence_chunks]
    supported = best_entail.result.entailment >= threshold

    # --- polarity: a negative paraphrase must beat its opposite ----------
    # See polarity.py. Measured: "The number of classes attended does not
    # affect the detention policy" scored 0.76 against a passage saying
    # attendance DOES matter, and its positive twin scored 0.66 - the passage
    # was matching words, not meaning. A word-for-word quote cannot have its
    # meaning inverted, so quotes are not tested.
    if supported and not best_entail.result.exact_quote and has_negation(claim.text):
        twin = flip_negation(claim.text)
        twin_entail = max(
            nli.check(premise_with_heading(chunk), twin).entailment
            for check in resolved
            for chunk in check.chunks
        )
        if polarity_undecided(
            best_entail.result.entailment, twin_entail, threshold, _POLARITY_MARGIN
        ):
            checks_neutral = CheckResults(
                nli_label=best_entail.result.label,
                nli_score=round(best_entail.result.score, 4),
                numeric_ok=check_numbers(claim.text, premises).ok,
                citation_exists=True,
                citation_supports=False,
                evidence_text=best_entail.result.window or None,
                evidence_chunk_id=best_entail.chunk_id,
            )
            return ClaimVerdict(
                claim=claim,
                status=VerdictStatus.NEUTRAL,
                score=round(best_entail.result.entailment * 0.5, 4),
                evidence_chunk_ids=[chunk.chunk_id for chunk in evidence_chunks],
                checks=checks_neutral,
                explanation=(
                    "The cited text does not settle this: it supports the opposite "
                    f"statement (\"{twin}\") almost as well (confidence "
                    f"{twin_entail:.2f} against {best_entail.result.entailment:.2f})."
                ),
            )

    conflict = _find_conflict(scored, threshold, config.nli.conflict_threshold) if supported else None

    # Which result to report in the checks panel: whichever decided the verdict.
    if supported:
        decisive = best_entail
    elif best_contra.result.max_contradiction >= threshold:
        decisive = best_contra
    else:
        decisive = max(
            scored,
            key=lambda s: (max(s.result.entailment, s.result.max_contradiction), s.chunk_id),
        )

    checks = CheckResults(
        nli_label=decisive.result.label,
        nli_score=round(decisive.result.score, 4),
        numeric_ok=numeric.ok,
        numeric_detail=numeric.detail,
        citation_exists=True,
        citation_supports=supported and conflict is None,
        exact_quote=supported and best_entail.result.exact_quote,
    )

    # --- decide ---------------------------------------------------------
    # One rule, applied to the best results across ALL cited passages, so
    # the order the citations were listed in cannot change the outcome:
    #   1. strongest support >= threshold  -> supported (unless the numbers
    #      disagree, or a different cited section strongly contradicts it)
    #   2. else strongest contradiction >= threshold -> refuted
    #   3. else neutral
    if supported and conflict is not None:
        support, against = conflict
        checks.conflict = True
        checks.evidence_text = support.result.window or None
        checks.evidence_chunk_id = support.chunk_id
        checks.contradicting_text = against.result.conflict_text or None
        checks.contradicting_chunk_id = against.chunk_id
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.NEUTRAL,
            # Halved, like the numeric veto below: support exists, but it is
            # not support we can stand behind.
            score=round(support.result.entailment * 0.5, 4),
            evidence_chunk_ids=chunk_ids,
            checks=checks,
            explanation=(
                f"The cited sources disagree. {support.citation_id} supports this "
                f"(confidence {support.result.entailment:.2f}): "
                f"{_quote(support.result.window)} But {against.citation_id} "
                f"contradicts it (confidence {against.result.conflict_strength:.2f}): "
                f"{_quote(against.result.conflict_text)}"
            ),
        )

    if not supported and best_contra.result.max_contradiction >= threshold:
        checks.evidence_text = best_contra.result.max_contradiction_text or None
        checks.evidence_chunk_id = best_contra.chunk_id
        confidence = best_contra.result.max_contradiction
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.REFUTED,
            score=round(confidence, 4),
            evidence_chunk_ids=chunk_ids,
            checks=checks,
            explanation=(
                f"The cited policy text contradicts this claim "
                f"(confidence {confidence:.2f}). "
                f"The policy says: {_quote(best_contra.result.max_contradiction_text)} "
                f"({best_contra.citation_id}). {_QUOTE_NOTE}"
            ),
        )

    best = best_entail.result
    if supported:
        checks.evidence_text = best.window or None
        checks.evidence_chunk_id = best_entail.chunk_id
        if not numeric.ok:
            # The passage reads as supporting the claim, but a number in the
            # claim is not in the passage. This is the "75% vs 80%" trap, and
            # the numbers win: fluent agreement does not make a figure right.
            return ClaimVerdict(
                claim=claim,
                status=VerdictStatus.NEUTRAL,
                score=round(best.entailment * 0.5, 4),
                evidence_chunk_ids=chunk_ids,
                checks=checks,
                explanation=(
                    f"The cited text appears to support this, but the numbers "
                    f"do not match: {numeric.detail}."
                ),
            )
        note = ""
        if fabricated:
            note = (
                f" (note: {len(fabricated)} other citation(s) on this claim "
                f"could not be resolved)"
            )
        how = (
            "quotes the cited policy text word for word"
            if best.exact_quote
            else f"is supported by the cited policy text (confidence {best.entailment:.2f})"
        )
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.SUPPORTED,
            score=round(best.entailment, 4),
            evidence_chunk_ids=chunk_ids,
            checks=checks,
            explanation=f"This claim {how}.{note}",
        )

    return ClaimVerdict(
        claim=claim,
        status=VerdictStatus.NEUTRAL,
        score=round(best.entailment, 4),
        evidence_chunk_ids=chunk_ids,
        checks=checks,
        explanation=(
            "The cited policy text neither clearly supports nor contradicts "
            f"this claim (support confidence {best.entailment:.2f})."
        ),
    )


def verify_claims(
    claims: list[Claim],
    retrieved: list[RetrievedChunk],
    nli: NLIChecker | None = None,
    store=None,
    config: Config | None = None,
) -> list[ClaimVerdict]:
    """Verify every claim, reusing one loaded NLI model across all of them."""
    config = config or get_config()
    nli = nli or NLIChecker(config)
    return [verify_claim(c, retrieved, nli=nli, store=store, config=config) for c in claims]
