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
"""

from __future__ import annotations

from policyverify.config import Config, get_config
from policyverify.schema import (
    CheckResults,
    Claim,
    ClaimVerdict,
    RetrievedChunk,
    VerdictStatus,
)
from policyverify.verify.citation import CitationCheck, check_citations
from policyverify.verify.nli import CONTRADICTION, ENTAILMENT, NLIChecker, NLIResult
from policyverify.verify.numeric import check_numbers


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
    best: NLIResult | None = None
    for premise in premises:
        result = nli.check(premise, claim.text)
        if best is None or result.entailment > best.entailment:
            best = result
        # A contradiction is worth keeping if nothing has entailed yet.
        if best.entailment < config.nli.entailment_threshold and (
            result.contradiction > best.contradiction
        ):
            best = result

    assert best is not None  # premises is non-empty, so the loop ran

    # --- numeric guard: can veto, never grant ---------------------------
    numeric = check_numbers(claim.text, premises)

    checks = CheckResults(
        nli_label=best.label,
        nli_score=round(best.score, 4),
        numeric_ok=numeric.ok,
        numeric_detail=numeric.detail,
        citation_exists=True,
        citation_supports=best.label == ENTAILMENT,
    )

    fabricated = [c for c in citation_checks if c.fabricated]
    threshold = config.nli.entailment_threshold
    chunk_ids = [chunk.chunk_id for chunk in evidence_chunks]

    # --- decide ---------------------------------------------------------
    if best.label == CONTRADICTION and best.contradiction >= threshold:
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.REFUTED,
            score=round(best.contradiction, 4),
            evidence_chunk_ids=chunk_ids,
            checks=checks,
            explanation=(
                f"The cited policy text contradicts this claim "
                f"(confidence {best.contradiction:.2f})."
            ),
        )

    if best.label == ENTAILMENT and best.entailment >= threshold:
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
        return ClaimVerdict(
            claim=claim,
            status=VerdictStatus.SUPPORTED,
            score=round(best.entailment, 4),
            evidence_chunk_ids=chunk_ids,
            checks=checks,
            explanation=(
                f"The cited policy text supports this claim "
                f"(confidence {best.entailment:.2f}).{note}"
            ),
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
