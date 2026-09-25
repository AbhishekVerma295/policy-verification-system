"""
abstain.py - deciding whether to answer at all, and what to stand behind.

Two decisions live here, and they are different:

  WHICH CLAIMS TO KEEP   a claim survives only if verification supported it
  WHETHER TO ANSWER      if too little survives, decline the whole question

REMOVED CLAIMS ARE KEPT, NOT DELETED
    This is the transparency feature, and it is the reason `Answer` has a
    `claims_removed` field at all. The user sees what the system started to
    say and then decided it could not stand behind.

    Silently dropping them would produce a shorter, cleaner-looking answer
    that is strictly less honest: the reader could not tell the difference
    between "the policy does not mention this" and "we checked and it was
    wrong". Showing the removals is what makes the system auditable rather
    than merely trustworthy-looking.

WHY OVER-ABSTENTION IS ALSO A FAILURE
    A system that refuses everything scores perfectly on hallucination rate
    and is useless. The thresholds here are therefore something to tune
    against real questions, not to set as high as possible - and Phase 8
    measures false abstention explicitly, as its own number, so that tuning
    has something honest to optimise against.

    Tune on the tuning half of the adversarial set, report on the held-out
    half. Tuning and reporting on the same questions would only prove the
    system works on the questions it was tuned for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from policyverify.config import Config, get_config
from policyverify.schema import ClaimVerdict, RetrievedChunk, VerdictStatus


@dataclass
class AbstentionDecision:
    """What survived verification, and whether that is enough to answer."""

    abstained: bool
    reason: str | None
    kept: list[ClaimVerdict] = field(default_factory=list)
    removed: list[ClaimVerdict] = field(default_factory=list)

    @property
    def refuted(self) -> list[ClaimVerdict]:
        """Removed claims the policy text actively contradicted.

        Worth separating from merely-unsupported ones. "The policy says the
        opposite" is a more serious finding than "the policy does not say",
        and it is the single most useful thing to show a reader.
        """
        return [v for v in self.removed if v.status is VerdictStatus.REFUTED]

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        """Removed claims that simply could not be confirmed either way."""
        return [v for v in self.removed if v.status is not VerdictStatus.REFUTED]


def should_keep(verdict: ClaimVerdict, config: Config | None = None) -> bool:
    """Does this claim survive verification?

    Deliberately strict: a claim is kept only if it was positively SUPPORTED
    and confidently so. NEUTRAL means the evidence did not settle it, which is
    not the same as the claim being true, and REFUTED means the cited text
    says otherwise.
    """
    config = config or get_config()
    return (
        verdict.status is VerdictStatus.SUPPORTED
        and verdict.score >= config.abstention.min_claim_score
    )


def decide(
    verdicts: list[ClaimVerdict],
    retrieved: list[RetrievedChunk] | None = None,
    config: Config | None = None,
) -> AbstentionDecision:
    """Split claims into kept and removed, and decide whether to answer."""
    config = config or get_config()
    cfg = config.abstention

    # --- Nothing retrieved, or nothing relevant enough ---------------------
    # Checked before looking at claims: if the corpus does not cover the
    # question, any claims produced were not grounded in anything worth
    # standing behind, whatever the verifier made of them.
    if retrieved is not None:
        if not retrieved:
            return AbstentionDecision(
                abstained=True,
                reason="No relevant policy passages were found for this question.",
                kept=[],
                removed=list(verdicts),
            )
        best = max(r.score for r in retrieved)
        if best < cfg.min_retrieval_score:
            return AbstentionDecision(
                abstained=True,
                reason=(
                    f"The policy corpus does not appear to cover this question "
                    f"(best passage scored {best:.2f}, below the "
                    f"{cfg.min_retrieval_score:.2f} required)."
                ),
                kept=[],
                removed=list(verdicts),
            )

    # --- The model produced nothing ---------------------------------------
    if not verdicts:
        return AbstentionDecision(
            abstained=True,
            reason=(
                "The retrieved passages do not answer this question, so no "
                "claims were made."
            ),
        )

    # --- Partition -------------------------------------------------------
    kept = [v for v in verdicts if should_keep(v, config)]
    removed = [v for v in verdicts if not should_keep(v, config)]

    if len(kept) >= cfg.min_supported_claims:
        return AbstentionDecision(abstained=False, reason=None, kept=kept, removed=removed)

    # --- Not enough survived ---------------------------------------------
    decision = AbstentionDecision(abstained=True, reason=None, kept=[], removed=removed)
    refuted = len(decision.refuted)
    total = len(verdicts)

    if refuted:
        detail = (
            f"{refuted} of {total} were contradicted by the policy text"
            if refuted < total
            else f"all {total} were contradicted by the policy text"
        )
    else:
        detail = f"none of the {total} could be confirmed against the cited policy text"

    # Claims that survived checking but fell short of the minimum are moved
    # into `removed` too, so nothing generated disappears without a trace.
    decision.removed = removed + kept
    decision.reason = (
        f"Not enough of this answer could be verified - {detail}. "
        f"The unverified claims are shown below rather than hidden."
    )
    return decision
