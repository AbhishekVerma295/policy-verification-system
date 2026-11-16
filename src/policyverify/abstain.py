"""
abstain.py - deciding whether to answer at all, and what to stand behind.

Two decisions live here, and they are different:

  WHICH CLAIMS TO KEEP   a claim survives only if verification supported it
                         AND the relevance gate found it answers the question
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

    The same goes for a claim withheld for being off topic: it is TRUE of its
    passage, so it is listed as removed with that reason rather than dropped,
    and if nothing on topic is left the reason says the policies do not
    appear to contain what was asked.

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

    @property
    def off_topic(self) -> list[ClaimVerdict]:
        """Removed claims that were TRUE but did not answer the question.

        Kept apart from refuted and unsupported ones: nothing is wrong with
        these claims, they are simply not what was asked, and saying so is
        more honest than calling them unverified.
        """
        return [v for v in self.removed if v.checks.relevant is False]


def should_keep(verdict: ClaimVerdict, config: Config | None = None) -> bool:
    """Does this claim survive verification?

    Deliberately strict: a claim is kept only if it was positively SUPPORTED
    and confidently so. NEUTRAL means the evidence did not settle it, which is
    not the same as the claim being true, and REFUTED means the cited text
    says otherwise.

    A supported claim the relevance gate marked off topic (`relevant is
    False`) is not kept either: true, but not an answer to the question. A
    claim the gate never judged (`relevant is None`) is unaffected, which is
    what keeps saved runs and gate-off runs behaving as before.
    """
    config = config or get_config()
    return (
        verdict.status is VerdictStatus.SUPPORTED
        and verdict.score >= config.abstention.min_claim_score
        and verdict.checks.relevant is not False
    )


def decide(
    verdicts: list[ClaimVerdict],
    retrieved: list[RetrievedChunk] | None = None,
    config: Config | None = None,
) -> AbstentionDecision:
    """Split claims into kept and removed, and decide whether to answer."""
    config = config or get_config()
    cfg = config.abstention

    # --- Nothing retrieved -------------------------------------------------
    # Checked before looking at claims: with no passage at all, any claims
    # produced were not grounded in anything worth standing behind.
    #
    # There used to be a second check here: decline if the best passage's
    # similarity was below `min_retrieval_score` (0.25). It was removed
    # after measuring it (scripts/calibrate.py --retrieval, tuning half
    # only, Nov 2026), because it never fired and no value of it is safe:
    #   - the lowest best-passage score of ANY question measured, gibberish
    #     included, was 0.31, so 0.25 could never trigger;
    #   - in-domain unanswerable questions score like answerable ones (7
    #     tuning unanswerable: 0.46-0.65; 26 answerable: 0.46-0.76), so no
    #     retrieval threshold separates them - the verifier and the
    #     relevance gate are what catch those;
    #   - off-topic and gibberish questions (14 written for the script)
    #     scored 0.31-0.57, overlapping the lowest answerable question
    #     (0.46), and all 14 were ALREADY declined end to end by the later
    #     stages (no claims made, or the relevance gate);
    #   - the check ran after generation, so it never saved any time.
    # A threshold high enough to fire (about 0.43) would sit 0.03-0.06 from
    # real answerable questions, buy no measured gain, and risk declining
    # good answers. The setting is kept in config.py, unread, so existing
    # config.yaml files still load; see AbstentionConfig.
    if retrieved is not None and not retrieved:
        return AbstentionDecision(
            abstained=True,
            reason="No relevant policy passages were found for this question.",
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

    # The claims were true but none of them answered the question: the
    # corpus does not contain what was asked. Say that, rather than the
    # misleading "could not be verified".
    off_topic = len(decision.off_topic)
    if off_topic and not kept:
        others = total - off_topic
        extra = f" ({others} other claim(s) could not be verified.)" if others else ""
        decision.removed = removed + kept
        decision.reason = (
            f"The policies do not appear to contain what was asked: {off_topic} "
            f"statement(s) found were supported by the policy but do not answer "
            f"the question.{extra} They are shown below rather than hidden."
        )
        return decision

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
