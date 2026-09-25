"""
Tests for Phase 5: abstention and transparent correction.

Pure decision logic over already-computed verdicts, so no model is involved
and the whole file runs in milliseconds.

The most important property asserted here is that nothing generated ever
disappears silently: every claim the model produced ends up in either
`kept` or `removed`. That invariant is the transparency feature, and it is
easy to break by accident when adding a new abstention branch.
"""

from __future__ import annotations

from policyverify.abstain import AbstentionDecision, decide, should_keep
from policyverify.config import load_config
from policyverify.schema import (
    Chunk,
    Claim,
    ClaimVerdict,
    PolicyType,
    RetrievedChunk,
    VerdictStatus,
)

CITATION = "srm/attendance/7.3-minimum_attendance"


def _verdict(
    text: str = "Students need 75% attendance.",
    status: VerdictStatus = VerdictStatus.SUPPORTED,
    score: float = 0.95,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(text=text, citation_ids=[CITATION]),
        status=status,
        score=score,
        evidence_chunk_ids=[f"{CITATION}#0"],
        explanation="test verdict",
    )


def _retrieved(score: float = 0.8) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{CITATION}#0",
        text="A student must maintain at least 75% attendance.",
        citation_id=CITATION,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Regulations > Minimum Attendance",
        source_url="https://example.srmist.edu.in/regs",
    )
    return RetrievedChunk(chunk=chunk, score=score, rank=0)


# ---------------------------------------------------------------------------
# should_keep
# ---------------------------------------------------------------------------


def test_supported_and_confident_claims_are_kept():
    assert should_keep(_verdict(score=0.95), load_config())


def test_supported_but_weak_claims_are_not_kept():
    assert not should_keep(_verdict(score=0.20), load_config())


def test_neutral_claims_are_not_kept():
    """"Could not be confirmed" is not the same as "true"."""
    assert not should_keep(
        _verdict(status=VerdictStatus.NEUTRAL, score=0.9), load_config()
    )


def test_refuted_claims_are_not_kept():
    assert not should_keep(
        _verdict(status=VerdictStatus.REFUTED, score=0.99), load_config()
    )


# ---------------------------------------------------------------------------
# The core invariant
# ---------------------------------------------------------------------------


def test_no_claim_ever_disappears():
    """Every generated claim must end up in kept or removed. Silently losing
    one would make the answer look cleaner while being less honest."""
    verdicts = [
        _verdict("Supported claim."),
        _verdict("Refuted claim.", VerdictStatus.REFUTED, 0.99),
        _verdict("Unverified claim.", VerdictStatus.NEUTRAL, 0.1),
        _verdict("Weakly supported.", VerdictStatus.SUPPORTED, 0.2),
    ]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert len(decision.kept) + len(decision.removed) == len(verdicts)


def test_no_claim_disappears_even_when_abstaining():
    verdicts = [
        _verdict("Refuted.", VerdictStatus.REFUTED, 0.99),
        _verdict("Unverified.", VerdictStatus.NEUTRAL, 0.1),
    ]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert decision.abstained
    assert len(decision.removed) == 2


def test_claims_are_removed_not_deleted_on_weak_retrieval():
    verdicts = [_verdict()]
    decision = decide(verdicts, [_retrieved(score=0.05)], load_config())
    assert decision.abstained
    assert len(decision.removed) == 1, "the claim must still be visible"


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------


def test_answers_when_a_claim_is_supported():
    decision = decide([_verdict()], [_retrieved()], load_config())
    assert not decision.abstained
    assert decision.reason is None
    assert len(decision.kept) == 1


def test_keeps_good_claims_and_removes_bad_ones_in_the_same_answer():
    """Partial answers are the normal case - one bad claim should not throw
    away the good ones."""
    verdicts = [
        _verdict("Good claim."),
        _verdict("Bad claim.", VerdictStatus.REFUTED, 0.99),
    ]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert not decision.abstained
    assert [v.claim.text for v in decision.kept] == ["Good claim."]
    assert [v.claim.text for v in decision.removed] == ["Bad claim."]


# ---------------------------------------------------------------------------
# Abstaining
# ---------------------------------------------------------------------------


def test_abstains_when_nothing_was_retrieved():
    decision = decide([], [], load_config())
    assert decision.abstained
    assert "No relevant policy passages" in decision.reason


def test_abstains_when_retrieval_is_too_weak():
    """If the corpus does not cover the question, claims built on it were not
    grounded in anything worth standing behind."""
    decision = decide([_verdict()], [_retrieved(score=0.01)], load_config())
    assert decision.abstained
    assert "does not appear to cover" in decision.reason


def test_abstains_when_the_model_produced_no_claims():
    decision = decide([], [_retrieved()], load_config())
    assert decision.abstained
    assert "do not answer this question" in decision.reason


def test_abstains_when_no_claim_survives_verification():
    verdicts = [_verdict("Unverified.", VerdictStatus.NEUTRAL, 0.1)]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert decision.abstained
    assert "none of the 1 could be confirmed" in decision.reason


def test_abstention_reason_names_contradiction_when_claims_were_refuted():
    """"The policy says the opposite" is a more serious finding than "the
    policy does not say", and the reason should reflect that."""
    verdicts = [_verdict("Wrong.", VerdictStatus.REFUTED, 0.99)]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert decision.abstained
    assert "contradicted by the policy text" in decision.reason


def test_abstention_reason_says_removals_are_visible():
    verdicts = [_verdict("Unverified.", VerdictStatus.NEUTRAL, 0.1)]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert "shown below rather than hidden" in decision.reason


def test_respects_a_higher_minimum_supported_claims_setting():
    config = load_config()
    config.abstention.min_supported_claims = 3
    decision = decide([_verdict(), _verdict()], [_retrieved()], config)
    assert decision.abstained
    # the two supported claims are still surfaced, not thrown away
    assert len(decision.removed) == 2


# ---------------------------------------------------------------------------
# Refuted vs merely unsupported
# ---------------------------------------------------------------------------


def test_decision_separates_refuted_from_unsupported():
    verdicts = [
        _verdict("Contradicted.", VerdictStatus.REFUTED, 0.99),
        _verdict("Unconfirmed.", VerdictStatus.NEUTRAL, 0.1),
    ]
    decision = decide(verdicts, [_retrieved()], load_config())
    assert [v.claim.text for v in decision.refuted] == ["Contradicted."]
    assert [v.claim.text for v in decision.unsupported] == ["Unconfirmed."]


def test_empty_decision_has_no_refuted_or_unsupported():
    decision = AbstentionDecision(abstained=False, reason=None)
    assert decision.refuted == []
    assert decision.unsupported == []


def test_retrieval_check_is_skipped_when_not_provided():
    """Callers that only have verdicts (e.g. re-scoring a saved run) can pass
    no retrieval and still get a decision."""
    decision = decide([_verdict()], None, load_config())
    assert not decision.abstained
