"""
Tests for the end-to-end pipeline.

Everything expensive is substituted: a stub store, a FakeLLM and a stub NLI
checker. That is the point of making them injectable - the whole
question-to-Answer path runs here in milliseconds with nothing downloaded,
so the wiring is actually covered rather than only exercised by hand.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeLLM
from policyverify.config import load_config
from policyverify.generate import GenerationError
from policyverify.pipeline import answer_question
from policyverify.schema import Chunk, PolicyType, RetrievedChunk, VerdictStatus
from policyverify.verify.nli import ENTAILMENT, NEUTRAL, NLIResult

CITATION = "srm/attendance/7.3-minimum_attendance"
PASSAGE = "A student must maintain a minimum attendance record of at least 75%."


def _chunk() -> Chunk:
    return Chunk(
        chunk_id=f"{CITATION}#0",
        text=PASSAGE,
        citation_id=CITATION,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Regulations > Minimum Attendance",
        source_url="https://example.srmist.edu.in/regs",
    )


class StubStore:
    """Returns fixed search results, so no index or embedding model is needed."""

    def __init__(self, score: float = 0.8, results: int = 1):
        self.score = score
        self.results = results

    def search(self, question, k=None, university=None, policy_type=None):
        return [
            RetrievedChunk(chunk=_chunk(), score=self.score, rank=i)
            for i in range(self.results)
        ]

    def get_by_citation(self, citation_id):
        return [_chunk()] if citation_id == CITATION else []

    def read_manifest(self):
        return None  # no entity weights, so no re-ranking


class StubNLI:
    def __init__(self, label: str = ENTAILMENT, entailment: float = 0.95):
        self._result = NLIResult(
            label=label,
            entailment=entailment,
            neutral=1.0 - entailment,
            contradiction=0.0,
        )

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        return self._result


def _reply(claims: list[dict]) -> str:
    return json.dumps({"claims": claims})


def _llm_saying(text: str, citations: list[str] | None = None) -> FakeLLM:
    return FakeLLM([_reply([{"text": text, "citation_ids": citations or [CITATION]}])])


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_answers_a_supported_question_end_to_end():
    answer = answer_question(
        "What is the minimum attendance requirement?",
        config=load_config(),
        store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."),
        nli=StubNLI(),
    )
    assert not answer.abstained
    assert len(answer.claims_kept) == 1
    assert answer.claims_kept[0].status is VerdictStatus.SUPPORTED
    assert answer.all_citations() == [CITATION]


def test_answer_records_timings_for_every_stage():
    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
    )
    assert answer.timings.retrieve_ms > 0
    assert answer.timings.generate_ms > 0
    assert answer.timings.verify_ms > 0
    assert answer.timings.total_ms == pytest.approx(
        answer.timings.retrieve_ms + answer.timings.generate_ms + answer.timings.verify_ms
        + answer.timings.compose_ms
    )


def test_answer_records_the_config_that_produced_it():
    """So a saved run can still be understood months later."""
    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
    )
    assert answer.config_fingerprint["llm_model"]
    assert answer.config_fingerprint["embedding_model"]


def test_university_filter_is_recorded_on_the_answer():
    answer = answer_question(
        "q", university="srm", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
    )
    assert answer.university_filter == "srm"


# ---------------------------------------------------------------------------
# Abstention paths
# ---------------------------------------------------------------------------


def test_abstains_when_the_model_makes_no_claims():
    answer = answer_question(
        "What is the wifi password?",
        config=load_config(),
        store=StubStore(),
        llm=FakeLLM([_reply([])]),
        nli=StubNLI(),
    )
    assert answer.abstained
    assert answer.claims_kept == []
    assert answer.reason


def test_weak_retrieval_is_left_to_verification():
    """The retrieval-score gate is retired (see abstain.decide): a weakly
    matching passage no longer forces a refusal by itself. A claim the
    passage does not support is still removed by verification."""
    answer = answer_question(
        "Something the corpus does not cover.",
        config=load_config(),
        store=StubStore(score=0.01),
        llm=_llm_saying("An unsupported assertion."),
        nli=StubNLI(label=NEUTRAL, entailment=0.05),
    )
    assert answer.abstained
    assert "does not appear to cover" not in (answer.reason or "")
    assert len(answer.claims_removed) == 1


def test_abstains_when_nothing_survives_verification():
    answer = answer_question(
        "q",
        config=load_config(),
        store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."),
        nli=StubNLI(label=NEUTRAL, entailment=0.05),
    )
    assert answer.abstained
    assert len(answer.claims_removed) == 1, "the rejected claim must stay visible"


def test_abstention_is_a_result_not_an_error():
    """An abstention still returns a full Answer, with the reason and the
    claims that were considered - callers should not have to catch anything."""
    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=FakeLLM([_reply([])]), nli=StubNLI(),
    )
    assert answer.question == "q"
    assert answer.reason is not None
    assert answer.timings.total_ms >= 0


# ---------------------------------------------------------------------------
# Transparent correction
# ---------------------------------------------------------------------------


def test_unsupported_claims_are_reported_not_dropped():
    llm = FakeLLM(
        [
            _reply(
                [
                    {"text": "Students need 75% attendance.", "citation_ids": [CITATION]},
                    {"text": "Students must pay a fine.", "citation_ids": []},
                ]
            )
        ]
    )
    answer = answer_question(
        "q", config=load_config(), store=StubStore(), llm=llm, nli=StubNLI()
    )
    assert len(answer.claims_kept) == 1
    assert len(answer.claims_removed) == 1
    assert answer.hallucination_count == 1
    assert answer.claims_removed[0].claim.text == "Students must pay a fine."


def test_removed_refuted_claim_carries_the_policy_sentence():
    """A wrong claim that was removed still shows the policy sentence that
    contradicted it, so the removed-claims panel can quote it."""
    from policyverify.verify.nli import CONTRADICTION

    class ContradictingNLI:
        def check(self, premise, hypothesis):
            return NLIResult(
                label=CONTRADICTION, entailment=0.01, neutral=0.03, contradiction=0.96,
                strongest_contradiction=0.96, contradiction_window=PASSAGE, window=PASSAGE,
            )

    answer = answer_question(
        "Why 90%?", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 90% attendance."), nli=ContradictingNLI(),
    )
    removed = answer.claims_removed[0]
    assert removed.status is VerdictStatus.REFUTED
    assert removed.checks.evidence_text == PASSAGE
    assert PASSAGE in removed.explanation
    assert CITATION in removed.explanation


def test_fabricated_citations_are_surfaced_in_the_answer():
    """A citation pointing at nothing is a finding about the model, so it
    belongs in the answer rather than only in the logs."""
    llm = _llm_saying("Invented rule.", citations=["srm/attendance/99-not-real"])
    answer = answer_question(
        "q", config=load_config(), store=StubStore(), llm=llm, nli=StubNLI()
    )
    assert "do not exist" in answer.reason
    assert "srm/attendance/99-not-real" in answer.reason


def test_answer_serialises_and_round_trips():
    """Every run gets written to a JSONL log in Phase 8, so this must hold."""
    from policyverify.schema import Answer

    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
    )
    restored = Answer.model_validate_json(answer.model_dump_json())
    assert restored.question == answer.question
    assert restored.claims_kept[0].status is VerdictStatus.SUPPORTED


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_unparseable_model_output_raises_rather_than_guessing():
    with pytest.raises(GenerationError):
        answer_question(
            "q",
            config=load_config(),
            store=StubStore(),
            llm=FakeLLM(["this is not json"]),
            nli=StubNLI(),
        )


def test_on_stage_reports_each_stage_in_order():
    seen: list[str] = []
    answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
        on_stage=seen.append,
    )
    assert seen == ["retrieve", "generate", "verify", "decide"]


# ---------------------------------------------------------------------------
# The relevance gate
# ---------------------------------------------------------------------------


def _gate_saying(relevant: bool, judge_failed: bool = False):
    """A relevance gate that marks every supported claim as `relevant`."""
    from policyverify.verify.relevance import OFF_TOPIC_NOTE, RelevanceReport

    def gate(question, verdicts):
        out = []
        for v in verdicts:
            checks = v.checks.model_copy(update={"relevant": relevant})
            update = {"checks": checks}
            if not relevant:
                update["explanation"] = f"{OFF_TOPIC_NOTE} {v.explanation}"
            out.append(v.model_copy(update=update))
        return RelevanceReport(verdicts=out, judge_failed=judge_failed)

    return gate


def test_off_topic_answer_is_declined_with_its_own_reason():
    answer = answer_question(
        "Who is the current Vice Chancellor?", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
        relevance=_gate_saying(False),
    )
    assert answer.abstained
    assert answer.claims_kept == []
    assert "do not appear to contain what was asked" in answer.reason
    # nothing generated disappears: the true-but-off-topic claim is shown
    assert len(answer.claims_removed) == 1
    assert answer.claims_removed[0].status is VerdictStatus.SUPPORTED
    assert answer.claims_removed[0].checks.relevant is False
    assert "does not answer the question" in answer.claims_removed[0].explanation


def test_relevant_answer_is_kept_when_the_gate_approves():
    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
        relevance=_gate_saying(True),
    )
    assert not answer.abstained
    assert answer.claims_kept[0].checks.relevant is True


def test_gate_runs_after_verification_and_sees_only_the_question_and_verdicts():
    seen = {}

    def gate(question, verdicts):
        from policyverify.verify.relevance import RelevanceReport

        seen["question"] = question
        seen["statuses"] = [v.status for v in verdicts]
        return RelevanceReport(verdicts=verdicts)

    answer_question(
        "What is the attendance rule?", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(), relevance=gate,
    )
    assert seen["question"] == "What is the attendance rule?"
    assert seen["statuses"] == [VerdictStatus.SUPPORTED]


def test_gate_time_is_recorded_and_counted_inside_verify_time():
    import time

    def slow_gate(question, verdicts):
        from policyverify.verify.relevance import RelevanceReport

        time.sleep(0.02)
        return RelevanceReport(verdicts=verdicts)

    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(), relevance=slow_gate,
    )
    t = answer.timings
    assert t.relevance_ms >= 15
    assert t.verify_ms >= t.relevance_ms
    # included in verify_ms, so not added to the total a second time
    assert t.total_ms == pytest.approx(
        t.retrieve_ms + t.generate_ms + t.verify_ms + t.compose_ms
    )


def test_a_failed_judge_is_recorded_on_the_answer():
    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
        relevance=_gate_saying(True, judge_failed=True),
    )
    assert answer.relevance_judge_failed


def test_default_gate_with_a_store_that_has_no_embedder_does_not_crash():
    """Stub stores have no embedding model; the real gate then has only the
    language model, and an unusable reply leaves claims unchecked and flagged
    rather than raising."""
    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
    )
    assert not answer.abstained
    assert answer.claims_kept[0].checks.relevance_source == "unchecked"
    assert answer.relevance_judge_failed


def test_gate_switched_off_leaves_the_answer_as_verification_made_it():
    config = load_config()
    config.relevance.enabled = False
    answer = answer_question(
        "q", config=config, store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
    )
    assert not answer.abstained
    assert answer.claims_kept[0].checks.relevant is None
    assert not answer.relevance_judge_failed


def test_answers_saved_before_the_gate_existed_still_load():
    from policyverify.schema import Answer

    old = {
        "question": "q",
        "claims_kept": [
            {
                "claim": {"text": "Students need 75% attendance.", "citation_ids": [CITATION]},
                "status": "SUPPORTED",
                "score": 0.9,
                "checks": {"nli_label": "entailment", "citation_exists": True},
            }
        ],
        "timings": {"retrieve_ms": 1.0, "generate_ms": 2.0, "verify_ms": 3.0},
    }
    answer = Answer.model_validate(old)
    assert answer.claims_kept[0].checks.relevant is None
    assert answer.timings.relevance_ms == 0.0
    assert answer.relevance_judge_failed is False


def test_baseline_without_verification_counts_relevance_removed_claims_as_generated():
    """The verification-off baseline means "nothing was checked", so a claim
    withheld for being off topic is part of what the model generated."""
    from policyverify.evaluate import without_verification

    answer = answer_question(
        "q", config=load_config(), store=StubStore(),
        llm=_llm_saying("Students need 75% attendance."), nli=StubNLI(),
        relevance=_gate_saying(False),
    )
    assert answer.abstained
    baseline = without_verification(answer)
    assert not baseline.abstained
    assert [v.claim.text for v in baseline.claims_kept] == ["Students need 75% attendance."]
