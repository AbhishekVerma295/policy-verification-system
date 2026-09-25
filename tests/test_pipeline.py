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


def test_abstains_when_retrieval_is_too_weak():
    answer = answer_question(
        "Something the corpus does not cover.",
        config=load_config(),
        store=StubStore(score=0.01),
        llm=_llm_saying("An unsupported assertion."),
        nli=StubNLI(),
    )
    assert answer.abstained
    assert "does not appear to cover" in answer.reason


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
