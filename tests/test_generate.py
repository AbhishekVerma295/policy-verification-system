"""
Tests for prompt building, structured-output parsing, and the LLM boundary.

All of it runs against FakeLLM (see conftest.py), so no GPU, no Ollama and no
network. The point of these tests is the plumbing: does a malformed reply
fail loudly, does a fabricated citation get reported rather than swallowed,
does an empty retrieval avoid asking the model at all.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeLLM
from policyverify.config import load_config
from policyverify.generate import (
    GenerationError,
    build_prompt,
    drop_unknown_citations,
    extract_json,
    format_passages,
    generate_answer_draft,
    generate_claims,
    parse_draft,
)
from policyverify.llm import LLMError, get_llm
from policyverify.schema import Chunk, DraftAnswer, PolicyType, RetrievedChunk


def _retrieved(citation: str = "srm/attendance/7.3-minimum_attendance") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{citation}#0",
        text="A student must maintain a minimum attendance record of at least 75%.",
        citation_id=citation,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Academic Regulations > Minimum Attendance",
        source_url="https://example.srmist.edu.in/regs",
    )
    return RetrievedChunk(chunk=chunk, score=0.9, rank=0)


def _reply(claims: list[dict]) -> str:
    return json.dumps({"claims": claims})


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_labels_each_passage_with_its_citation_id():
    prompt = build_prompt("What is the attendance rule?", [_retrieved()])
    assert "citation_id: srm/attendance/7.3-minimum_attendance" in prompt
    assert "What is the attendance rule?" in prompt


def test_prompt_tells_the_model_to_treat_passages_as_data():
    """Prompt-injection defence. This system reads documents it did not
    write, and a document can contain text shaped like a command."""
    prompt = build_prompt("q", [_retrieved()])
    lowered = prompt.lower()
    assert "reference material" in lowered
    assert "ignore" in lowered


def test_prompt_asks_for_atomic_claims_and_allows_an_empty_answer():
    prompt = build_prompt("q", [_retrieved()])
    assert "SEPARATE FACTUAL CLAIMS" in prompt
    assert "empty claims list" in prompt


def test_format_passages_truncates_very_long_chunks():
    long_chunk = _retrieved()
    long_chunk.chunk.text = "x" * 5000
    rendered = format_passages([long_chunk], max_chars=100)
    assert "x" * 100 in rendered
    assert "x" * 200 not in rendered


# ---------------------------------------------------------------------------
# Parsing the reply
# ---------------------------------------------------------------------------


def test_extract_json_handles_a_bare_object():
    assert extract_json('{"claims": []}') == {"claims": []}


def test_extract_json_handles_markdown_fences():
    raw = 'Here you go:\n```json\n{"claims": []}\n```'
    assert extract_json(raw) == {"claims": []}


def test_extract_json_strips_thinking_tags():
    """Qwen3 can emit reasoning inline. It must not break parsing."""
    raw = '<think>Let me consider the passages...</think>{"claims": []}'
    assert extract_json(raw) == {"claims": []}


def test_extract_json_returns_none_on_prose():
    assert extract_json("The attendance requirement is 75 percent.") is None


def test_parse_draft_validates_against_the_schema():
    draft = parse_draft(
        _reply([{"text": "Attendance is 75%.", "citation_ids": ["srm/attendance/7.3-x"]}])
    )
    assert isinstance(draft, DraftAnswer)
    assert draft.claims[0].text == "Attendance is 75%."


def test_parse_draft_rejects_the_wrong_shape():
    """Prose where a claims list belongs must fail, not slip through."""
    assert parse_draft('{"claims": "attendance is 75%"}') is None


# ---------------------------------------------------------------------------
# Fabricated citations
# ---------------------------------------------------------------------------


def test_fabricated_citations_are_reported_not_silently_dropped():
    """A citation pointing at nothing is a finding about the model, not noise.
    If ingestion discarded it, fabrication could never be measured."""
    draft = DraftAnswer.model_validate(
        {
            "claims": [
                {
                    "text": "Attendance is 75%.",
                    "citation_ids": [
                        "srm/attendance/7.3-minimum_attendance",
                        "srm/attendance/99-invented-section",
                    ],
                }
            ]
        }
    )
    cleaned, unknown = drop_unknown_citations(draft, [_retrieved()])
    assert unknown == ["srm/attendance/99-invented-section"]
    assert cleaned.claims[0].citation_ids == ["srm/attendance/7.3-minimum_attendance"]


def test_claims_with_only_fabricated_citations_survive_with_none_left():
    """The claim is kept so the verifier can mark it unsupported - dropping it
    here would hide the failure instead of reporting it."""
    draft = DraftAnswer.model_validate(
        {"claims": [{"text": "Invented.", "citation_ids": ["srm/x/made-up"]}]}
    )
    cleaned, unknown = drop_unknown_citations(draft, [_retrieved()])
    assert len(cleaned.claims) == 1
    assert cleaned.claims[0].citation_ids == []
    assert unknown == ["srm/x/made-up"]


# ---------------------------------------------------------------------------
# generate_claims
# ---------------------------------------------------------------------------


def test_generate_claims_returns_parsed_claims():
    llm = FakeLLM([_reply([{"text": "Attendance is 75%.", "citation_ids": []}])])
    draft = generate_claims("q", [_retrieved()], llm=llm, config=load_config())
    assert len(draft.claims) == 1
    assert llm.call_count == 1


def test_generate_claims_does_not_call_the_model_with_no_passages():
    """Nothing retrieved means nothing to ground a claim in. Asking anyway
    would invite the model to answer from memory - the exact failure mode
    this project exists to prevent."""
    llm = FakeLLM([_reply([{"text": "From memory!", "citation_ids": []}])])
    draft = generate_claims("q", [], llm=llm, config=load_config())
    assert draft.claims == []
    assert llm.call_count == 0


def test_generate_claims_retries_once_on_unparseable_output():
    llm = FakeLLM(["not json at all", _reply([{"text": "Recovered.", "citation_ids": []}])])
    draft = generate_claims("q", [_retrieved()], llm=llm, config=load_config())
    assert draft.claims[0].text == "Recovered."
    assert llm.call_count == 2


def test_generate_claims_raises_after_retries_are_exhausted():
    """Never guess. A reply we could not read is a failure, and pretending
    otherwise would hide quiet wrongness."""
    llm = FakeLLM(["still not json"])
    with pytest.raises(GenerationError, match="unparseable"):
        generate_claims("q", [_retrieved()], llm=llm, config=load_config())


def test_retry_prompt_differs_from_the_first_attempt():
    llm = FakeLLM(["nope", _reply([])])
    generate_claims("q", [_retrieved()], llm=llm, config=load_config())
    assert llm.prompts[0] != llm.prompts[1]
    assert "could not be parsed" in llm.prompts[1]


def test_generate_answer_draft_returns_claims_and_fabrications():
    llm = FakeLLM(
        [
            _reply(
                [
                    {"text": "Real.", "citation_ids": ["srm/attendance/7.3-minimum_attendance"]},
                    {"text": "Invented.", "citation_ids": ["srm/nope/nope"]},
                ]
            )
        ]
    )
    draft, fabricated = generate_answer_draft(
        "q", [_retrieved()], llm=llm, config=load_config()
    )
    assert len(draft.claims) == 2
    assert fabricated == ["srm/nope/nope"]


# ---------------------------------------------------------------------------
# The LLM boundary
# ---------------------------------------------------------------------------


def test_get_llm_rejects_an_unknown_backend():
    config = load_config()
    config.llm.backend = "definitely-not-a-backend"
    with pytest.raises(LLMError, match="unknown llm backend"):
        get_llm(config)


def test_get_llm_builds_the_ollama_backend():
    config = load_config()
    config.llm.backend = "ollama"
    backend = get_llm(config)
    assert backend.model == config.llm.model
