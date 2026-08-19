"""
Tests for the data contracts.

These are the cheapest and most useful tests in the project. Everything here is
a pure function - no network, no models, no files - so the whole file runs in
well under a second and can never be flaky.

The citation tests carry the most weight. A citation we cannot parse is a
citation we cannot verify, and a system that quietly accepts a broken citation
has defeated its own purpose.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from policyverify.schema import (
    Answer,
    Chunk,
    CitationID,
    Claim,
    ClaimVerdict,
    Document,
    DraftAnswer,
    PolicyType,
    Section,
    Timings,
    VerdictStatus,
)

# ---------------------------------------------------------------------------
# CitationID - the contract everything else hangs off
# ---------------------------------------------------------------------------


def test_citation_round_trip():
    """Parsing then rendering must give back exactly what we started with."""
    raw = "uni_a/academic_integrity/4.2"
    cid = CitationID.parse(raw)
    assert cid.university == "uni_a"
    assert cid.policy == "academic_integrity"
    assert cid.section == "4.2"
    assert cid.render() == raw
    assert str(cid) == raw


def test_citation_accepts_heading_slug_sections():
    """Not every document numbers its sections, so slugs must work too."""
    cid = CitationID.parse("uni_b/code_of_conduct/plagiarism")
    assert cid.section == "plagiarism"


def test_citation_accepts_deep_numbering():
    cid = CitationID.parse("uni_a/examination/4.2.1")
    assert cid.section == "4.2.1"


@pytest.mark.parametrize(
    "bad",
    [
        "uni_a/attendance",  # only two parts
        "uni_a/attendance/4.2/extra",  # four parts
        "",  # empty
        "Uni_A/attendance/4.2",  # capitals in the university slug
        "uni a/attendance/4.2",  # space in the slug
        "uni_a//4.2",  # empty policy
        "uni_a/attendance/",  # empty section
        "1uni/attendance/4.2",  # slug must start with a letter
    ],
)
def test_citation_rejects_malformed(bad: str):
    """We are strict on purpose. Silently accepting a broken citation would
    defeat the entire point of the project."""
    with pytest.raises((ValueError, ValidationError)):
        CitationID.parse(bad)


def test_citation_is_valid_never_raises():
    """is_valid is for filtering, so it must return False rather than blow up."""
    assert CitationID.is_valid("uni_a/attendance/4.2") is True
    assert CitationID.is_valid("nonsense") is False
    assert CitationID.is_valid("") is False


def test_citation_is_frozen():
    """A citation you can edit in place is a citation you cannot trust."""
    cid = CitationID.parse("uni_a/attendance/4.2")
    with pytest.raises(ValidationError):
        cid.university = "uni_b"


def test_citation_tolerates_surrounding_whitespace():
    """Models love to add stray spaces. That should not be fatal."""
    assert CitationID.parse("  uni_a/attendance/4.2  ").section == "4.2"


# ---------------------------------------------------------------------------
# Section and Document
# ---------------------------------------------------------------------------


def test_section_key_prefers_document_numbering(sample_section: Section):
    """When the document numbers its own sections, use those numbers - that is
    what a reader would actually quote."""
    assert sample_section.section_key() == "4.2"


def test_section_key_falls_back_to_heading_slug():
    section = Section(heading="Plagiarism and Collusion", text="Some text here.")
    assert section.section_key() == "plagiarism_and_collusion"


def test_section_key_handles_awkward_headings():
    section = Section(heading="  Fees & Charges!  ", text="Some text here.")
    key = section.section_key()
    assert key == "fees_charges"
    # Whatever it produces has to survive being put into a citation.
    assert CitationID.is_valid(f"uni_a/attendance/{key}")


def test_section_rejects_empty_text():
    with pytest.raises(ValidationError):
        Section(heading="Empty", text="   ")


def test_document_builds_citation_for_section(
    sample_document: Document, sample_section: Section
):
    cid = sample_document.citation_for(sample_section)
    assert cid.render() == "uni_a/attendance/4.2"


def test_document_rejects_bad_university_slug(sample_document: Document):
    data = sample_document.model_dump()
    data["university"] = "Uni A"
    with pytest.raises(ValidationError):
        Document.model_validate(data)


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def test_chunk_validates_its_own_citation(sample_chunk: Chunk):
    assert sample_chunk.citation().section == "4.2"


def test_chunk_rejects_malformed_citation(sample_chunk: Chunk):
    data = sample_chunk.model_dump()
    data["citation_id"] = "not-a-citation"
    with pytest.raises(ValidationError):
        Chunk.model_validate(data)


def test_chunk_metadata_is_flat_primitives(sample_chunk: Chunk):
    """Chroma only accepts str/int/float/bool as metadata. Anything nested
    fails at write time, which is an annoying thing to discover late."""
    meta = sample_chunk.to_metadata()
    assert meta["university"] == "uni_a"
    assert meta["policy_type"] == "attendance"  # the enum's string value
    for key, value in meta.items():
        assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value)}"


# ---------------------------------------------------------------------------
# Claim and DraftAnswer - the contract the language model must satisfy
# ---------------------------------------------------------------------------


def test_claim_strips_whitespace():
    claim = Claim(text="  Students need 75% attendance.  ")
    assert claim.text == "Students need 75% attendance."


def test_claim_rejects_empty_text():
    with pytest.raises(ValidationError):
        Claim(text="   ")


def test_claim_keeps_malformed_citations():
    """A fabricated citation is a finding, not noise. If we dropped it here the
    verifier could never report that the model made it up."""
    claim = Claim(text="Some claim.", citation_ids=["totally/made/up", "  ", ""])
    assert claim.citation_ids == ["totally/made/up"]


def test_draft_answer_parses_model_json():
    """The exact shape we require back from the language model."""
    payload = {
        "claims": [
            {"text": "Students need 75% attendance.", "citation_ids": ["uni_a/attendance/4.2"]},
            {"text": "Appeals close after 14 days.", "citation_ids": ["uni_a/examination/7.1"]},
        ]
    }
    draft = DraftAnswer.model_validate(payload)
    assert len(draft.claims) == 2
    assert draft.claims[0].citation_ids == ["uni_a/attendance/4.2"]


def test_draft_answer_rejects_wrong_shape():
    """Prose instead of structured claims must fail loudly, not slip through."""
    with pytest.raises(ValidationError):
        DraftAnswer.model_validate({"claims": "Students need 75% attendance."})


def test_draft_answer_exposes_json_schema():
    """We show this schema to the model so it knows the required shape."""
    schema = DraftAnswer.json_schema_for_prompt()
    assert "claims" in schema["properties"]


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------


def _verdict(text: str, status: VerdictStatus, score: float) -> ClaimVerdict:
    return ClaimVerdict(
        claim=Claim(text=text, citation_ids=["uni_a/attendance/4.2"]),
        status=status,
        score=score,
    )


def test_answer_counts_removed_claims():
    answer = Answer(
        question="What is the attendance requirement?",
        claims_kept=[_verdict("75% required.", VerdictStatus.SUPPORTED, 0.9)],
        claims_removed=[_verdict("Fines apply.", VerdictStatus.NEUTRAL, 0.1)],
    )
    assert answer.hallucination_count == 1


def test_answer_lists_citations_without_duplicates():
    answer = Answer(
        question="q",
        claims_kept=[
            _verdict("a", VerdictStatus.SUPPORTED, 0.9),
            _verdict("b", VerdictStatus.SUPPORTED, 0.8),  # same citation
        ],
    )
    assert answer.all_citations() == ["uni_a/attendance/4.2"]


def test_answer_removed_claims_are_kept_not_deleted():
    """This is the transparency feature: the user gets to see what the system
    started to say and then decided it could not stand behind."""
    removed = _verdict("Something unsupported.", VerdictStatus.NEUTRAL, 0.1)
    answer = Answer(question="q", claims_removed=[removed])
    assert answer.claims_removed[0].claim.text == "Something unsupported."


def test_verdict_score_must_be_a_probability():
    with pytest.raises(ValidationError):
        _verdict("x", VerdictStatus.SUPPORTED, 1.5)


def test_timings_total_is_the_sum():
    t = Timings(retrieve_ms=10.0, generate_ms=200.0, verify_ms=50.0)
    assert t.total_ms == 260.0


def test_answer_serialises_to_json_and_back():
    """Every run gets written to a JSONL log, so this has to round-trip."""
    answer = Answer(
        question="What is the attendance requirement?",
        university_filter="uni_a",
        claims_kept=[_verdict("75% required.", VerdictStatus.SUPPORTED, 0.9)],
        abstained=False,
        timings=Timings(retrieve_ms=1.0, generate_ms=2.0, verify_ms=3.0),
        config_fingerprint={"llm_model": "qwen3:4b"},
    )
    restored = Answer.model_validate_json(answer.model_dump_json())
    assert restored.question == answer.question
    assert restored.claims_kept[0].status is VerdictStatus.SUPPORTED
    assert restored.timings.total_ms == 6.0


def test_policy_type_values_are_plain_strings():
    """These end up in Chroma metadata and in log files, so they need to be
    strings rather than Python enum objects."""
    assert PolicyType.ACADEMIC_INTEGRITY.value == "academic_integrity"
    assert PolicyType("attendance") is PolicyType.ATTENDANCE
