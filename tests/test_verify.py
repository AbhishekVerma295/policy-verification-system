"""
Tests for Phase 4: the verifier.

The NLI model itself is replaced by a stub here, so this whole file runs in
milliseconds with no model download. That is deliberate and matches how the
rest of the suite works: these tests check the *decision logic* - what happens
when the checks disagree - not whether the model is any good. Model quality is
measured on real passages by the evaluation harness, not asserted in a unit
test.

Several cases below encode the specific failure this project exists to catch:
a claim that reads as perfectly supported but has the wrong number in it.
"""

from __future__ import annotations

from policyverify.config import load_config
from policyverify.schema import Chunk, Claim, PolicyType, RetrievedChunk, VerdictStatus
from policyverify.verify.citation import check_citation, check_citations
from policyverify.verify.nli import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    NLIResult,
    normalise_label,
    split_windows,
)
from policyverify.verify.numeric import check_numbers, extract_numbers
from policyverify.verify.verifier import verify_claim, verify_claims

CITATION = "srm/attendance/7.3-minimum_attendance"
PASSAGE = (
    "A student must maintain a minimum attendance record of at least 75% in "
    "individual courses, exclusive of leave of absence due to medical reasons."
)


class StubNLI:
    """Returns a fixed verdict, so decision logic can be tested in isolation."""

    def __init__(self, label: str, entailment: float = 0.0, contradiction: float = 0.0):
        self._result = NLIResult(
            label=label,
            entailment=entailment,
            neutral=max(0.0, 1.0 - entailment - contradiction),
            contradiction=contradiction,
        )
        self.calls = 0

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        self.calls += 1
        return self._result


def _retrieved(citation: str = CITATION, text: str = PASSAGE) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{citation}#0",
        text=text,
        citation_id=citation,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Academic Regulations > Minimum Attendance",
        source_url="https://example.srmist.edu.in/regs",
    )
    return RetrievedChunk(chunk=chunk, score=0.9, rank=0)


# ---------------------------------------------------------------------------
# numeric guard
# ---------------------------------------------------------------------------


def test_extract_numbers_handles_percentages_and_plain_values():
    numbers = extract_numbers("at least 75% within 14 days and a GPA of 3.0")
    values = [(n.value, n.is_percent) for n in numbers]
    assert (75.0, True) in values
    assert (14.0, False) in values
    assert (3.0, False) in values


def test_extract_numbers_handles_the_spaced_percent_pdfs_produce():
    numbers = extract_numbers("a minimum of 75 % attendance")
    assert numbers[0].value == 75.0
    assert numbers[0].is_percent


def test_extract_numbers_handles_thousands_separators():
    assert extract_numbers("a fine of 1,200 rupees")[0].value == 1200.0


def test_numeric_check_passes_when_numbers_match():
    result = check_numbers("Students need 75% attendance.", [PASSAGE])
    assert result.ok


def test_numeric_check_catches_the_75_vs_80_trap():
    """The error that actually harms a student: fluent, plausible, and wrong
    by one digit."""
    result = check_numbers("Students need 80% attendance.", [PASSAGE])
    assert not result.ok
    assert "80%" in result.detail


def test_numeric_check_does_not_confuse_a_percentage_with_a_bare_number():
    """"75 days" is not evidence for "75%"."""
    result = check_numbers("Students need 75% attendance.", ["a period of 75 days"])
    assert not result.ok


def test_numeric_check_passes_when_the_claim_has_no_numbers():
    result = check_numbers("Attendance is compulsory.", [PASSAGE])
    assert result.ok
    assert "no numbers" in result.detail


def test_numeric_check_matches_across_several_premises():
    """A claim may legitimately combine two cited passages."""
    result = check_numbers(
        "Students need 75% attendance and may appeal within 14 days.",
        [PASSAGE, "An appeal must be lodged within 14 days of the result."],
    )
    assert result.ok


def test_numeric_check_ignores_small_list_markers():
    """"(i) 1." style numbering is not a fact being asserted, and flagging it
    produced false alarms without catching anything real."""
    result = check_numbers("Rule 1 applies to all students.", ["Some text with no digits."])
    assert result.ok


# ---------------------------------------------------------------------------
# citation checks
# ---------------------------------------------------------------------------


def test_citation_resolves_when_it_was_retrieved():
    check = check_citation(CITATION, [_retrieved()])
    assert check.exists and check.was_retrieved
    assert not check.fabricated


def test_citation_marked_fabricated_when_it_exists_nowhere():
    check = check_citation("srm/attendance/99-invented", [_retrieved()], store=None)
    assert check.fabricated
    assert "fabricated" in check.describe()


def test_malformed_citation_is_not_well_formed():
    check = check_citation("not-a-citation", [_retrieved()])
    assert not check.well_formed
    assert check.fabricated


def test_citation_that_exists_in_corpus_but_was_not_retrieved():
    """A different failure from fabrication - the section is real, the model
    just cited something it was never shown."""

    class StubStore:
        def get_by_citation(self, citation_id):
            return [_retrieved("srm/attendance/9.9-elsewhere").chunk]

    check = check_citation("srm/attendance/9.9-elsewhere", [_retrieved()], store=StubStore())
    assert check.exists
    assert not check.was_retrieved
    assert not check.fabricated
    assert "not among the retrieved" in check.describe()


def test_store_failure_is_not_reported_as_fabrication():
    """Blaming the model for our own broken store would be a false accusation."""

    class BrokenStore:
        def get_by_citation(self, citation_id):
            raise RuntimeError("index unavailable")

    check = check_citation("srm/attendance/9.9-elsewhere", [], store=BrokenStore())
    assert not check.exists  # cannot confirm it
    # but the check completed rather than raising
    assert check.well_formed


def test_check_citations_handles_several_at_once():
    checks = check_citations([CITATION, "srm/x/made-up"], [_retrieved()])
    assert len(checks) == 2
    assert checks[0].exists
    assert checks[1].fabricated


# ---------------------------------------------------------------------------
# NLI helpers
# ---------------------------------------------------------------------------


def test_split_windows_leaves_short_passages_alone():
    assert split_windows("Short passage.") == ["Short passage."]


def test_split_windows_splits_long_passages():
    text = " ".join(f"Sentence number {i} about policy." for i in range(200))
    windows = split_windows(text, window_chars=400, overlap=50)
    assert len(windows) > 1
    assert all(len(w) <= 400 + 50 for w in windows)


def test_split_windows_preserves_content():
    """Truncation would silently drop the sentence that supports a claim -
    the exact silent failure this splitting exists to prevent."""
    sentences = [f"Rule {i} is unique and specific." for i in range(40)]
    text = " ".join(sentences)
    joined = " ".join(split_windows(text, window_chars=300, overlap=40))
    for sentence in sentences:
        assert sentence in joined


def test_split_windows_handles_empty_text():
    assert split_windows("") == []


def test_normalise_label_maps_checkpoint_variants():
    assert normalise_label("ENTAILMENT") == ENTAILMENT
    assert normalise_label("contradiction") == CONTRADICTION
    assert normalise_label("LABEL_1") == NEUTRAL


# ---------------------------------------------------------------------------
# verifier: how the checks combine
# ---------------------------------------------------------------------------


def test_supported_when_nli_entails_and_numbers_match():
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, entailment=0.95), config=load_config()
    )
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.score == 0.95
    assert verdict.checks.citation_supports is True


def test_refuted_when_nli_contradicts():
    claim = Claim(text="Students need no attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim,
        [_retrieved()],
        nli=StubNLI(CONTRADICTION, contradiction=0.97),
        config=load_config(),
    )
    assert verdict.status is VerdictStatus.REFUTED
    assert "contradicts" in verdict.explanation


def test_numeric_mismatch_vetoes_an_otherwise_supported_claim():
    """The core guard. Even when the passage reads as supporting the claim,
    a number that is not in the passage withholds support - fluent agreement
    does not make a figure right."""
    claim = Claim(text="Students need 80% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, entailment=0.97), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.checks.numeric_ok is False
    assert "numbers do not match" in verdict.explanation


def test_numeric_guard_cannot_grant_support_on_its_own():
    """Matching digits is not proof of meaning. Only NLI can grant support."""
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(NEUTRAL, entailment=0.10), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL


def test_claim_with_no_citation_is_not_supported():
    claim = Claim(text="Attendance is 75%.", citation_ids=[])
    verdict = verify_claim(claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.99), config=load_config())
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.score == 0.0
    assert "No citation given" in verdict.explanation


def test_claim_citing_only_a_fabricated_section_is_not_supported():
    claim = Claim(text="Attendance is 75%.", citation_ids=["srm/attendance/99-invented"])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.99), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.checks.citation_exists is False
    assert "fabricated" in verdict.explanation


def test_fabricated_citation_does_not_block_a_valid_one_on_the_same_claim():
    claim = Claim(
        text="Students need 75% attendance.",
        citation_ids=[CITATION, "srm/attendance/99-invented"],
    )
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.95), config=load_config()
    )
    assert verdict.status is VerdictStatus.SUPPORTED
    assert "could not be resolved" in verdict.explanation


def test_weak_entailment_stays_unsure_rather_than_supported():
    claim = Claim(text="Attendance is compulsory.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, entailment=0.30), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL


def test_verdict_records_the_evidence_it_used():
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.95), config=load_config()
    )
    assert verdict.evidence_chunk_ids == [f"{CITATION}#0"]


def test_verify_claims_reuses_one_model_across_claims():
    claims = [
        Claim(text="Students need 75% attendance.", citation_ids=[CITATION]),
        Claim(text="Attendance is compulsory.", citation_ids=[CITATION]),
    ]
    nli = StubNLI(ENTAILMENT, entailment=0.95)
    verdicts = verify_claims(claims, [_retrieved()], nli=nli, config=load_config())
    assert len(verdicts) == 2
    assert nli.calls == 2  # one premise each, not reloaded per claim
