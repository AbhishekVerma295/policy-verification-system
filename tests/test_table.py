"""Tests for verify/table.py - the table-row check (added 2026-11-20).

Each true/false pair is modelled on a real row from SRM's documents. The
false claims are the "right number, wrong row" kind the NLI model let
through on the held-out set, plus the near misses found while building the
check (each rule in table.py exists because one of these got through).
"""

from __future__ import annotations

import pytest

from policyverify.verify.table import match_table_row, ranges_in, table_rows, words

CODES = """The percentage of attendance will be indicated by a code as follows:
Attendance rounded to: 95% and above | Code: H.
Attendance rounded to: 85 to 94% | Code: 9.
Attendance rounded to: 75 to 84% | Code: 8.
Attendance rounded to: Below 75% | Code: L."""

GRADES = """Letter Grade: O (Outstanding) | Grade Points: 10 | Mark Range: 91 - 100.
Letter Grade: A+ (Excellent) | Grade Points: 9 | Mark Range: 81 - 90.
Letter Grade: A (Very Good) | Grade Points: 8 | Mark Range: 71 - 80."""

CLAT = """CLAT SCORE | Rank: TOP 2500 | Scholarship Name: Merit Scholarship 100% | Benefit: 100% Tuition Fee Waiver.
CLAT SCORE | Rank: 2501 to 5000 | Scholarship Name: Merit Scholarship 75% | Benefit: 75% Tuition Fee Waiver.
CLAT SCORE | Rank: 5001 to 7000 | Scholarship Name: Merit Scholarship 50% | Benefit: 50% Tuition Fee Waiver.
CLAT SCORE | Rank: 7001 to 9000 | Scholarship Name: Merit Scholarship 25% | Benefit: 25% Tuition Fee Waiver."""

FOUNDERS = """I. Founder's Scholarship
Category: Founder's Scholarship | Sub Category: SRMJEEE | Section: SRMJEEE Rank 1-100 | Application Fee: - | Tuition Fee: 100% | Hostel or Transport Fee: 100% | Exam Fee: -.
Category: Founder's Scholarship | Sub Category: IIT JEE (Mains) | Section: IIT JEE (Mains) Rank up to 2000 | Application Fee: - | Tuition Fee: 100% | Hostel or Transport Fee: 100% | Exam Fee: -."""

ALUMNI = """Category: Alumni Scholarship | Sub Category: Alumni Ward Scholarship | Section: UG at least 15 years/PG at least 10 years | Tuition Fee: 25% | Exam Fee: -."""

ARTS = """Category: SRM Arts and Culture Scholarship | Sub Category: Fine Arts | Remarks: Seniors Only | Tuition Fee: 25-100%.
Category: SRM Arts and Culture Scholarship | Sub Category: Achievements, Accolades | Remarks: IInd Year onwards | Tuition Fee: State-25% National-50% International-75%."""

WARDENS = """For Girls Hostels | Hostel Name: ESQ-A | Warden Mail ID: warden.esqa.ktr@srmist.edu.in | Hostel CUG No.: 8056016636.
For Girls Hostels | Hostel Name: ESQ-B | Warden Mail ID: warden.esqb.ktr@srmist.edu.in | Hostel CUG No.: 8056016637."""


@pytest.mark.parametrize(
    "premise, claim",
    [
        (CODES, "The attendance code for 90% attendance is 9."),
        (GRADES, "An A+ grade carries 9 grade points."),
        (GRADES, "A mark of 85 earns an A+ grade with 9 grade points."),
        (CLAT, "A CLAT rank of 8,500 gets a 25% tuition fee waiver."),
        (CLAT, "A CLAT rank of 4,200 gets a 75% tuition fee waiver."),
        (FOUNDERS, "The Founder's Scholarship for SRMJEEE rank 1-100 covers 100% of hostel fees."),
        (ALUMNI, "The Alumni ward scholarship provides a 25% tuition fee waiver."),
        (ARTS, "The Arts and Culture scholarship gives a 50% tuition fee concession for national-level achievements."),
        (WARDENS, "The CUG number of ESQ-A hostel is 8056016636."),
    ],
)
def test_true_table_claims_are_matched_to_one_row(premise, claim):
    assert match_table_row(claim, [premise]) is not None


@pytest.mark.parametrize(
    "premise, claim, why",
    [
        (CODES, "The attendance code for 90% attendance is H.", "H belongs to the 95% row"),
        (CODES, "The attendance code for 90% attendance is 8.", "8 is the 75-84% row"),
        (CODES, "A student with 97% attendance gets code H.", "only a range matched, no exact value"),
        (GRADES, "An A grade carries 9 grade points.", "9 points is A+, not A"),
        (CLAT, "A CLAT rank of 8,500 gets a 50% tuition fee waiver.", "8,500 is in the 25% band"),
        (FOUNDERS, "The Founder's Scholarship for IIT JEE (Mains) rank up to 2000 covers 100% of exam fees.", "exam fee is '-'"),
        (ALUMNI, "The Defence scholarship gives a 25% tuition fee waiver.", "names a different scheme"),
        (ALUMNI, "The Alumni ward scholarship provides a 50% tuition fee waiver.", "wrong number"),
        (ARTS, "The Arts and Culture scholarship gives a 75% tuition fee concession for national-level achievements.", "75% is International"),
        (ARTS, "The Arts and Culture scholarship provides a 25-100% concession for national-level achievements.", "the Fine Arts row is not national"),
        (WARDENS, "The CUG number of ESQ-B hostel is 8056016636.", "that number is ESQ-A's"),
    ],
)
def test_false_table_claims_are_not_supported(premise, claim, why):
    assert match_table_row(claim, [premise]) is None, why


def test_a_claim_without_numbers_is_left_to_the_model():
    assert match_table_row("The warden of ESQ-A can be reached by e-mail.", [WARDENS]) is None


def test_prose_is_not_a_table_row():
    assert table_rows(["Students must attend 75% of classes.\nRanks: none."]) == []


def test_two_rows_that_fit_equally_well_give_no_support():
    """The claim does not say which of two identical-looking rows it means."""
    rows = "Category: M.Tech | CGPA: ≥ 8.0 | Class: First Class.\nCategory: B.Tech | CGPA: ≥ 8.0 | Class: First Class."
    assert match_table_row("A CGPA of 8.0 gives First Class.", [rows]) is None


def test_ranges_are_read_from_row_text():
    def bounds(text):
        return [(r.low, r.high) for r in ranges_in(text)]

    assert (85.0, 94.0) in bounds("85 to 94%")
    assert (7001.0, 9000.0) in bounds("Rank: 7001 to 9000")
    assert (95.0, float("inf")) in bounds("95% and above")
    assert (float("-inf"), 75.0) in bounds("Below 75%")
    assert (1.0, 2500.0) in bounds("TOP 2500")
    assert bounds("044-27434506") == []  # a phone number, not a range


def test_letter_codes_are_words_but_the_article_is_not():
    assert "h#" in words("is H.")
    assert "a+#" in words("Letter Grade: A+ (Excellent)")
    assert "a#" in words("Letter Grade: A (Very Good)")
    assert "a#" in words("An A grade")
    assert "a#" not in words("A student must attend")


# ---------------------------------------------------------------------------
# Inside the verifier: the check adds support, never removes it
# ---------------------------------------------------------------------------

from policyverify.config import load_config  # noqa: E402
from policyverify.schema import (  # noqa: E402
    Chunk,
    Claim,
    PolicyType,
    RetrievedChunk,
    VerdictStatus,
)
from policyverify.verify.nli import CONTRADICTION, NEUTRAL, NLIResult  # noqa: E402
from policyverify.verify.verifier import verify_claim  # noqa: E402

CODE_CITATION = "srm/attendance/7.2-attendance_percentage_code"


class _FixedNLI:
    """The model's verdict on every passage, fixed for the test."""

    def __init__(self, result: NLIResult):
        self.result = result

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        return self.result


def _codes_chunk() -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{CODE_CITATION}#0",
        text=CODES,
        citation_id=CODE_CITATION,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Academic Regulations > Attendance Percentage & Code",
        source_url="https://example.srmist.edu.in/regs",
    )
    return RetrievedChunk(chunk=chunk, score=0.9, rank=0)


def _verify(text: str, nli_result: NLIResult, table_check: bool = True):
    config = load_config()
    config.verification.table_row_check = table_check
    claim = Claim(text=text, citation_ids=[CODE_CITATION])
    return verify_claim(claim, [_codes_chunk()], nli=_FixedNLI(nli_result), config=config)


def _row_contradiction() -> NLIResult:
    """What the model really did: a neighbouring ROW read as a contradiction."""
    row = "Attendance rounded to: 95% and above | Code: H."
    return NLIResult(
        CONTRADICTION, 0.05, 0.08, 0.87, window=row,
        strongest_contradiction=0.87, contradiction_window=row,
        prose_contradiction=0.0, prose_contradiction_window="",
    )


def test_verifier_supports_a_range_lookup_the_model_rejected():
    verdict = _verify("The attendance code for 90% attendance is 9.", _row_contradiction())
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.checks.table_row
    assert verdict.checks.evidence_text == "Attendance rounded to: 85 to 94% | Code: 9."


def test_verifier_keeps_the_model_verdict_for_a_wrong_row():
    verdict = _verify("The attendance code for 90% attendance is H.", _row_contradiction())
    assert verdict.status is VerdictStatus.REFUTED
    assert not verdict.checks.table_row


def test_a_prose_contradiction_blocks_table_support():
    """A rule written in words is never outvoted by a table row."""
    prose = "Codes are no longer used from 2022 onwards."
    result = NLIResult(
        CONTRADICTION, 0.0, 0.05, 0.95, window=prose,
        strongest_contradiction=0.95, contradiction_window=prose,
        prose_contradiction=0.95, prose_contradiction_window=prose,
    )
    verdict = _verify("The attendance code for 90% attendance is 9.", result)
    assert verdict.status is VerdictStatus.REFUTED


def test_table_check_can_be_switched_off():
    neutral = NLIResult(NEUTRAL, 0.01, 0.98, 0.01)
    verdict = _verify("The attendance code for 90% attendance is 9.", neutral, table_check=False)
    assert verdict.status is VerdictStatus.NEUTRAL
