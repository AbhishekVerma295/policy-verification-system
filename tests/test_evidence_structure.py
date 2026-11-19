"""
Tests for structure-aware evidence units, the table-row conflict rule, the
section-heading premise, and clock times in the numeric guard.

These encode the causes found when answerable questions started being
declined after the ingestion repair (tables written one self-describing row
per line):

  - two rows of different tables were treated as two SOURCES disagreeing,
    so true scholarship claims were withheld as "conflicts";
  - lettered and numbered lists ran together into one unit the model could
    not read ("a) ... b) ...", "1. Offence 1. Consequence 2. Consequence");
  - a timetable under a lead-in line lost which time belonged to which meal;
  - a section whose items only make sense under their heading ("For all I
    Year Students:") could not support a claim about first-year students;
  - "7:30 pm" was read as the numbers 7 and 30.

No model is loaded: the NLI pipeline is a fake, as in test_verify.py.
"""

from __future__ import annotations

from policyverify.config import load_config
from policyverify.schema import Chunk, Claim, PolicyType, RetrievedChunk, VerdictStatus
from policyverify.verify.nli import (
    CONTRADICTION,
    ENTAILMENT,
    NLIChecker,
    NLIResult,
    _layout,
    combine_unit_results,
    evidence_units,
    is_table_row,
    split_sentences,
)
from policyverify.verify.numeric import check_numbers, extract_numbers
from policyverify.verify.verifier import premise_with_heading, verify_claim

ROW_A = "SAT SCORE | Category: 1501 above | Scholarship Name: Merit Scholarship 100% | Benefit: 100% Tuition Fee Waiver."
ROW_B = "SAT SCORE | Category: 1401 -1500 | Scholarship Name: Merit Scholarship 75% | Benefit: 75% Tuition Fee Waiver."


def _chunk(text: str, citation: str = "srm/residence/7.1-for_all_i_year_students", part: int = 0,
           heading: str = "For all I Year Students:") -> Chunk:
    return Chunk(
        chunk_id=f"{citation}#{part}",
        text=text,
        citation_id=citation,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.RESIDENCE,
        section_path=f"Hostel Rules > {heading}",
        source_url="https://example.srmist.edu.in/hostel",
    )


# ---------------------------------------------------------------------------
# table rows
# ---------------------------------------------------------------------------


def test_a_self_describing_table_row_is_recognised():
    assert is_table_row(ROW_A)
    assert not is_table_row("Students must maintain at least 75% attendance.")
    # A pipe alone is not enough: the row format has "Label: value" cells.
    assert not is_table_row("Breakfast | Lunch | Dinner")


def test_table_rows_are_units_of_their_own_and_never_windowed_together():
    """Two neighbouring rows describe two categories. Joined, they "support"
    a claim that mixes them (the right number from the wrong row)."""
    units = evidence_units(f"{ROW_A}\n{ROW_B}")
    assert units == [ROW_A, ROW_B]


def test_prose_around_a_table_is_windowed_on_each_side_only():
    text = f"First rule. Second rule.\n{ROW_A}\nThird rule. Fourth rule."
    units = evidence_units(text)
    assert "First rule. Second rule." in units
    assert "Third rule. Fourth rule." in units
    assert not any("Second rule." in u and "Third rule." in u for u in units)
    assert not any(ROW_A in u and u != ROW_A for u in units)


def test_combine_keeps_the_strongest_prose_contradiction_apart_from_rows():
    row = NLIResult(CONTRADICTION, 0.0, 0.01, 0.99, window=ROW_A)
    prose = NLIResult(CONTRADICTION, 0.0, 0.4, 0.6, window="A rule in prose.")
    support = NLIResult(ENTAILMENT, 0.9, 0.05, 0.05, window=ROW_B)
    combined = combine_unit_results([row, prose, support], threshold=0.5)
    assert combined.max_contradiction == 0.99  # still recorded in full
    assert combined.conflict_strength == 0.6  # but rows do not count here
    assert combined.conflict_text == "A rule in prose."


def test_result_without_the_prose_field_falls_back_to_the_full_contradiction():
    """Stubs and the old interface leave the new field unset."""
    result = NLIResult(CONTRADICTION, 0.0, 0.05, 0.95, window="x")
    assert result.conflict_strength == 0.95


class _ByPremise:
    def __init__(self, results):
        self.results = results

    def check(self, premise, hypothesis):
        return self.results[premise.split("\n", 1)[-1] if premise not in self.results else premise]


def _retrieved(citation: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(text, citation=citation, heading="SAT SCORE"), score=0.9, rank=0)


def test_a_row_of_another_table_is_not_a_conflicting_source():
    """The measured case: "the 75% merit scholarship is a 75% waiver" was
    withheld because the MAT table's 100% row "contradicted" it at 0.99."""
    sat, mat = "srm/scholarship/3-sat_score", "srm/scholarship/3-mat_score"
    mat_row = "MAT SCORE | Percentile: 98 | Benefit: 100% Tuition Fee Waiver."
    nli = _ByPremise({
        ROW_B: NLIResult(ENTAILMENT, 0.99, 0.0, 0.01, window=ROW_B, prose_contradiction=0.0),
        mat_row: NLIResult(CONTRADICTION, 0.0, 0.01, 0.99, window=mat_row, prose_contradiction=0.0),
    })
    claim = Claim(text="The Merit Scholarship 75% provides a 75% tuition fee waiver.", citation_ids=[sat, mat])
    verdict = verify_claim(claim, [_retrieved(sat, ROW_B), _retrieved(mat, mat_row)],
                           nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.checks.conflict is False


def test_a_prose_contradiction_from_another_section_is_still_a_conflict():
    one, two = "srm/attendance/7.3-minimum_attendance", "srm/attendance/7.5-condonation"
    nli = _ByPremise({
        "one": NLIResult(ENTAILMENT, 0.9, 0.05, 0.05, window="Students need 75%.", prose_contradiction=0.0),
        "two": NLIResult(CONTRADICTION, 0.0, 0.05, 0.95, window="No minimum applies.",
                         prose_contradiction=0.95, prose_contradiction_window="No minimum applies."),
    })
    claim = Claim(text="Students need 75% attendance.", citation_ids=[one, two])
    verdict = verify_claim(claim, [_retrieved(one, "one"), _retrieved(two, "two")], nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.checks.conflict is True
    assert verdict.checks.contradicting_text == "No minimum applies."


# ---------------------------------------------------------------------------
# lists
# ---------------------------------------------------------------------------


def test_lettered_items_after_a_full_stop_are_separate_sentences():
    text = "a) Full fee will be refunded before the program. b) If a student withdraws later, rent is charged."
    assert split_sentences(text) == [
        "a) Full fee will be refunded before the program.",
        "b) If a student withdraws later, rent is charged.",
    ]


def test_list_items_on_their_own_lines_are_separate_even_without_full_stops():
    text = "c) Withdrawal from these courses is not permitted\nd) These courses run for 7 to 8 weeks"
    assert split_sentences(text) == [
        "c) Withdrawal from these courses is not permitted",
        "d) These courses run for 7 to 8 weeks",
    ]


def test_numbered_markers_stay_with_their_item():
    """Code-of-conduct tables flatten to "1. Offence. 1. Consequence or / and
    2. Consequence": the numbers must not become units of their own, and a
    number left at the end of one item belongs to the next."""
    text = (
        "1. Misuse or Tampering of ID cards.\n1.\xa0 Intimation to parents\nor / and\n"
        "2. Five days academic suspension\n2. Any complaint of misconduct received."
    )
    assert split_sentences(text) == [
        "1. Misuse or Tampering of ID cards.",
        "1. Intimation to parents or / and",
        "2. Five days academic suspension",
        "2. Any complaint of misconduct received.",
    ]
    # ...and a window joins the offence to both of its consequences.
    assert (
        "1. Misuse or Tampering of ID cards. 1. Intimation to parents or / and "
        "2. Five days academic suspension"
    ) in evidence_units(text)


def test_a_number_after_a_capitalised_word_is_not_moved():
    """Only "<lowercase word> N." counts as a stray list number."""
    assert split_sentences("Read Rule 2. Then apply.") == ["Read Rule 2.", "Then apply."]


def test_timetable_items_are_offered_with_their_lead_in():
    text = (
        "12.12\nFood will be served at the following timings: -\nBreakfast\n07.00 am\n"
        "to 09.00 am\nLunch\n11.30 am\n01.30 pm\nDinner\n07.30 pm\n09.00 pm"
    )
    units = evidence_units(text)
    assert "12.12 Food will be served at the following timings: - Dinner 07.30 pm 09.00 pm" in units
    assert "12.12 Food will be served at the following timings: - Breakfast 07.00 am to 09.00 am" in units


def test_ordinary_prose_after_a_colon_is_not_cut_into_items():
    """PDF lines broken mid-sentence must never become "items": a fragment
    can drop the qualifier printed on the next line."""
    text = (
        "The following applies:\nStudents who are found guilty of ragging will be expelled from the hostel\n"
        "and may not be readmitted under any circumstances."
    )
    assert _layout(text).lead_ins == []


def test_windows_do_not_cross_a_numbered_subsection_heading():
    text = (
        "(iii) M.Tech Degree - 10 full semesters with maximum of 14 semesters\nR\n"
        "5.2 Temporary withdrawal from the programme:\nA student may withdraw for a semester.\n"
        "5.3 Exit option:\nStudents who join the B.Tech program can opt for a diploma."
    )
    units = evidence_units(text)
    assert not any("14 semesters" in u and "B.Tech" in u for u in units)
    assert not any("withdraw for a semester" in u and "diploma" in u for u in units)


def test_check_scores_lead_in_units_in_the_second_pass():
    class Fake:
        calls: list[list[str]] = []

        def __call__(self, inputs, batch_size=None):
            Fake.calls.append([p["text"] for p in inputs])
            out = []
            for p in inputs:
                t = p["text"]
                e = 0.9 if t.endswith("Dinner 07.30 pm 09.00 pm") and "timings" in t and "Breakfast" not in t else 0.05
                out.append([{"label": "entailment", "score": e}, {"label": "neutral", "score": 0.9 - e},
                            {"label": "contradiction", "score": 0.05}])
            return out

    checker = NLIChecker(load_config())
    checker._pipeline = Fake()
    text = "Food will be served at the following timings: -\nBreakfast\n07.00 am\nDinner\n07.30 pm\n09.00 pm"
    result = checker.check(text, "Dinner is served from 7:30 pm to 9:00 pm.")
    assert result.label == ENTAILMENT
    assert result.window.endswith("Dinner 07.30 pm 09.00 pm")
    assert len(Fake.calls) == 2  # sentences first, then windows and lead-ins


# ---------------------------------------------------------------------------
# section heading in front of the first chunk
# ---------------------------------------------------------------------------


def test_the_first_chunk_of_a_section_is_read_under_its_heading():
    chunk = _chunk("a) Full fee (except the cancellation fee of Rs.5000/-) will be refunded.")
    assert premise_with_heading(chunk).startswith("For all I Year Students:\na) Full fee")


def test_a_later_chunk_does_not_get_the_heading():
    """A later chunk can run on into the next subsection, where the heading
    is wrong (the summer-semester case in premise_with_heading)."""
    chunk = _chunk("3.7 Adding and dropping courses: ...", part=1, heading="Summer Semester:")
    assert premise_with_heading(chunk) == chunk.text


def test_heading_is_not_added_to_table_rows_or_repeated():
    assert premise_with_heading(_chunk(ROW_A, heading="SAT SCORE")) == ROW_A
    chunk = _chunk("Rules apply to all.", heading="Mess")
    assert premise_with_heading(chunk) == "Mess:\nRules apply to all."
    repeated = _chunk("Mess: rules apply to all.", heading="Mess")
    assert premise_with_heading(repeated) == repeated.text


def test_numeric_guard_reads_the_chunk_text_not_the_heading():
    chunk = _chunk("A fine of Rs 500 applies.", heading="Rule 12.10:")
    nli = _ByPremise({"A fine of Rs 500 applies.": NLIResult(ENTAILMENT, 0.9, 0.05, 0.05, window="x")})
    claim = Claim(text="The fine under rule 12.10 is Rs 500.", citation_ids=[chunk.citation_id])
    verdict = verify_claim(claim, [RetrievedChunk(chunk=chunk, score=0.9, rank=0)], nli=nli, config=load_config())
    assert verdict.checks.numeric_ok is False  # 12.10 is only in the heading


# ---------------------------------------------------------------------------
# clock times in the numeric guard
# ---------------------------------------------------------------------------

TIMETABLE = "Breakfast 07.00 am to 09.00 am Lunch 11.30 am 01.30 pm Dinner 07.30 pm 09.00 pm 12.10 Rs.500"


def test_a_time_is_one_number_not_two():
    numbers = extract_numbers("Dinner from 7:30 pm")
    assert len(numbers) == 1 and numbers[0].is_time and numbers[0].meridiem == "p"


def test_the_same_time_written_differently_matches():
    assert check_numbers("Dinner is served from 7:30 pm to 9:00 pm.", [TIMETABLE]).ok
    assert check_numbers("Dinner starts at 19:30.", [TIMETABLE]).ok


def test_a_different_time_does_not_match():
    assert not check_numbers("Dinner is served from 7:00 pm to 9:00 pm.", [TIMETABLE]).ok
    assert not check_numbers("Dinner is served from 8:30 pm.", [TIMETABLE]).ok


def test_am_and_pm_are_not_interchangeable():
    assert not check_numbers("Lunch ends at 1:30 am.", [TIMETABLE]).ok


def test_a_clause_number_is_not_read_as_a_time():
    numbers = extract_numbers("12.10 If a day scholar is found, a fine of Rs 2000 applies.")
    assert not any(n.is_time for n in numbers)
    assert check_numbers("Clause 12.10 sets the fine.", ["12.10 If a day scholar ..."]).ok


def test_a_colon_time_matches_a_timetable_printed_without_am_pm():
    assert check_numbers("Dinner starts at 7:30.", ["Dinner 07.30 to 09.00"]).ok
    assert not check_numbers("Attendance is 7:30.", ["Attendance 7.30%"]).ok
