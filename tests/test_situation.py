"""Tests for applying a verified threshold to the asker's own figure."""

from __future__ import annotations

from policyverify.schema import Claim, ClaimVerdict, VerdictStatus
from policyverify.situation import apply_to_situation, user_percentage

RULE = "srm/attendance/7.3-minimum_attendance"


def _kept(*texts, citation=RULE):
    return [
        ClaimVerdict(claim=Claim(text=t, citation_ids=[citation]),
                     status=VerdictStatus.SUPPORTED, score=0.97)
        for t in texts
    ]


def test_the_users_question_74_out_of_100():
    """The real question that was declined: 74 of 100 classes, 75% rule."""
    note = apply_to_situation(
        "i have attended 74 classes out of 100, will i get detained?",
        _kept("A student must maintain a minimum attendance record of at least 75% "
              "in individual courses."),
    )
    assert note is not None
    assert note.user_value == 74 and note.threshold == 75 and not note.meets
    assert "74 classes out of 100 = 74%" in note.message
    assert "below the minimum of 75%" in note.message
    assert "short by 1 percentage point)" in note.message
    assert note.citation == RULE


def test_a_stated_percentage_that_meets_the_rule():
    note = apply_to_situation(
        "My attendance is 80 percent. Can I write the exam?",
        _kept("Students with less than 75% attendance will not be permitted to appear."),
    )
    assert note.meets and "meets the minimum of 75%" in note.message


def test_the_figure_forms_people_use():
    assert user_percentage("I attended 45 out of 60 lectures")[0] == 75
    assert user_percentage("I have 72% attendance")[0] == 72
    assert user_percentage("74/100 classes so far")[0] == 74
    assert user_percentage("Can parents stay overnight?") is None


def test_silent_without_a_verified_threshold_or_a_figure():
    assert apply_to_situation("I attended 74 of 100 classes.", []) is None
    assert apply_to_situation(
        "Will I be detained?", _kept("A student must maintain at least 75% attendance.")
    ) is None
    assert apply_to_situation(
        "I attended 74 of 100 classes.", _kept("Fees once paid will not be refunded.")
    ) is None


def test_silent_when_two_different_minimums_could_apply():
    """Not clear-cut, so no comparison: better nothing than the wrong one."""
    kept = _kept("At least 75% attendance is required in each course.",
                 "Scholarship renewal needs at least 60% marks.")
    assert apply_to_situation("I have 70% attendance.", kept) is None


def test_a_maximum_is_never_compared_with_the_askers_figure():
    """'Condonation cannot exceed 10%' limits the remedy, not attendance."""
    kept = _kept("At least 75% attendance is required.",
                 "Condonation of attendance cannot exceed 10%.")
    note = apply_to_situation("I have 70% attendance.", kept)
    assert note.threshold == 75 and not note.meets
