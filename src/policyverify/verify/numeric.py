"""
numeric.py - do the numbers in the claim actually appear in the passage?

A deterministic guard, not a model. It answers one narrow question: does every
number the claim states also occur in the text that is supposed to support it?

WHY THIS EXISTS ALONGSIDE THE NLI MODEL
    Numbers are where a wrong answer does real damage here. "75% attendance"
    versus "80% attendance" is the difference between a student sitting an
    exam and being barred from it, and the two sentences are otherwise
    identical - which is exactly the kind of difference a similarity-based
    model can miss.

    In practice the NLI model handles these well (measured: 100% on numeric
    cases in the Phase 0 spike, and it returned contradiction at 1.00
    confidence for a planted "80%" claim against the real 75% rule). This
    guard is a cheap, independent second opinion that cannot be fooled by
    fluent phrasing, because it does not read the sentence at all - it just
    compares digits.

WHAT IT DELIBERATELY DOES NOT DO
    It does not understand numbers written as words ("two weeks" vs "14
    days"), unit conversion, or arithmetic. Those need real comprehension,
    which is the NLI model's job. This stays dumb on purpose: a guard that
    tries to be clever starts producing its own false alarms, and a false
    alarm here would suppress a correct claim.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Matches 75, 75.5, 1,200, 75%, the spaced form "75 %" that PDF extraction
# often produces, and a percentage written as a word: "75 percent",
# "75 per cent", "95 Percentage and above". Measured: without the words, a
# correct claim saying "75 percent" was vetoed against evidence saying "75%",
# and a correct "95%" was vetoed against the scholarship table's
# "95 Percentage and Above" - the same figure, written two ways.
_PERCENT_WORD = r"(?:per\s?cent|percent)(?:age)?\b"
_NUMBER_RE = re.compile(
    rf"(\d[\d,]*(?:\.\d+)?)\s*(%|{_PERCENT_WORD})?", re.IGNORECASE
)

# "24/7" and "24x7" mean round the clock - one figure, 24, not 24 and 7.
_ROUND_THE_CLOCK_RE = re.compile(r"(?<![\d/])24\s*[/x×]\s*7(?![\d/])", re.IGNORECASE)

# Numbers written as words, read ONLY when a unit follows ("three days",
# "ten percent", "twenty-five per cent"). Without them a claim saying "three
# days" was never checked against evidence saying "7 days". Without the unit
# requirement, "one of the following" would become the number 1 and fail
# every claim that used the phrase.
_WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100,
}
_UNITS = (
    r"days?|weeks?|months?|years?|hours?|minutes?|semesters?|times|credits?"
)
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(_WORD_VALUES) + r")(?:[\s-](one|two|three|four|five|six|seven|eight|nine))?"
    rf"\b(?=\s*(%|{_PERCENT_WORD}|(?:{_UNITS})\b))",
    re.IGNORECASE,
)

# A clock time: "7:30", "07.30 pm", "9:00 a.m.". With a colon it is always a
# time; with a dot only when am/pm follows, because "12.10" on its own is a
# clause number. Without this, "7:30 pm" was read as the numbers 7 and 30,
# and a correct dinner time was vetoed because "30" is not in "07.30 pm".
_TIME_RE = re.compile(
    r"(?<![\d.])(\d{1,2})(?::(\d{2})|\.(\d{2})(?=\s*[ap]\.?\s?m\b))"
    r"(?:\s*([ap])\.?\s?m\b\.?)?",
    re.IGNORECASE,
)

# Numbers this small and common are almost always list markers, clause
# numbering or ordinals ("(i) 1.", "within 2 of"), not facts being asserted.
# Flagging them produced noise without catching anything real.
_IGNORE_BARE_VALUES = {0.0, 1.0, 2.0}


@dataclass(frozen=True)
class Number:
    """One numeric value found in text.

    A clock time is kept as one value (`is_time`, `value` = hour * 60 +
    minute, `meridiem` "a"/"p" or None when not written), so "7:30 pm" is
    compared as a time rather than as the two numbers 7 and 30.
    """

    value: float
    is_percent: bool
    raw: str
    is_time: bool = False
    meridiem: str | None = None

    def _clock(self) -> tuple[int, int]:
        """(hour on a 12-hour clock, minute) of a time."""
        hour, minute = divmod(int(self.value), 60)
        return hour % 12, minute

    def matches(self, other: Number) -> bool:
        """Same quantity, allowing for formatting differences."""
        if self.is_time and other.is_time:
            # "7:30 pm", "07.30 pm" and "19:30" are the same time. A missing
            # am/pm on either side is not held against it; a different one is.
            if self.meridiem and other.meridiem and self.meridiem != other.meridiem:
                return False
            return self._clock() == other._clock()
        if self.is_time or other.is_time:
            # A time against a plain number: only "07.30" written without
            # am/pm, which is how some timetables print 7:30.
            time, plain = (self, other) if self.is_time else (other, self)
            if plain.is_percent:
                return False
            hour, minute = divmod(int(time.value), 60)
            return abs(plain.value - (hour + minute / 100)) < 1e-9
        if self.is_percent != other.is_percent:
            return False
        # Exact comparison. Policy numbers are thresholds, not measurements -
        # "at least 75%" and "at least 74%" are different rules, so there is
        # no tolerance to give.
        return abs(self.value - other.value) < 1e-9


@dataclass
class NumericCheck:
    """Result of comparing a claim's numbers against its evidence."""

    ok: bool
    detail: str
    claim_numbers: list[Number] = field(default_factory=list)
    unmatched: list[Number] = field(default_factory=list)


def extract_numbers(text: str) -> list[Number]:
    """Every numeric value in `text`, in order of appearance.

    Clock times are found first and taken out, so their parts are not also
    read as separate numbers.
    """
    found: list[tuple[int, Number]] = []
    taken: list[tuple[int, int]] = []
    for match in _TIME_RE.finditer(text):
        hour = int(match.group(1))
        minute = int(match.group(2) or match.group(3))
        if hour > 23 or minute > 59:
            continue
        meridiem = match.group(4).lower() if match.group(4) else None
        found.append(
            (
                match.start(),
                Number(
                    value=float(hour * 60 + minute),
                    is_percent=False,
                    raw=match.group(0).strip(),
                    is_time=True,
                    meridiem=meridiem,
                ),
            )
        )
        taken.append(match.span())
    for match in _ROUND_THE_CLOCK_RE.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        found.append((match.start(), Number(value=24.0, is_percent=False, raw=match.group(0))))
        taken.append(match.span())
    for match in _WORD_NUMBER_RE.finditer(text):
        tens, units, unit = match.group(1).lower(), match.group(2), match.group(3)
        value = _WORD_VALUES[tens] + (_WORD_VALUES[units.lower()] if units else 0)
        is_percent = unit == "%" or re.match(_PERCENT_WORD, unit, re.IGNORECASE) is not None
        found.append(
            (match.start(), Number(value=float(value), is_percent=is_percent, raw=match.group(0)))
        )
    for match in _NUMBER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in taken):
            continue
        digits, percent = match.group(1), match.group(2)
        try:
            value = float(digits.replace(",", ""))
        except ValueError:  # pragma: no cover - regex should prevent this
            continue
        found.append(
            (
                match.start(),
                Number(value=value, is_percent=bool(percent), raw=match.group(0).strip()),
            )
        )
    return [number for _, number in sorted(found, key=lambda pair: pair[0])]


def significant(numbers: list[Number]) -> list[Number]:
    """Drop values that are list markers rather than asserted facts."""
    return [
        n
        for n in numbers
        if n.is_percent or n.is_time or n.value not in _IGNORE_BARE_VALUES
    ]


def check_numbers(claim_text: str, premises: list[str]) -> NumericCheck:
    """Check that every number in the claim appears in at least one premise.

    Matching against the premises collectively, rather than one at a time, is
    deliberate: a claim may legitimately combine two cited passages, and a
    number present in either one is properly sourced.
    """
    claim_numbers = significant(extract_numbers(claim_text))

    if not claim_numbers:
        return NumericCheck(
            ok=True,
            detail="no numbers in claim",
            claim_numbers=[],
            unmatched=[],
        )

    premise_numbers = [n for p in premises for n in extract_numbers(p)]

    unmatched = [
        number
        for number in claim_numbers
        if not any(number.matches(candidate) for candidate in premise_numbers)
    ]

    if not unmatched:
        found = ", ".join(n.raw for n in claim_numbers)
        return NumericCheck(
            ok=True,
            detail=f"all numbers found in evidence ({found})",
            claim_numbers=claim_numbers,
            unmatched=[],
        )

    missing = ", ".join(n.raw for n in unmatched)
    available = ", ".join(sorted({n.raw for n in premise_numbers})) or "none"
    return NumericCheck(
        ok=False,
        detail=f"claim states {missing}, not found in evidence (evidence has: {available})",
        claim_numbers=claim_numbers,
        unmatched=unmatched,
    )
