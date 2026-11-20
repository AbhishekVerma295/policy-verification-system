"""
table.py - checking a claim against ONE table row, without a model.

WHY THE NLI MODEL IS NOT ENOUGH FOR TABLES
    Ingestion stores every table one row per line ("Attendance rounded to:
    85 to 94% | Code: 9."), so a row can be read on its own. The NLI model
    still fails on three kinds of correct answer, all measured on 20 Nov 2026:

    - a value inside a range: "the code for 90% attendance is 9" - the row
      says "85 to 94%", and the model cannot see that 90 lies in it. The
      numeric guard also vetoes it, because "90" is not written anywhere;
    - a neighbouring row: "95% and above | Code: H" reads to the model as a
      contradiction of "the code is 9";
    - "Header: value" rows are not sentences: "Alumni Ward Scholarship ...
      Tuition Fee: 25%." scored 0.006 support for "the Alumni ward
      scholarship gives a 25% tuition waiver".

    The same weakness in the other direction produced the held-out false
    accepts of 16 Nov ("right number, wrong row").

WHAT THIS CHECK DOES
    It looks for the ONE table row, among the rows of the cited passages,
    that the claim is about, and grants support only when that row is
    unambiguous. A row matches when all of these hold:

    1. Every number in the claim is in that row - exactly ("25%" = "25%"),
       or inside a range the row states ("90%" in "85 to 94%", "8,500" in
       "7001 to 9000", "≥ 8.0", "Below 75%", "TOP 2500"). At least one number
       must be in the claim (without one there is nothing to look up, and
       the NLI model is the right judge), and at least one must be written
       in the row exactly: the answer, not just the value asked about.
    2. Most of the claim's words are in the row (`min_coverage`), so the row
       is about what the claim is about.
    3. No word of the claim belongs to a DIFFERENT row and not to this one.
       That is the "wrong row" test: "the code for 90% is H" fails, because
       H is in the "95% and above" row, not in the 85-94% row.
    4. Every name the claim uses (a word it capitalises mid-sentence) is in
       the row: "the Defence scholarship gives 25%" is not the Alumni row.
    5. Each number the claim states sits in a cell the claim talks about:
       "100% of exam fees" is not supported by a row whose 100% is under
       Tuition Fee while its Exam Fee is "-". Inside one cell, "State-25%
       National-50% International-75%", a number counts only next to a
       label the claim uses.

    Every rule here was added for a false support seen while building it,
    on claims written to be wrong (see tests/test_table.py).

    And then: EXACTLY ONE row may match. If two rows fit the claim equally
    well, the claim does not say which one it means, and support is not
    granted.

    Like the numeric guard, this check never refutes anything. It can only
    add support that the model missed; a claim it cannot place is left to
    the model. verifier.py also refuses its support when a cited PROSE
    passage strongly contradicts the claim (the same conflict rule as
    always), so a table can never outvote a rule written in words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from policyverify.verify.nli import is_table_row
from policyverify.verify.numeric import Number, extract_numbers, significant

_SEPARATOR = " | "

# Words that carry no meaning for matching a claim to a row.
_STOPWORDS = frozenset(
    """
    a an the and or of to in on for by with at from as is are was were be been
    being this that these those it its which who whom whose what when where how
    can may must shall should will would does do did has have had per any all
    each every than then there their they them such not no only also into under
    upto up about out over if so but his her he she you your our we us
    give gives given get gets provide provides provided offer offers offered
    carry carries receive receives received cover covers covered student
    students candidate candidates policy says state states according
    """.split()
)

# A number written with a unit or ordinal stuck to it: "95th", "1st", "10th".
_ORDINAL_RE = re.compile(r"(\d)(st|nd|rd|th)\b", re.IGNORECASE)
_NUM = r"(\d[\d,]*(?:\.\d+)?)"
# "85 to 94%", "7001 - 9000", "90th to 94.99th", "25-100%", "between 85 and 89.99"
_RANGE_RE = re.compile(
    rf"(?:between\s+)?{_NUM}\s*%?\s*(?:to|-|–|and)\s*{_NUM}\s*(%?)", re.IGNORECASE
)
# "95% and above", "≥ 8.0", "> 15", "above 90", "more than 10"
_LOWER_RE = re.compile(
    rf"(?:≥|>=|>|above|more than|over|minimum of|at least)\s*{_NUM}\s*(%?)|{_NUM}\s*(%?)\s*(?:and above|or above|and more|or more|\+)",
    re.IGNORECASE,
)
# "Below 75%", "< 20", "up to 25%", "TOP 2500", "less than 50"
_UPPER_RE = re.compile(
    rf"(?:≤|<=|<|below|less than|up to|upto|top)\s*{_NUM}\s*(%?)", re.IGNORECASE
)


@dataclass(frozen=True)
class _Range:
    low: float
    high: float
    is_percent: bool | None  # None: the row did not say

    def contains(self, n: Number) -> bool:
        if n.is_time:
            return False
        if self.is_percent is not None and n.is_percent and not self.is_percent:
            return False
        return self.low - 1e-9 <= n.value <= self.high + 1e-9


@dataclass(frozen=True)
class TableMatch:
    """The one row a claim was matched to, and how."""

    row: str
    coverage: float
    detail: str


def _num(text: str) -> float:
    return float(text.replace(",", ""))


def ranges_in(text: str) -> list[_Range]:
    """Every numeric range a row states (pure; used by tests too)."""
    text = _ORDINAL_RE.sub(r"\1", text)
    out: list[_Range] = []
    for m in _RANGE_RE.finditer(text):
        # "044-27434506" is a phone number, not a range.
        if m.group(1).startswith("0") and len(m.group(1)) > 1:
            continue
        low, high = _num(m.group(1)), _num(m.group(2))
        if low < high:
            out.append(_Range(low, high, True if m.group(3) else None))
    for m in _LOWER_RE.finditer(text):
        value, pct = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        out.append(_Range(_num(value), float("inf"), True if pct else None))
    for m in _UPPER_RE.finditer(text):
        out.append(_Range(float("-inf") if not m.group(0).lower().startswith("top") else 1.0,
                          _num(m.group(1)), True if m.group(2) else None))
    return out


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    return word[:5]


def words(text: str) -> set[str]:
    """Content words of a text, crudely stemmed ("fees" = "fee", "completed" = "completion").

    Single capital letters are kept as codes ("H", "L", "O", "A+"): they are
    what tells one row of an attendance-code or grade table from another.
    "A" and "I" are usually the article and the pronoun, so they count as
    codes only with a "+", after a colon or bracket ("Letter Grade: A"), or
    before "grade" ("an A grade").
    """
    out = {_stem(w) for w in re.findall(r"[a-z]{2,}", text.lower()) if w not in _STOPWORDS}
    out |= {c.lower() + "#" for c in re.findall(r"(?<![\w'’+])([B-HJ-Z]\+?)(?![\w'’+])", text)}
    out |= {c.lower() + "#" for c in re.findall(r"(?<![\w'’])([AI]\+)(?![\w'’+])", text)}
    out |= {
        c.lower() + "#"
        for c in re.findall(r"(?:(?<=: )|(?<=\())([AI])(?![\w'’+])|\b([AI])(?= grade\b)", text)
        for c in c
        if c
    }
    return out


def _numbers_in_row(claim_numbers: list[Number], row: str) -> bool:
    """Every claim number is in the row, and at least one EXACTLY.

    A range only places a number the claim was asked about ("90% attendance"
    in "85 to 94%"). The answer itself must be written in the row: otherwise
    "renewal needs a CGPA of 9" would be "in" the row's "CGPA > 8".
    """
    row_numbers = extract_numbers(_ORDINAL_RE.sub(r"\1", row))
    row_ranges = ranges_in(row)
    exact = [any(n.matches(r) for r in row_numbers) for n in claim_numbers]
    if not any(exact):
        return False
    return all(
        is_exact or any(rg.contains(n) for rg in row_ranges)
        for n, is_exact in zip(claim_numbers, exact, strict=True)
    )


def _names(claim_text: str) -> set[str]:
    """Words the claim capitalises mid-sentence: names of schemes, hostels,
    exams ("Defence", "Alumni", "SRMJEEE", "ESQ"). A row about something else
    cannot support a claim that names it - "the Defence scholarship gives
    25%" was matched to the Alumni row (25%, "scholarship", "tuition fee")
    before this rule."""
    tokens = re.findall(r"[A-Za-z][A-Za-z'’]*", claim_text)
    names = set()
    for i, tok in enumerate(tokens):
        if i == 0 or not tok[0].isupper() or tok.lower() in _STOPWORDS:
            continue
        stem = _stem(re.sub(r"['’]s?$", "", tok).lower())
        if len(stem) >= 2:
            names.add(stem)
    return names


def _answer_cell_fits(claim_numbers: list[Number], claim_words: set[str], row: str) -> bool:
    """The cell holding the answer is a column the claim talks about.

    "IIT JEE ... covers 100% of exam fees" found "100%" in the row - under
    Tuition Fee and Hostel Fee, while the row's Exam Fee is "-". So a cell
    that holds a claim number exactly must have a column name with at least
    one word the claim uses, not counting words every column of the row
    shares ("Fee", "Concession details"). A cell without a column name, or
    whose name has no word of its own, does not object.
    """
    cells = [c.split(": ", 1) for c in row.split(_SEPARATOR) if ": " in c]
    if not cells:
        return True
    headers = [words(h) for h, _ in cells]
    for number in claim_numbers:
        holding = [
            (i, value)
            for i, (_, value) in enumerate(cells)
            if any(number.matches(v) for v in extract_numbers(_ORDINAL_RE.sub(r"\1", value)))
        ]
        if not holding:
            continue  # in the row's title, or only inside a range
        fits = False
        for i, value in holding:
            if not _labelled_value_fits(value, [number], claim_words):
                continue
            # The cell's own words: those of its column name that no other
            # column shares ("Tuition", not "Fee" or "Concession details"),
            # plus the words of its value ("Benefit: 25% Tuition Fee Waiver"
            # is about a tuition waiver although its column is "Benefit").
            others = set().union(*(h for j, h in enumerate(headers) if j != i))
            own = (headers[i] - others) | words(value)
            if not own or own & claim_words:
                fits = True
                break
        if not fits:
            return False
    return True


# "State-25%", "National-50%": one label, one value, inside a single cell.
_LABELLED_VALUE_RE = re.compile(r"([A-Za-z][A-Za-z ]{1,30}?)\s*[-:–]\s*(\d[\d,]*(?:\.\d+)?\s*%?)")


def _labelled_value_fits(value: str, claim_numbers: list[Number], claim_words: set[str]) -> bool:
    """A cell holding several labelled values ("State-25% National-50%
    International-75%") supports a number only next to a label the claim
    uses: "75% for national-level achievements" is the International value,
    not the National one. A cell with fewer than two labelled values does
    not object."""
    pairs = _LABELLED_VALUE_RE.findall(value)
    if len(pairs) < 2:
        return True
    for label, number in pairs:
        parsed = extract_numbers(number)
        if any(n.matches(p) for n in claim_numbers for p in parsed) and words(label) & claim_words:
            return True
    return False


def table_rows(premises: list[str]) -> list[str]:
    """The distinct table rows in the cited passages, in order."""
    seen: dict[str, None] = {}
    for premise in premises:
        for line in premise.splitlines():
            line = line.strip()
            if _SEPARATOR in line and is_table_row(line):
                seen.setdefault(line, None)
    return list(seen)


def match_table_row(
    claim_text: str, premises: list[str], min_coverage: float = 0.6
) -> TableMatch | None:
    """The single table row that supports the claim, or None.

    See the module docstring for the three conditions and the uniqueness rule.
    """
    claim_numbers = significant(extract_numbers(claim_text))
    if not claim_numbers:
        return None
    rows = table_rows(premises)
    if not rows:
        return None
    claim_words = words(claim_text)
    if not claim_words:
        return None
    row_words = [words(r) for r in rows]
    names = _names(claim_text)

    matches: list[TableMatch] = []
    for i, row in enumerate(rows):
        if not _numbers_in_row(claim_numbers, row):
            continue
        mine = row_words[i]
        if names - mine:
            continue  # the claim names something this row is not about
        coverage = len(claim_words & mine) / len(claim_words)
        if coverage < min_coverage:
            continue
        if not _answer_cell_fits(claim_numbers, claim_words, row):
            continue
        others = set().union(*(w for j, w in enumerate(row_words) if j != i)) if len(rows) > 1 else set()
        wrong_row = sorted((claim_words & others) - mine)
        if wrong_row:
            continue
        matches.append(
            TableMatch(
                row=row,
                coverage=round(coverage, 2),
                detail=(
                    f"every number in the claim is in this table row "
                    f"({', '.join(n.raw for n in claim_numbers)}), and "
                    f"{coverage:.0%} of its words"
                ),
            )
        )
    return matches[0] if len(matches) == 1 else None
