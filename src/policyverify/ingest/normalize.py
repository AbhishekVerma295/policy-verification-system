"""
normalize.py - plain text -> a Document with a section tree.

Real institutional documents rarely mark their structure with semantic HTML
headings or any machine-readable markers at all (confirmed by hand against
SRM's actual pages and PDFs - see extract.py). What they do have, reliably,
is one of two visual conventions for a section heading:

  numbered   "4.2 Minimum Attendance ..."       -> becomes the citation number
  ALL CAPS   "REGISTRATION AND ENROLLMENT ..."  -> becomes a heading slug

split_into_sections() looks for lines matching either convention and treats
them as section boundaries. This is a FLAT split, not a tree: "4.2.1" is
still recognised and still becomes an exact citation, but no attempt is made
to nest it under "4.2" as a parent. Citation accuracy does not depend on
that nesting - only the section number string matters for a citation ID -
so this is a real simplification, not a correctness compromise. See
CitationID and Section.section_key() in schema.py.

If a document has fewer than 2 lines that look like headings, the whole
document becomes a single Section rather than failing. A document you can
still cite by title beats one you crash on.

A third convention, Roman numerals ("II. ADMISSION TO EXAMINATIONS"), is
used by SRM's examination policy. It used to go undetected - for fear of
matching the word "I" - and the cost turned out to be large: everything
from "V. DETENTION/CANCELLATION ..." to the end of Section A (hall tickets,
question papers, conduct of examinations, malpractice, results, grievances,
certificates) folded into ONE 42,000-character section, a fifth of the
whole corpus, which the retriever (two chunks per section at most) could
barely see into. Roman headings are now detected, strictly enough that "I"
starting a sentence never matches; see _is_roman_heading(). The numeral
becomes the section number, so "II. ADMISSION TO EXAMINATIONS" is cited as
"II-admission_to_examinations".

Table rows are never headings. extract.py writes each HTML table row as one
line with cells joined by TABLE_CELL_SEPARATOR, and such a line is skipped
by every detector here. Before that, the scholarship cell "95 Percentile and
above" was read as heading number 95, and the page fell apart into dozens
of 45-character sections; _is_numbered_heading() now also refuses a
"heading" that is really a numeric band ("90 to 94.99 Percentile").
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

from policyverify.ingest.extract import TABLE_CELL_SEPARATOR
from policyverify.schema import Document, PolicyType, Section, SourceFormat

# "4", "4.2", "4.2.1", "A.1" at the start of a line, followed by a space and
# more text on the same line - the numbering convention used throughout
# SRM's regulation and rules PDFs.
#
# Also "8.Employee ward scholarship (EWS)": SRM's scholarship page drops the
# space after the dot for five of its headings (5, 6, 8, 9 and a CUET
# heading), and each of those sections used to be swallowed by the one
# before it - the Employee Ward and Alumni scholarships were cited as part
# of "7. SRM Arts and Culture Scholarship". Without a space, the text must
# start with a capital letter, so "8.5 marks" or "2.3.1" never match.
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*|[A-Z]\.\d+)(?:[.):]?\s+|\.(?=[A-Z][a-z]))(\S.*)$")

# A regulation number at the start of a line in SRM's Academic Regulations:
# "R 2.2 Category of Courses", "R7.1 Attendance Definition: It is ...",
# "R 1.3  SRM Joint Engineering Entrance Examination - UG (SRMJEEE - UG):
# SRMJEEE (UG) is a ...". The "R" marks a regulation heading beyond doubt,
# but 12 of them were missed: some have no colon, some a long title, some
# are mostly lower case ("Course Registration for a semester"). See
# split_regulation_headings().
_REGULATION_RE = re.compile(r"^\s*R\s?(\d+(?:\.\d+)+)\s*(\S.*)$")

# A short, mostly-uppercase line - the convention SRM's HTML policy pages use
# for section headers within a single block of body text (see extract.py).
_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 /&,\-–()]{3,90}:?$")

# A numbered, Title Case heading ending in a colon, appearing anywhere in a
# line rather than at the start of one:
#
#     R 7.3 Minimum Attendance:  A student must maintain ...
#       ^^^^^^^^^^^^^^^^^^^^^^^  heading            ^^^^  body, same line
#
# PDF text extraction produces this constantly. Page furniture (here a stray
# table-cell "R") lands in front of the heading, and the body text that
# follows the heading stays on the same line, so neither of the patterns
# above sees it.
#
# This is not a rare edge case: matching it recovers 52 real section headings
# from SRM's Academic Regulations PDF alone. Without it that entire 57,000
# character document collapses into 7 sections, one of which is a 39,000
# character blob containing the 75% minimum-attendance rule under the
# meaningless heading "Yy Dd C L Ss A" - a citation no human could check.
#
# The pattern is deliberately strict to avoid false positives: it requires a
# dotted number (so "1 Hour of learning ..." does not match), a capitalised
# start, and a trailing colon.
_INLINE_HEADING_RE = re.compile(r"(?:^|\s)(\d+(?:\.\d+)+)\s+([A-Z][A-Za-z][^:\n]{2,58}):")

_MAX_HEADING_LEN = 100
# Text before the first heading becomes a "Preamble" section only when it is
# this long: a bare document title ("Examination Policy") is not content.
_MIN_PREAMBLE_WORDS = 25
# The longest title accepted in the no-space form "8.Employee ward
# scholarship (EWS)" without the Title Case check.
_MAX_TIGHT_HEADING = 60

# A Roman numeral from I to XXXIX, then "." or ")" and (optionally) the
# heading text. Only I, V and X are accepted: C, D, L and M are Roman too,
# but "C." and "D." start ordinary lettered lists far more often than they
# number a heading. SRM's examination policy runs from I to XXVIII.
_ROMAN_RE = re.compile(r"^(?=[IVX])(X{0,3}(?:IX|IV|V?I{0,3}))\s*[.)]\s*(.*)$")

# An ALL CAPS Roman heading may be long: SRM's examination policy has
# "VII. COLLECTION OF INTERNAL ASSESSMENT MARKS FOR THE THEORY OR PRACTICAL/
# PURELY INTERNAL COURSES AND EXTERNAL ASSESSMENT MARKS PRACTICAL / CLINICAL
# EXAMINATIONS" (147 characters). A numeral followed by that much capitals
# is still unmistakably a heading; prose is never written in capitals.
_MAX_ROMAN_CAPS_LEN = 180

# The first word of a "heading" that is really a numeric band from a table
# row: "95 Percentile and above", "90 to 94.99 Percentile", "85 and 89.99
# Percentile", "95 Percentage and Above". A real numbered heading names a
# topic; it does not start with a range word or a unit.
_BAND_WORDS = {"to", "and", "or", "percentile", "percentage", "th", "st", "nd", "rd", "-", "–"}


def split_regulation_headings(text: str) -> str:
    """Rewrite each "R n.n Title: body" line as "n.n Title:" plus the body.

    The colon-terminated form is what _is_numbered_heading() accepts with a
    relaxed Title Case check (see there), so every regulation heading is
    found however it is capitalised. Lines that are too long to hold a
    title before any colon are left alone. Only line breaks and the "R"
    marker change; the regulation text itself is untouched.
    """
    out = []
    for line in text.splitlines():
        m = None if TABLE_CELL_SEPARATOR in line else _REGULATION_RE.match(line)
        if not m:
            out.append(line)
            continue
        number, rest = m.group(1), m.group(2).strip()
        title, colon, body = rest.partition(":")
        title = title.strip()
        if not title or len(title) > _MAX_HEADING_LEN or not title[0].isupper():
            out.append(line)
            continue
        out.append(f"{number} {title}:")
        if colon and body.strip():
            out.append(body.strip())
    return "\n".join(out)


def join_lone_section_numbers(text: str) -> str:
    """Join "16.0" alone on its line to the title on the next line.

    The hostel rules print "16.0" and "GENERAL (Discipline)" as separate
    lines. Other sections there have ALL CAPS titles ("RAGGING") that are
    found as headings anyway, but this one is not all capitals, so all of
    General (Discipline) was folded into Ragging - one 8,948-character
    section. Joined only when the next line is a short title that is not
    already a heading by itself (so the citations of the sections that were
    already found do not change).
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (
            re.fullmatch(r"\d{1,2}\.0", line)
            and nxt
            and len(nxt) <= 60
            and nxt[0].isupper()
            and nxt[-1] not in ".,;"
            and not _is_caps_heading(nxt)
            and _is_titlecase_majority(nxt)
        ):
            out.append(f"{line} {nxt}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def join_wrapped_caps_headings(text: str) -> str:
    """Join an ALL CAPS numbered heading to its ALL CAPS second line.

    The code of conduct's "3. SRMIST STUDENT CONDUCT EXPECTATIONS AS AN
    INDIVIDUAL" wraps onto "OR AS A GROUP:". Read as two headings, the first
    had no text under it and was dropped, and the section was cited as
    "or_as_a_group".
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        numbered = _is_numbered_heading(line)
        if (
            numbered
            and not line.endswith(":")
            and _all_caps(numbered[1])
            and nxt
            and _is_caps_heading(nxt)
        ):
            out.append(f"{line} {nxt}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def _all_caps(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def promote_inline_headings(text: str) -> str:
    """Put mid-line numbered headings onto their own line.

    A normalising pass, not a parsing one: it only moves line breaks so the
    heading detectors below can see headings that PDF extraction buried
    inside a line. No characters are added or removed, so no content can be
    lost by running it.
    """
    out = []
    for line in text.splitlines():
        # A table row is never split: "90 to 94.99 Percentile | Scholarship
        # Name: ..." would otherwise look like heading 94.99.
        match = None if TABLE_CELL_SEPARATOR in line else _INLINE_HEADING_RE.search(line)
        # Only act when the heading is genuinely mid-line. A heading already
        # at the start of its line is fine as it is.
        if match and match.start(1) > 0:
            number, title = match.group(1), match.group(2).strip()
            before = line[: match.start(1)].strip()
            after = line[match.end() :].strip()
            if before:
                out.append(before)
            out.append(f"{number} {title}:")
            if after:
                out.append(after)
        else:
            out.append(line)
    return "\n".join(out)


def _is_titlecase_majority(text: str, min_ratio: float = 0.5) -> bool:
    """True if at least half the words start with a capital letter.

    Real headings are Title Case or ALL CAPS ("Minimum Attendance",
    "ADMISSION TO EXAMINATIONS"). Numbered *list items* inside a section
    ("1. All examinations undertaken shall be cancelled, ...") are ordinary
    sentence-case prose that just happens to start with a digit - this
    check is what tells the two apart. Found necessary by running this
    against SRM's real examination policy: without it, enumerated list
    items were being mistaken for section headings.
    """
    # Bracketed text is a gloss on the heading, not the heading, so it is left
    # out of the count. Before, half of it counted: "Secondary sources
    # (Inaccurate citation)" - the plagiarism policy's type 1 - scored 1 of 3
    # ("(Inaccurate" skipped, "citation)" counted) and was rejected, losing
    # that section. Counting ALL of it instead broke "Founder's Scholarship
    # (applicable for 4 years)", so the brackets are removed whole.
    words = [w for w in re.sub(r"\([^)]*\)?", " ", text).split() if w[:1].isalpha()]
    if not words:
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    return (capitalized / len(words)) >= min_ratio


def _is_numbered_heading(line: str) -> tuple[str, str] | None:
    """If `line` looks like '4.2 Minimum Attendance', return (number, rest)."""
    if TABLE_CELL_SEPARATOR in line:
        return None  # a table row (see extract.py), never a heading
    m = _NUMBERED_RE.match(line)
    if not m:
        return None
    number, rest = m.group(1), m.group(2).strip()
    # A real heading is short and title-cased. "4 students must attend..."
    # and "1. All examinations undertaken shall be cancelled," are ordinary
    # sentences that happen to start with a digit, not headings - reject them.
    if len(rest) > _MAX_HEADING_LEN or rest.endswith("."):
        return None
    # A dotted number and a title ending in a colon ("2.3 Structure of
    # credits and courses:") is a heading even when most of its words are
    # lower case; it only has to start with a capital. This is the shape
    # split_regulation_headings() writes.
    colon_heading = "." in number and rest.endswith(":") and rest[0].isupper()
    # "8.Employee ward scholarship (EWS)": the no-space form is a heading
    # style, not how a sentence in a list is written, so a short title is
    # enough there.
    tight_heading = bool(re.match(r"\d{1,2}\.[A-Z]", line)) and len(rest) <= _MAX_TIGHT_HEADING
    if not (colon_heading or tight_heading) and not _is_titlecase_majority(rest):
        return None
    # "95 Percentile and above" is a table cell, not heading 95.
    if rest.split()[0].lower() in _BAND_WORDS:
        return None
    return number, rest


def _is_roman_heading(line: str) -> tuple[str, str] | None:
    """If `line` looks like 'II. ADMISSION TO EXAMINATIONS', return (numeral, heading).

    Strict on purpose, because "I" is also an English word. All of these
    must hold:
      - a valid numeral I..XXXIX, immediately followed by "." or ")" -
        so "I am a student" and a line that is just "I" never match;
      - heading text after it that starts with a capital letter - so
        "I. e. the student" and "I.e." never match;
      - the text is ALL CAPS (up to _MAX_ROMAN_CAPS_LEN characters, a
        trailing "." or ":" allowed), or it is a short Title Case phrase
        that does not end like a sentence (".", "?").
    A sentence such as "VI. The maximum number of such courses ..." (found
    in SRM's attendance regulations) is mostly lower case and is rejected.
    """
    if TABLE_CELL_SEPARATOR in line:
        return None
    m = _ROMAN_RE.match(line)
    if not m:
        return None
    numeral, rest = m.group(1), m.group(2).strip()
    if not rest or not rest[0].isupper():
        return None
    letters = [c for c in rest if c.isalpha()]
    if len(letters) >= 4 and all(c.isupper() for c in letters):
        if len(rest) > _MAX_ROMAN_CAPS_LEN:
            return None
        return numeral, rest.rstrip(".:").strip().title()
    if len(rest) > _MAX_HEADING_LEN or rest[-1] in ".?!" or len(letters) < 4:
        return None
    if not _is_titlecase_majority(rest):
        return None
    return numeral, rest.rstrip(":").strip()


def join_lone_roman_numerals(text: str) -> str:
    """Join a numeral left alone on its line to the caps heading below it.

    The HTML of SRM's examination policy puts "V." and "DETENTION/
    CANCELLATION OF CANDIDATURE FOR APPEARING IN THE EXAMINATION" in
    separate elements, so they reach us as two lines. Only joined when the
    numeral has its "." or ")" and the next line is itself an ALL CAPS
    heading, so a lone "I" is never touched. Like promote_inline_headings()
    this only moves a line break; no characters are added or lost.
    """
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _ROMAN_RE.match(line.strip())
        if m and not m.group(2).strip() and i + 1 < len(lines) and _is_caps_heading(lines[i + 1].strip()):
            out.append(f"{line.strip()} {lines[i + 1].strip()}")
            i += 2
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _is_caps_heading(line: str) -> bool:
    """True for short, mostly-uppercase lines like a caps section header."""
    if TABLE_CELL_SEPARATOR in line:
        return False
    if len(line) > _MAX_HEADING_LEN or len(line) < 4:
        return False
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4 or not all(c.isupper() for c in letters):
        return False
    return bool(_CAPS_RE.match(line))


def split_into_sections(text: str, title: str) -> list[Section]:
    """Split cleaned document text into sections at detected heading lines."""
    # Rescue headings that PDF extraction buried mid-line before looking for
    # them - see promote_inline_headings().
    text = split_regulation_headings(text)
    text = join_lone_section_numbers(text)
    text = join_wrapped_caps_headings(text)
    lines = join_lone_roman_numerals(promote_inline_headings(text)).splitlines()

    # (line index, section number if numbered, heading text)
    headings: list[tuple[int, str | None, str]] = []
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        numbered = _is_numbered_heading(line) or _is_roman_heading(line)
        if numbered:
            number, heading = numbered
            headings.append((i, number, heading))
        elif _is_caps_heading(line):
            headings.append((i, None, line.rstrip(":").title()))

    # Too few candidates to trust as real structure - keep the whole
    # document as one section rather than mis-splitting it.
    if len(headings) < 2:
        body = "\n".join(lines).strip()
        if not body:
            return []
        return [Section(number=None, heading=title, text=body, path=[title], level=1)]

    sections: list[Section] = []
    # Text before the first heading. Usually only the document's title,
    # which is dropped; but the plagiarism policy opens with its Preamble
    # and the DEFINITION of plagiarism, and dropping that made "what is
    # plagiarism?" unanswerable. Kept as a section when it is real content.
    lead = "\n".join(lines[: headings[0][0]]).strip()
    if len(lead.split()) >= _MIN_PREAMBLE_WORDS:
        sections.append(
            Section(number=None, heading="Preamble", text=lead, path=[title, "Preamble"], level=2)
        )
    for idx, (line_no, number, heading) in enumerate(headings):
        start = line_no + 1
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue  # a heading with nothing under it before the next one
        sections.append(
            Section(number=number, heading=heading, text=body, path=[title, heading], level=2)
        )
    return sections


def build_document(
    university: str,
    university_name: str,
    policy_type: PolicyType,
    title: str,
    source_url: str,
    source_format: SourceFormat,
    retrieved_at: date,
    raw_bytes: bytes,
    text: str,
    key: str | None = None,
) -> Document:
    """Assemble a validated Document from already-extracted plain text.

    `key` names the document when it is not simply its policy type (see
    fetch.document_key); it becomes the middle part of the document ID and
    of every citation.
    """
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    sections = split_into_sections(text, title)
    return Document(
        doc_id=f"{university}/{key or policy_type.value}",
        university=university,
        university_name=university_name,
        policy_type=policy_type,
        title=title,
        source_url=source_url,
        source_format=source_format,
        retrieved_at=retrieved_at,
        checksum=checksum,
        sections=sections,
    )
