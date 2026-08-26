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

Known limitation, found by running this against the real examination policy:
some SRM documents also use Roman numerals ("II. ADMISSION TO EXAMINATIONS")
as a third heading style, which neither pattern above catches. That content
does not get lost - it folds into whichever heading came before it - but it
means those particular sub-sections are citable only at a coarser grain than
their own heading. Not fixed here: a Roman-numeral pattern risks a much
worse problem, false positives on the word "I" alone.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date

from policyverify.schema import Document, PolicyType, Section, SourceFormat

# "4", "4.2", "4.2.1", "A.1" at the start of a line, followed by a space and
# more text on the same line - the numbering convention used throughout
# SRM's regulation and rules PDFs.
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+)*|[A-Z]\.\d+)[.):]?\s+(\S.*)$")

# A short, mostly-uppercase line - the convention SRM's HTML policy pages use
# for section headers within a single block of body text (see extract.py).
_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9 /&,\-()]{3,90}:?$")

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


def promote_inline_headings(text: str) -> str:
    """Put mid-line numbered headings onto their own line.

    A normalising pass, not a parsing one: it only moves line breaks so the
    heading detectors below can see headings that PDF extraction buried
    inside a line. No characters are added or removed, so no content can be
    lost by running it.
    """
    out = []
    for line in text.splitlines():
        match = _INLINE_HEADING_RE.search(line)
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
    words = [w for w in text.split() if w[:1].isalpha()]
    if not words:
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    return (capitalized / len(words)) >= min_ratio


def _is_numbered_heading(line: str) -> tuple[str, str] | None:
    """If `line` looks like '4.2 Minimum Attendance', return (number, rest)."""
    m = _NUMBERED_RE.match(line)
    if not m:
        return None
    number, rest = m.group(1), m.group(2).strip()
    # A real heading is short and title-cased. "4 students must attend..."
    # and "1. All examinations undertaken shall be cancelled," are ordinary
    # sentences that happen to start with a digit, not headings - reject them.
    if len(rest) > _MAX_HEADING_LEN or rest.endswith("."):
        return None
    if not _is_titlecase_majority(rest):
        return None
    return number, rest


def _is_caps_heading(line: str) -> bool:
    """True for short, mostly-uppercase lines like a caps section header."""
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
    lines = promote_inline_headings(text).splitlines()

    # (line index, section number if numbered, heading text)
    headings: list[tuple[int, str | None, str]] = []
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        numbered = _is_numbered_heading(line)
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
) -> Document:
    """Assemble a validated Document from already-extracted plain text."""
    checksum = hashlib.sha256(raw_bytes).hexdigest()
    sections = split_into_sections(text, title)
    return Document(
        doc_id=f"{university}/{policy_type.value}",
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
