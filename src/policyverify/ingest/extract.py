"""
extract.py - turn downloaded bytes into plain text.

Two functions, one per format:
  extract_html(raw)  strips SRM's site chrome (mega-menus, the sidebar list
                      of every other policy, breadcrumbs) and returns the
                      real page text
  extract_pdf(raw)   pulls text page by page with pymupdf

What "site chrome" means here was found by hand, not guessed: real SRM
policy pages are built with the Elementor page builder, and their body
prose sits inside <main> but shares that container with mega-menus, a
sidebar list of every other policy, and a footer call-to-action. None of
that is wrapped in <nav> or <footer>, so tag-based stripping alone leaves
it behind - see the class-based strips below, which were identified by
downloading a real page (the Examination Policy) and inspecting its DOM.

Footer call-to-action. Every SRM policy page ends with a small block - the
prompt "Where could your journey at SRMIST take you?" followed by links to
two neighbouring policies and the policy index ("Hostel Policy",
"Policies", "Research Policy", ...). It sits inside <main>, so it used to
leak onto the end of the last section (e.g. the code of conduct's appeal
process ended with "... Internal Complaints Committee"). It is now removed
by finding that prompt and dropping the Elementor section that holds it -
but only after checking that the section contains nothing except the
prompt and links, so real text can never go with it. See
_strip_footer_cta().

Tables. SRM's HTML pages carry real <table> markup - the scholarship page
alone has 34 tables mapping exam percentiles, scores and ranks to tuition
waivers. Flattening them cell by cell (get_text with a newline between
every cell) destroyed them twice over: each row lost the column headers
and the exam it belongs to, and normalize.py then mistook the cell
"95 Percentile and above" for a numbered heading, cutting the scholarship
page into ~20 sections of 45 characters reading only "Merit Scholarship 75%
75% Tuition Fee Waiver 3." - true, cited, and impossible to check against
anything. Simple tables are now written out one row per line, each line
naming its table and its columns; see _render_tables().
"""

from __future__ import annotations

import re
from collections import Counter

import pymupdf
from bs4 import BeautifulSoup, Comment, Tag

# The separator between the cells of a table row written out as one line
# (see _render_tables). normalize.py imports it: a line containing it is a
# table row, and a table row is never a section heading. " | " was chosen
# because ordinary policy prose essentially never contains it - checked
# against the whole SRM corpus (2026-11-15): one PDF line, which is not a
# heading candidate in any case.
TABLE_CELL_SEPARATOR = " | "

# Elementor widget classes that are reliably NOT body prose on SRM's site,
# found by inspecting the real DOM of srmist.edu.in/policies/*  (2026-08-19):
#   icon-list  -> every menu, sidebar policy list and table-of-contents on
#                 the site uses this widget type; real paragraphs never do
#   tabs       -> the department/faculty picker inside the mega-menu
#   breadcrumb -> the "Home > Policies > X" trail at the top of every page
_SRM_CHROME_SELECTORS = [
    ".elementor-widget-icon-list",
    ".elementor-widget-tabs",
    ".breadcrumb-block",
]

# The prompt that opens SRM's end-of-page call-to-action block. Each was
# checked against the raw HTML of all three SRM HTML pages (2026-11-15): the
# Elementor section holding it contains only this prompt and three links to
# other pages. Only the prompt is listed, not the link texts after it: those
# name neighbouring policies and change from page to page, and a bare line
# such as "Policies" could in principle be real content elsewhere.
_FOOTER_CTA_PROMPTS = [
    "Where could your journey at SRMIST take you?",
]

_BLANK_RUN_RE = re.compile(r"\n{3,}")
_SPACE_RUN_RE = re.compile(r"\s+")

# A column header that only numbers the rows ("S. No.", "Sl. No", "Sl .No",
# "SL. No.", "S.No"), compared after dropping spaces and dots. Its cells
# ("1.", "2.") carry no meaning once each row is its own line, and leaving
# them in would start every row with a number.
_SERIAL_HEADERS = {"sno", "slno"}

# Bold lines that are form labels repeated above many tables ("Conditions:",
# "Note:") rather than a name for the table. Passed over when looking for a
# table's title, so a waiver row is not labelled "Conditions".
_GENERIC_LABELS = {"conditions", "note", "who can apply", "objective", "examples", "important note"}

# A table title is a short bold line, not a sentence.
_MAX_LABEL_CHARS = 120
# How many text blocks back from a table to look for its title.
_LABEL_LOOKBACK = 8

# A list enumerator at the start of a title ("2. CLAT SCORE", "a). Applicable
# for BBA", "2.CAT/XAT/GMAT SCORE"). Dropped from the label so a row line
# never starts with a number - see normalize.py, which would otherwise be
# tempted to read it as a numbered heading.
_LEADING_ENUM_RE = re.compile(r"^\s*(?:\(?(?:\d{1,2}|[A-Za-z])\s*[.)]+\s*)+")

_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCK_TAGS = ["p", "li", *sorted(_HEADING_TAGS)]


def extract_html(raw: bytes) -> str:
    """HTML bytes -> cleaned plain text, one line per visible text block."""
    soup = BeautifulSoup(raw, "lxml")

    # Structural chrome: never real content on any site.
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()

    # SRM-specific chrome that survives the structural strip above - see the
    # module docstring for how these were identified.
    for selector in _SRM_CHROME_SELECTORS:
        for el in soup.select(selector):
            el.decompose()
    _strip_footer_cta(soup)

    _join_superscripts(soup)
    _render_tables(soup)

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    return _clean_text(text)


def _strip_footer_cta(soup: BeautifulSoup) -> None:
    """Remove the end-of-page call-to-action block (see the module docstring).

    The block is found by its prompt, then the whole Elementor <section>
    around it is dropped - prompt and links together. A section is only
    dropped if every piece of text in it is either the prompt or the text
    of a link: if the page ever puts real content in the same section, the
    block is left alone rather than taking that content with it.
    """
    for prompt in _FOOTER_CTA_PROMPTS:
        for s in soup.find_all(string=lambda x, p=prompt: bool(x) and x.strip() == p):
            section = s.find_parent("section")
            if section is None:
                continue
            # Climb to the outermost section that still holds nothing else.
            while (outer := section.find_parent("section")) is not None and _only_cta(outer, prompt):
                section = outer
            if _only_cta(section, prompt):
                section.decompose()


def _only_cta(section: Tag, prompt: str) -> bool:
    """True if every visible string in `section` is the prompt or link text."""
    for s in section.find_all(string=True):
        if isinstance(s, Comment) or not s.strip():
            continue
        if s.strip() == prompt or s.find_parent("a") is not None:
            continue
        return False
    return True


def _join_superscripts(soup: BeautifulSoup) -> None:
    """Glue <sup>/<sub> text onto the word it belongs to.

    get_text(separator="\\n") puts a line break between every text node,
    so "95<sup>th</sup> Percentile" came out as three lines, "95", "th",
    " Percentile" - and "95" alone, or "90 th Percentile to 94.99 th", is
    what the heading detector then saw. Replacing each superscript by its
    plain text and merging neighbouring text nodes gives "95th Percentile".
    """
    for tag in soup.find_all(["sup", "sub"]):
        tag.replace_with(tag.get_text())
    soup.smooth()


def _cell_text(el: Tag) -> str:
    """All text in a cell or block on one line, whitespace collapsed."""
    return _SPACE_RUN_RE.sub(" ", el.get_text(" ")).strip()


def _is_serial_header(text: str) -> bool:
    return re.sub(r"[\s.]", "", text).lower() in _SERIAL_HEADERS


def _render_tables(soup: BeautifulSoup) -> None:
    """Write each simple table out as one self-contained line per row.

    A row of the CUET table becomes

        Applicable for BBA (Only for KTR campus) - Based on Common University
        Entrance Test (CUET) scores | Percentile: 90 to 94.99 Percentile |
        Scholarship Name: Merit Scholarship 75% | Benefit: 75% Tuition Fee
        Waiver.

    (one line in the output). Every row now says which table it belongs to
    and what each value means, so it can be retrieved, read and checked on
    its own - a passage saying only "Merit Scholarship 75%" cannot be
    checked against a question about CUET at all. Each row ends with a full
    stop so the verifier's sentence splitter treats it as one sentence.

    Only tables that are unambiguous to read row by row are rewritten:
    rectangular (no merged cells - no rowspan or colspan), at least two
    columns, and a first row that is clearly a header (all <th> cells, or a
    first column headed "S. No." or similar). Every scholarship score, rank
    and percentile table on SRM's pages has that shape. Tables with merged
    cells - the code of conduct's misconduct/consequence tables, where one
    consequence cell spans eight misconduct rows - are left exactly as they
    were: guessing which merged cell belongs to which row is how a
    consequence would end up attached to the wrong misconduct, and those
    tables currently read correctly as flattened text.

    The rows replace the table in place, so each table stays inside the
    section that introduces it.
    """
    for table in soup.find_all("table"):
        lines = _table_rows_as_lines(table)
        if not lines:
            continue
        block = soup.new_tag("div")
        for line in lines:
            p = soup.new_tag("p")
            p.string = line
            block.append(p)
        table.replace_with(block)


def _table_rows_as_lines(table: Tag) -> list[str]:
    """The row lines for one table, or [] to leave the table untouched."""
    if table.find("table") is not None:
        return []  # nested tables: not a shape we can read row by row
    rows = [tr.find_all(["td", "th"], recursive=False) for tr in table.find_all("tr")]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return []
    width = len(rows[0])
    if width < 2 or any(len(r) != width for r in rows):
        return []
    for row in rows:
        for cell in row:
            if int(cell.get("rowspan", 1) or 1) > 1 or int(cell.get("colspan", 1) or 1) > 1:
                return []

    header = [_cell_text(c) for c in rows[0]]
    all_th = all(c.name == "th" for c in rows[0])
    if not all(header) or not (all_th or _is_serial_header(header[0])):
        return []

    label = _table_label(table)
    lines = []
    for row in rows[1:]:
        pairs = [
            f"{h}: {v}"
            for h, v in zip(header, (_cell_text(c) for c in row), strict=True)
            if v and not _is_serial_header(h)
        ]
        if not pairs:
            continue  # an empty row, e.g. a blank line on a printed form
        line = TABLE_CELL_SEPARATOR.join(([label] if label else []) + pairs)
        if line[-1] not in ".!?":
            line += "."
        lines.append(line)
    return lines


def _table_label(table: Tag) -> str:
    """The table's title: the nearest bold line above it, plus its parent.

    SRM's pages have no <caption>; a table is introduced by one or two bold
    lines - "b). Applicable for BBA (Only for KTR campus)" then "Based on
    Common University Entrance Test (CUET) scores". The nearest bold line
    that is a title (short, not a sentence, not a generic "Conditions:"
    label) is taken, and if the block just above it is another such title,
    that one is prepended, so a row names both the programme and the exam.
    Looking stops at the previous table, so a title is never borrowed
    from someone else's table. Returns "" if nothing suitable is found -
    the row still carries its column headers.
    """
    blocks = _preceding_blocks(table, _LABEL_LOOKBACK)
    for i, (el, strings) in enumerate(blocks):
        title = _as_title(el, strings)
        if not title:
            continue
        if i + 1 < len(blocks):
            parent = _as_title(*blocks[i + 1])
            if parent:
                return f"{parent} - {title}"
        return title
    return ""


def _preceding_blocks(table: Tag, limit: int) -> list[tuple[Tag, list]]:
    """Up to `limit` text blocks before `table`, nearest first.

    Each item is (block element, its text nodes that come before the table,
    in document order). Built from text nodes rather than elements so a
    block that also contains the table (a list item with a table inside)
    contributes only the text above the table.
    """
    blocks: list[tuple[Tag, list]] = []
    for s in table.find_all_previous(string=True):
        if isinstance(s, Comment) or not s.strip():
            continue
        if s.find_parent("table") is not None:
            break  # reached the previous table
        block = s.find_parent(_BLOCK_TAGS) or s.parent
        if blocks and blocks[-1][0] is block:
            blocks[-1][1].insert(0, s)
            continue
        if len(blocks) == limit:
            break
        blocks.append((block, [s]))
    return blocks


def _as_title(block: Tag, strings: list) -> str:
    """The block's text as a table title, or "" if it is not one."""
    text = _SPACE_RUN_RE.sub(" ", " ".join(s.strip() for s in strings)).strip()
    if not text or len(text) > _MAX_LABEL_CHARS or text.endswith("."):
        return ""
    if text.rstrip(":?-").strip().lower() in _GENERIC_LABELS:
        return ""
    if block.name not in _HEADING_TAGS and not all(_is_bold(s, block) for s in strings):
        return ""
    return _LEADING_ENUM_RE.sub("", text).rstrip(":").strip()


def _is_bold(s, block: Tag) -> bool:
    """True if text node `s` sits inside <strong>/<b> (or a heading) within `block`."""
    for parent in s.parents:
        if parent.name in ("strong", "b") or parent.name in _HEADING_TAGS:
            return True
        if parent is block:
            return False
    return False


def extract_pdf(raw: bytes) -> str:
    """PDF bytes -> plain text, page by page, in order."""
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    text = _strip_repeated_lines("\n".join(pages))
    return _clean_text(text)


def _strip_repeated_lines(text: str, min_repeats: int = 3, max_len: int = 100) -> str:
    """Remove lines that repeat verbatim many times across the document.

    Found necessary by running this against a real SRM PDF: a running page
    header ("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY") was appearing once
    per page and being mistaken for a section heading downstream in
    normalize.py - it looks exactly like one, short and title-cased, on its
    own line. Real body content essentially never repeats verbatim three or
    more times across a whole document, so frequency is a cheap and reliable
    signal for "this is page furniture (header/footer/running title), not
    content" - specific to PDFs, since page.get_text() concatenates pages
    with no marker showing where one page's header repeats the last one's.
    """
    lines = text.splitlines()
    counts = Counter(ln.strip() for ln in lines if ln.strip())
    noisy = {line for line, n in counts.items() if n >= min_repeats and len(line) <= max_len}
    return "\n".join(ln for ln in lines if ln.strip() not in noisy)


def extract(raw: bytes, fmt: str) -> str:
    """Dispatch to the right extractor by format ('html' or 'pdf')."""
    if fmt == "pdf":
        return extract_pdf(raw)
    if fmt == "html":
        return extract_html(raw)
    raise ValueError(f"unknown format {fmt!r} - expected 'html' or 'pdf'")


def _clean_text(text: str) -> str:
    """Collapse repeated blank lines and trim trailing whitespace per line."""
    text = _BLANK_RUN_RE.sub("\n\n", text)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()
