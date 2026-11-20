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
    # A combined PDF header ("Details of Schemes - Sl. No") counts by its last part.
    last = text.rsplit(" - ", 1)[-1]
    return re.sub(r"[\s.]", "", last).lower() in _SERIAL_HEADERS


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
    merged = any(
        int(cell.get("rowspan", 1) or 1) > 1 or int(cell.get("colspan", 1) or 1) > 1
        for row in rows
        for cell in row
    )
    if merged:
        return _merged_table_lines(table)
    width = len(rows[0])
    if width < 2 or any(len(r) != width for r in rows):
        return []

    header = [_cell_text(c) for c in rows[0]]
    all_th = all(c.name == "th" for c in rows[0])
    if not all(header) or not (all_th or _is_serial_header(header[0])):
        return _headerless_lines([[_cell_text(c) for c in r] for r in rows])
    return _simple_table_lines(table, rows, header)


def _simple_table_lines(table: Tag, rows: list[list[Tag]], header: list[str]) -> list[str]:
    """Rows of a rectangular table with a clear header, one line each."""
    label = _table_label(table)
    lines = []
    for row in rows[1:]:
        line = row_line(label, header, [_cell_text(c) for c in row])
        if line:
            lines.append(line)
    return lines


def row_line(label: str, header: list[str], values: list[str]) -> str:
    """One table row as one self-contained line, or "" for an empty row.

    "Label | Header 1: value 1 | Header 2: value 2." Serial-number columns
    are dropped, empty cells are skipped, and a cell identical to the one
    before it, header and value (one merged cell spread over two columns),
    is written once. Shared by the HTML and PDF table readers.
    """
    pairs: list[str] = []
    previous: tuple[str, str] | None = None
    for h, v in zip(header, values, strict=True):
        if not v or _is_serial_header(h) or previous == (h, v):
            continue
        pairs.append(f"{h}: {v}" if h and h != v else v)
        previous = (h, v)
    if not pairs:
        return ""  # an empty row, e.g. a blank line on a printed form
    line = TABLE_CELL_SEPARATOR.join(([label] if label else []) + pairs)
    if line[-1] not in ".!?":
        line += "."
    return line


def _headerless_lines(rows: list[list[str]]) -> list[str]:
    """A table with no header row: each row's cells kept together on one line.

    There are no column names to attach ("The Chancellor | Velvet (Blue with
    gold border)."), but a value at least stays on the same line as the row
    it belongs to. A row with a single non-empty cell is written as plain
    text: with nothing beside it there is nothing to keep together.
    """
    lines = []
    for row in rows:
        cells = [c for c in row if c]
        deduped = [c for i, c in enumerate(cells) if i == 0 or c != cells[i - 1]]
        if not deduped:
            continue
        if len(deduped) == 1:
            lines.append(deduped[0])
            continue
        line = TABLE_CELL_SEPARATOR.join(deduped)
        lines.append(line if line[-1] in ".!?" else line + ".")
    return lines


# --- tables with merged cells ----------------------------------------------
#
# Ten scholarship tables, six of the code of conduct's seven and three of the
# examination policy's six have merged cells (rowspan/colspan). They used to
# be left flat, one cell per line, for fear of attaching a merged cell to the
# wrong row - and flat they could not be read at all: the Employee Ward
# Scholarship table became a column of "100% / 75% / 50% / 25%" followed by a
# column of "20 years above / > 15 < 20 yrs / ...", and the code of conduct's
# consequences sat between the misconduct items they punish.
#
# There is no guessing involved, though. rowspan="3" says exactly which rows a
# cell covers, so the table is laid out as a grid with every merged cell
# copied into each position it covers - the same grid a reader sees - and
# each row is then written out like any other. Tested against every merged
# table on the three SRM pages (2026-11-20).


class _GridCell:
    __slots__ = ("text", "bold", "origin")

    def __init__(self, text: str, bold: bool, origin: tuple[int, int]) -> None:
        self.text = text
        self.bold = bold
        self.origin = origin  # (row, col) of the <td> this position belongs to


def _table_grid(table: Tag) -> list[list[_GridCell]]:
    """The table as a rectangular grid, merged cells copied where they span."""
    grid: dict[tuple[int, int], _GridCell] = {}
    trs = [tr for tr in table.find_all("tr") if tr.find_all(["td", "th"], recursive=False)]
    for r, tr in enumerate(trs):
        c = 0
        for cell in tr.find_all(["td", "th"], recursive=False):
            while (r, c) in grid:
                c += 1
            rowspan = max(1, int(cell.get("rowspan", 1) or 1))
            colspan = max(1, int(cell.get("colspan", 1) or 1))
            text = _cell_text(cell)
            bold = cell.name == "th" or (bool(text) and _all_bold(cell))
            for i in range(rowspan):
                for j in range(colspan):
                    grid[(r + i, c + j)] = _GridCell(text, bold, (r, c))
            c += colspan
    if not grid:
        return []
    height = max(r for r, _ in grid) + 1
    width = max(c for _, c in grid) + 1
    empty = _GridCell("", False, (-1, -1))
    return [[grid.get((r, c), empty) for c in range(width)] for r in range(height)]


def _all_bold(cell: Tag) -> bool:
    strings = [s for s in cell.find_all(string=True) if s.strip()]
    return bool(strings) and all(_is_bold(s, cell) for s in strings)


def _is_full_width(row: list[_GridCell]) -> bool:
    """One cell spanning the whole row: a title ("LEVEL – 1") or a note."""
    return len({cell.origin for cell in row}) == 1 and bool(row[0].text)


def _is_header_row(row: list[_GridCell]) -> bool:
    """Column names, not data.

    Bold or <th> cells; or a serial-number first column ("Sl. No"); or short
    names with no digits and no brackets ("Sub Category | Prerequisites").
    The last rule is what keeps a header-less list such as the examination
    policy's robe colours ("The Chancellor | Velvet (Blue with gold
    border)") from being read as column names.
    """
    cells = [cell for cell in row if cell.text]
    if not cells:
        return False
    if all(cell.bold for cell in cells) or _is_serial_header(row[0].text):
        return True
    return all(
        len(cell.text) <= 50 and not re.search(r"[\d()]", cell.text) for cell in cells
    )


def _merged_table_lines(table: Tag) -> list[str]:
    grid = _table_grid(table)
    if len(grid) < 2 or len(grid[0]) < 2:
        return []

    # Leading full-width rows are the table's title ("LEVEL – 1"). The title
    # is written on its own line first, so the section heading detector can
    # use it, and also starts every row, so each row says which level it is.
    titles: list[str] = []
    body = grid
    while body and _is_full_width(body[0]):
        titles.append(body[0][0].text)
        body = body[1:]
    if not body:
        return []

    # Header rows: the first row if it looks like one, plus every following
    # row that a header cell reaches down into (the Employee Ward
    # Scholarship header is three rows deep: "Prerequisites" over "Staff" /
    # "Concession Amount in Tuition Fee" over "100% | 75% | 50% | 25%").
    first = len(titles)  # grid index of body[0]
    n_header = 0
    if _is_header_row(body[0]):
        header_rows = {first}
        n_header = 1
        while n_header < len(body) - 1 and any(
            cell.origin[0] in header_rows for cell in body[n_header]
        ):
            header_rows.add(first + n_header)
            n_header += 1

    lines = list(titles)
    if n_header == 0:
        lines += _headerless_lines([[cell.text for cell in row] for row in body])
        return lines

    width = len(body[0])
    header = []
    for c in range(width):
        parts: list[str] = []
        for row in body[:n_header]:
            text = row[c].text
            if text and text not in parts:
                parts.append(text)
        # The two innermost levels name a column well enough
        # ("Concession Amount in Tuition Fee - 100%"); the outermost level
        # ("Prerequisites-Years of Continuous Service in SRM") would repeat
        # on every cell of the row.
        header.append(" - ".join(parts[-2:]))

    label = " - ".join(titles) or _table_label(table)
    for row in body[n_header:]:
        if _is_full_width(row):
            lines.append(row[0].text)  # a note across the table ("NOTE: ...")
            continue
        line = row_line(label, header, [cell.text for cell in row])
        if line:
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


# --- PDF ----------------------------------------------------------------------
#
# Two things went wrong with PDFs until 2026-11-20, both found by comparing
# the stored text against the real documents:
#
# 1. Tables were read one cell per line, like everything else on the page,
#    so a row fell apart: the hostel rules' warden table became a list of
#    hostel names followed by a list of e-mail addresses with nothing to say
#    which belonged to which.
# 2. Worse, the old header/footer filter deleted ANY short line seen three
#    or more times in the document. In a table a cell is often its own line,
#    and cells repeat: it deleted every grade point from the grade table
#    (O -> 10, A+ -> 9, ...), codes 9 and 8 from the attendance-code table,
#    "First Class" / "Second Class" and the CGPA bands from the
#    class/distinction tables, and every "%" - 628 lines of the Academic
#    Regulations alone. An evaluation label had even been written from the
#    damaged text (num-011, "attendance code at 90%").
#
# Now each page's tables are found with pymupdf's table finder and written
# out one row per line (see _pdf_table_lines), the rest of the page is read
# as text blocks with the table areas cut out, and the header/footer filter
# only looks at the top and bottom lines of each page (_strip_page_furniture).

# Page furniture lives in the first and last few lines of a page.
_EDGE_LINES = 3
# A line is a running header/footer if it sits at a page edge on this many pages.
_FURNITURE_MIN_PAGES = 3
_FURNITURE_MAX_LEN = 100
# A table whose first rows contain one of these is a table of contents.
_TOC_HEADERS = {"page", "page no", "page no.", "pageno"}
# A cell that holds only a number, percentage, dash or range: data, not a column name.
_NUMERIC_CELL_RE = re.compile(r"^[\s\d.,%\-–/<>≥≤]*$")


class _PdfTable:
    """One table found on a PDF page, cleaned and ready to write out."""

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        rows: list[list[str | None]],
        filled: list[list[str]],
    ):
        self.bbox = bbox
        self.rows = rows  # None = a position covered by a merged cell
        self.filled = filled  # the same rows, each merged position holding its cell's text
        self.header: list[str] | None = None
        self.body: list[list[str]] = []
        self.toc = False
        self.label = ""
        self.n_header = 0
        self.last_group: tuple[str, str] | None = None
        # The table this one continues (same columns, no header of its own).
        self.continues: _PdfTable | None = None

    def group_seen(self) -> tuple[str, str] | None:
        """The last row group written, here or in the table this one continues."""
        if self.last_group is None and self.continues is not None:
            return self.continues.group_seen()
        return self.last_group


def extract_pdf(raw: bytes) -> str:
    """PDF bytes -> plain text, page by page, in order, tables row by row."""
    doc = pymupdf.open(stream=raw, filetype="pdf")
    try:
        page_tables = [_find_pdf_tables(page) for page in doc]
        _prepare_pdf_tables([t for tables in page_tables for t in tables])
        # Two passes: the first only finds the running headers and footers,
        # so the second never takes one as a table's title.
        first = [_page_lines(page, tables, set()) for page, tables in zip(doc, page_tables, strict=True)]
        furniture = _page_furniture(first)
        pages = [
            _page_lines(page, tables, furniture)
            for page, tables in zip(doc, page_tables, strict=True)
        ]
    finally:
        doc.close()
    return _clean_text("\n".join(_strip_page_furniture(pages, furniture)))


def _find_pdf_tables(page) -> list[_PdfTable]:
    """The real tables on one page.

    pymupdf also reports some boxed prose as a "table" (the Academic
    Regulations' R 4.3 is a paragraph inside a ruled box). A table with
    fewer than two columns that hold any text is left to the text reader,
    so a heading inside such a box is still seen as a heading.
    """
    found = []
    words = page.get_text("words")
    for table in page.find_tables().tables:
        rows = [[_pdf_cell(c) for c in row] for row in _cell_texts(table, words)]
        width = max((len(r) for r in rows), default=0)
        used = [c for c in range(width) if any(c < len(r) and r[c] for r in rows)]
        if len(rows) < 1 or len(used) < 2:
            continue
        rows = [r + [""] * (width - len(r)) for r in rows]
        found.append(_PdfTable(tuple(table.bbox), rows, _fill_by_geometry(table, rows)))
    return found


def _cell_texts(table, words) -> list[list[str | None]]:
    """Each cell's text, rebuilt from the page's words inside the cell's box.

    pymupdf's own table.extract() drops the spaces between words in these
    PDFs on some installations ("Attendanceroundedto: 95%andabove"), which
    broke the attendance code and grade tables for search and for the
    checks. The words are gathered by position instead: words on one line
    joined by spaces, lines by newlines (which _pdf_cell relies on to rejoin
    an e-mail address wrapped across lines). A merged position stays None,
    as table.extract() reports it.
    """
    out: list[list[str | None]] = []
    for row in table.rows:
        texts: list[str | None] = []
        for box in row.cells:
            if box is None:
                texts.append(None)
                continue
            x0, y0, x1, y1 = box
            inside = sorted(
                (w for w in words
                 if x0 <= (w[0] + w[2]) / 2 <= x1 and y0 <= (w[1] + w[3]) / 2 <= y1),
                key=lambda w: (w[5], w[6], w[7]),  # block, line, word
            )
            lines: dict[tuple[int, int], list[str]] = {}
            for w in inside:
                lines.setdefault((w[5], w[6]), []).append(w[4])
            texts.append("\n".join(" ".join(ws) for ws in lines.values()))
        out.append(texts)
    return out


def _fill_by_geometry(table, rows: list[list[str | None]]) -> list[list[str]]:
    """Each merged position filled with the text of the cell that covers it.

    pymupdf reports a merged position as None without saying which way the
    merge runs. In the Scholarship Categories table "Founder's Scholarship"
    runs DOWN over six rows, while "SRM Merit Scholarship" runs ACROSS two
    columns - copying from the row above would give the Merit row the
    Founder's sub-category letter. So the cell is found by position: the
    real cell whose box contains the centre of the merged position.
    """
    origins: list[tuple[tuple[float, float, float, float], str]] = []
    col_x: dict[int, tuple[float, float]] = {}
    for r, row in enumerate(table.rows):
        for c, box in enumerate(row.cells):
            if box is None or r >= len(rows) or c >= len(rows[r]):
                continue
            origins.append((box, rows[r][c] or ""))
            # A column's own extent: its narrowest real cell (a merged cell is wider).
            if c not in col_x or (box[2] - box[0]) < (col_x[c][1] - col_x[c][0]):
                col_x[c] = (box[0], box[2])
    filled: list[list[str]] = []
    for r, row in enumerate(rows):
        y0, y1 = table.rows[r].bbox[1], table.rows[r].bbox[3]
        cy = (y0 + y1) / 2
        out = []
        for c, text in enumerate(row):
            if text is not None or c not in col_x:
                out.append(text or "")
                continue
            cx = (col_x[c][0] + col_x[c][1]) / 2
            cover = [
                t
                for box, t in origins
                if box[0] - 1 <= cx <= box[2] + 1 and box[1] - 1 <= cy <= box[3] + 1
            ]
            out.append(cover[0] if cover else "")
        filled.append(out)
    return filled


def _pdf_cell(cell: str | None) -> str | None:
    """Join a cell's lines. E-mail addresses wrapped mid-address are rejoined
    without a space ("warden.paari.ktr@srmist.edu.i" + "n")."""
    if cell is None:
        return None
    if "@" in cell:
        return re.sub(r"\s*\n\s*", "", cell).strip()
    return _SPACE_RUN_RE.sub(" ", cell).strip()


def _prepare_pdf_tables(tables: list[_PdfTable]) -> None:
    """Find each table's header and fill its merged cells, across the document.

    Done over all tables at once because a table can run onto the next page:
    the warden table's last rows sit on the next page with no header, after
    a fragment row holding only the ".in" that the page break cut off an
    e-mail address.
    """
    previous: _PdfTable | None = None
    toc_width = 0  # column count of the table of contents, once seen
    for table in tables:
        rows = table.rows
        width = max(len(r) for r in rows)
        top = " ".join((c or "").lower() for row in rows[:2] for c in row)
        is_toc = any(re.search(rf"\b{re.escape(h)}\b", top) for h in _TOC_HEADERS) and len(rows) > 5
        # The contents run onto the next page without their header row.
        continues_toc = toc_width == width and len(rows) > 5
        if is_toc or continues_toc:
            table.toc = True  # a table of contents: page furniture, dropped
            toc_width = width
            continue
        toc_width = 0

        # A fragment row at the very top continues the previous table's last row.
        filled = table.filled
        while rows and previous is not None and previous.body and _is_fragment_row(rows[0]):
            _append_fragment(previous.body[-1], rows[0])
            rows, filled = rows[1:], filled[1:]

        n_header = _pdf_header_rows(rows)
        if n_header:
            table.header = _pdf_header_names(filled[:n_header])
        elif previous is not None and previous.header and len(previous.header) == width:
            table.header = previous.header  # the same table, continued
            table.continues = previous
        table.body = [list(r) for r in filled[n_header:]]
        table.n_header = n_header
        if table.body or table.header:
            previous = table


def _is_fragment_row(row: list[str | None]) -> bool:
    cells = [c for c in row if c]
    return len(cells) == 1 and len(cells[0]) <= 15 and (cells[0][0] == "." or cells[0][0].islower())


def _append_fragment(target: list[str], fragment: list[str | None]) -> None:
    for c, text in enumerate(fragment):
        if text and c < len(target):
            joiner = "" if text.startswith(".") or "@" in target[c] else " "
            target[c] = f"{target[c]}{joiner}{text}"


def _pdf_header_rows(rows: list[list[str | None]]) -> int:
    """How many leading rows are column names (0 if the table has none).

    The first row is a header if no cell is a bare number ("1", "95%") -
    that rules out the hostel department list, whose first row is "1 |
    Office | 8056016601". Further rows are header rows too while the row
    above them has merged positions (a grouped header like "Concession
    details" over "Tuition Fee | Hostel Fee") and they still hold no bare
    numbers.
    """
    if not rows or not _is_pdf_header_row(rows[0]):
        return 0
    n = 1
    while (
        n < len(rows) - 1
        and any(c is None for c in rows[n - 1])
        and not _starts_with_serial(rows[n - 1])
        and _is_pdf_header_row(rows[n])
    ):
        n += 1
    return n


def _starts_with_serial(row: list[str | None]) -> bool:
    """A header row that opens with "Sl. No" is the last header row: the
    Scholarship Categories' third page has rows "X | Performance Based" and
    "XI | Concessions" with no bare numbers, which would otherwise be read
    as more header rows."""
    first = next((c for c in row if c), "")
    return _is_serial_header(first)


def _is_pdf_header_row(row: list[str | None]) -> bool:
    cells = [c for c in row if c]
    return bool(cells) and not any(_NUMERIC_CELL_RE.match(c) for c in cells)


def _pdf_header_names(filled: list[list[str]]) -> list[str]:
    """Column names from one or more header rows (merged cells already filled)."""
    names = []
    for c in range(len(filled[0])):
        parts: list[str] = []
        for row in filled:
            if row[c] and row[c] not in parts:
                parts.append(row[c])
        names.append(" - ".join(parts[-2:]))
    return names


def _pdf_table_lines(table: _PdfTable, label: str) -> list[str]:
    if table.toc:
        return []
    if table.header is None:
        return _headerless_lines(table.body)
    if _is_blank_template(table):
        # The Academic Regulations print blank assessment templates: Bloom's
        # levels down the side, "%" and "-" to be filled in. Row by row they
        # say nothing; the only information is in the column names
        # ("Formative CLA-1 Average of unit test (45%)"), written once.
        names: list[str] = []
        for row in table.rows[: table.n_header]:
            for cell in row:
                if cell and cell not in names:
                    names.append(cell)
        return [TABLE_CELL_SEPARATOR.join(([label] if label else []) + names) + "."] if names else []
    group = _group_columns(table.header)
    lines = []
    for row in table.body:
        if group is not None:
            serial, name = row[group[0]].strip(), row[group[1]].strip()
            if serial and name and (serial, name) != table.group_seen():
                # "VIII. Employee Ward Scholarship": a heading for each group
                # of rows, so each scholarship becomes its own citable section.
                lines.append(f"{serial}. {name}")
                table.last_group = (serial, name)
        line = row_line(label, table.header, row)
        if line:
            lines.append(line)
    return lines


def _group_columns(header: list[str]) -> tuple[int, int] | None:
    """(serial column, category column) when a table lists numbered categories
    ("Sl. No | Category | Sub Category ..."), else None."""
    for i, name in enumerate(header[:-1]):
        if _is_serial_header(name) and "category" in header[i + 1].lower().rsplit(" - ", 1)[-1]:
            return i, i + 1
    return None


_PLACEHOLDER_CELLS = {"%", "-", "–", "—"}


def _is_blank_template(table: _PdfTable) -> bool:
    """Most body cells are "%" or "-": a form to fill in, not data."""
    cells = [c for row in table.body for c in row[1:] if c]
    return bool(cells) and sum(c in _PLACEHOLDER_CELLS for c in cells) * 2 >= len(cells)


def _pdf_table_label(out: list[tuple[str, bool]], furniture: set[str]) -> str:
    """A short title line directly above the table ("For Girls Hostels"), or "".

    Taken only if it is the very last text line before the table and reads
    like a title - short, and not the end of a sentence.
    """
    for line, is_table in reversed(out):
        if not line.strip() or line.strip() in furniture or re.fullmatch(r"\d{1,3}", line.strip()):
            continue
        text = line.strip()
        if is_table or len(text) > _MAX_PDF_LABEL or text[-1] in ".;,":
            return ""
        return text.rstrip(":").strip()
    return ""


# A table title above a PDF table is a short line.
_MAX_PDF_LABEL = 60


def _page_lines(page, tables: list[_PdfTable], furniture: set[str]) -> list[tuple[str, bool]]:
    """One page as (line, is_table_row) pairs, tables written where they sit.

    Text blocks are read top to bottom (pymupdf's sort=True). The PDF's own
    order is not always that: in the Academic Regulations it put the
    attendance-code table of R 7.2 inside R 6.2. A block whose centre lies
    inside a table is skipped, since the table is written out row by row
    instead, just before the first block below its top edge.
    """
    out: list[tuple[str, bool]] = []
    pending = sorted(tables, key=lambda t: t.bbox[1])

    def emit_tables_above(y: float) -> None:
        while pending and pending[0].bbox[1] <= y:
            table = pending.pop(0)
            # A table continued from the page before keeps that table's title.
            table.label = _pdf_table_label(out, furniture) or (
                table.continues.label if table.continues is not None else ""
            )
            out.extend((line, True) for line in _pdf_table_lines(table, table.label))

    for x0, y0, x1, y1, text, _bno, btype in page.get_text("blocks", sort=True):
        if btype != 0:
            continue  # an image
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(t.bbox[0] <= cx <= t.bbox[2] and t.bbox[1] <= cy <= t.bbox[3] for t in tables):
            continue
        emit_tables_above(y0)
        out.extend((line, False) for line in text.splitlines())
    emit_tables_above(float("inf"))
    return out


def _page_furniture(pages: list[list[tuple[str, bool]]]) -> set[str]:
    """Running headers and footers: lines at a page edge on several pages.

    A running header ("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY") used to be
    mistaken for a section heading downstream, so it has to go - but only
    lines at the top or bottom of a page are candidates now: a line is page
    furniture if it sits within the first or last _EDGE_LINES text lines of
    at least _FURNITURE_MIN_PAGES pages. Bare numbers are never on the list
    (page numbers are handled separately, by position).
    """
    def edges(lines: list[tuple[str, bool]]) -> list[str]:
        text = [ln.strip() for ln, is_table in lines if not is_table and ln.strip()]
        return text[:_EDGE_LINES] + text[-_EDGE_LINES:]

    pages_with = Counter(line for page in pages for line in set(edges(page)))
    return {
        line
        for line, n in pages_with.items()
        if n >= _FURNITURE_MIN_PAGES and len(line) <= _FURNITURE_MAX_LEN and not line.isdigit()
    }


def _strip_page_furniture(pages: list[list[tuple[str, bool]]], furniture: set[str]) -> list[str]:
    """Drop the furniture lines, and a bare number at the very top or bottom
    of a page (a page number). Table rows are never removed."""
    out: list[str] = []
    for page in pages:
        text_idx = [i for i, (ln, is_table) in enumerate(page) if not is_table and ln.strip()]
        kept = [i for i in text_idx if page[i][0].strip() not in furniture]
        page_numbers = {
            i for i in kept[:1] + kept[-1:] if re.fullmatch(r"\d{1,3}", page[i][0].strip())
        }
        for i, (line, is_table) in enumerate(page):
            if not is_table and (i in page_numbers or line.strip() in furniture):
                continue
            out.append(line)
    return out


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
