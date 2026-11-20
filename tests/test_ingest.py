"""
Tests for Phase 1 ingestion: extract.py and normalize.py.

No network calls here - fetch.py (the one module that hits the internet) is
exercised for real by scripts/ingest.py, not by the test suite. These tests
use small synthetic HTML/text fixtures, several of them modelled directly on
real failures found by running the pipeline against SRM's actual documents
(see the comments in normalize.py and extract.py for what was found and why
each fix exists).
"""

from __future__ import annotations

import pymupdf

from policyverify.ingest.extract import (
    _page_furniture,
    _pdf_table_lines,
    _PdfTable,
    _prepare_pdf_tables,
    _strip_page_furniture,
    extract_html,
    extract_pdf,
    row_line,
)
from policyverify.ingest.normalize import (
    _is_numbered_heading,
    _is_roman_heading,
    build_document,
    join_lone_roman_numerals,
    join_lone_section_numbers,
    join_wrapped_caps_headings,
    promote_inline_headings,
    split_into_sections,
    split_regulation_headings,
)
from policyverify.schema import PolicyType, SourceFormat

# ---------------------------------------------------------------------------
# extract_html
# ---------------------------------------------------------------------------


def test_extract_html_strips_script_and_style():
    html = """
    <html><body>
      <script>trackVisit();</script>
      <style>.hidden { display: none; }</style>
      <main><p>Students must attend at least 75% of classes.</p></main>
    </body></html>
    """
    text = extract_html(html.encode())
    assert "trackVisit" not in text
    assert "display: none" not in text
    assert "75% of classes" in text


def test_extract_html_strips_nav_and_footer():
    html = """
    <html><body>
      <nav><a href="/a">Hostel Policy</a><a href="/b">Examination Policy</a></nav>
      <main><p>The real content lives here.</p></main>
      <footer>Copyright 2026</footer>
    </body></html>
    """
    text = extract_html(html.encode())
    assert "Hostel Policy" not in text
    assert "Copyright" not in text
    assert "The real content lives here." in text


def test_extract_html_strips_srm_icon_list_sidebar():
    """SRM's real policy pages carry a sidebar list of every other policy,
    styled as an Elementor 'icon list' widget rather than a semantic <nav> -
    found by inspecting the live Examination Policy page (2026-08-19). The
    'elementor-widget-icon-list' class lives on the outer wrapper div, with
    the actual <ul class="elementor-icon-list-items"> nested inside it -
    this fixture mirrors that real nesting, not a simplified guess."""
    html = """
    <html><body><main>
      <div class="elementor-element elementor-widget-icon-list">
        <div class="elementor-widget-container">
          <ul class="elementor-icon-list-items">
            <li><a href="/policies/hostel-policy/">Hostel Policy</a></li>
            <li><a href="/policies/scholarship-policy/">Scholarship Policy</a></li>
          </ul>
        </div>
      </div>
      <p>Students must attend at least 75% of classes.</p>
    </main></body></html>
    """
    text = extract_html(html.encode())
    assert "Hostel Policy" not in text
    assert "75% of classes" in text


def test_extract_html_strips_breadcrumb():
    html = """
    <html><body><main>
      <div class="breadcrumb-block">Home Policies Examination Policy</div>
      <p>The real content.</p>
    </main></body></html>
    """
    text = extract_html(html.encode())
    assert "Home Policies" not in text
    assert "The real content." in text


def test_extract_html_falls_back_to_body_when_no_main():
    html = "<html><body><p>No main tag here, just body text.</p></body></html>"
    text = extract_html(html.encode())
    assert "No main tag here" in text


# ---------------------------------------------------------------------------
# extract_html: footer call-to-action, superscripts, tables
# ---------------------------------------------------------------------------

# The shape of SRM's end-of-page block, copied from the real pages.
_CTA_SECTION = """
<section class="elementor-section elementor-top-section">
  <section class="elementor-section elementor-inner-section">
    <h2 class="elementor-heading-title">Where could your journey at SRMIST take you?</h2>
    <a href="/policies/hostel-policy/">Hostel Policy</a>
    <a href="/policies/">Policies</a>
    <a href="/policies/internal-complaints-committee/">Internal Complaints Committee</a>
  </section>
</section>
"""


def test_extract_html_strips_footer_call_to_action():
    """Real leak: the code of conduct's appeal process used to end with
    "... Hostel Policy / Policies / Internal Complaints Committee"."""
    html = f"""<html><body><main>
      <section><p>The decision of the Appellate Committee is final.</p></section>
      {_CTA_SECTION}
    </main></body></html>"""
    text = extract_html(html.encode())
    assert "The decision of the Appellate Committee is final." in text
    for leaked in ("Where could your journey", "Hostel Policy", "Policies", "Internal Complaints"):
        assert leaked not in text


def test_extract_html_keeps_a_cta_section_that_also_holds_real_text():
    """Never take real content with the footer: if the section holding the
    prompt contains anything except the prompt and links, leave it alone."""
    html = """<html><body><main><section>
      <h2>Where could your journey at SRMIST take you?</h2>
      <p>Hostel fees are payable before the start of each semester.</p>
    </section></main></body></html>"""
    text = extract_html(html.encode())
    assert "Hostel fees are payable before the start of each semester." in text


def test_extract_html_joins_superscripts_to_their_word():
    html = "<html><body><main><p>95<sup>th</sup> Percentile and above</p></main></body></html>"
    assert extract_html(html.encode()) == "95th Percentile and above"


# Modelled on SRM's scholarship page: a title in bold, then a table whose
# first column is a serial number, then the next table's title.
_SCORE_TABLES = """<html><body><main>
  <p><strong>b). Applicable for BBA (Only for KTR campus)</strong></p>
  <ol><li><strong>Based on Common University Entrance Test (CUET) scores</strong></li></ol>
  <table><tbody>
    <tr><td>S. No.</td><td>Percentile</td><td>Scholarship Name</td><td>Benefit</td></tr>
    <tr><td>1.</td><td>95 Percentile and Above</td><td>Merit Scholarship 100%</td><td>100% Tuition Fee Waiver</td></tr>
    <tr><td>2.</td><td>90 to 94.99 Percentile</td><td>Merit Scholarship 75%</td><td>75% Tuition Fee Waiver</td></tr>
  </tbody></table>
  <p><strong>2. CLAT SCORE</strong></p>
  <table><tbody>
    <tr><td>S. No.</td><td>Rank</td><td>Scholarship Name</td><td>Benefit</td></tr>
    <tr><td>1.</td><td>TOP 2500</td><td>Merit Scholarship 100%</td><td>100% Tuition Fee Waiver</td></tr>
  </tbody></table>
</main></body></html>"""


def test_extract_html_writes_each_table_row_as_one_self_contained_line():
    lines = extract_html(_SCORE_TABLES.encode()).splitlines()
    assert (
        "Applicable for BBA (Only for KTR campus) - Based on Common University Entrance "
        "Test (CUET) scores | Percentile: 90 to 94.99 Percentile | Scholarship Name: "
        "Merit Scholarship 75% | Benefit: 75% Tuition Fee Waiver."
    ) in lines
    # The serial-number column is dropped and the next table gets its own title,
    # not the previous table's.
    assert (
        "CLAT SCORE | Rank: TOP 2500 | Scholarship Name: Merit Scholarship 100% | "
        "Benefit: 100% Tuition Fee Waiver."
    ) in lines
    assert not any(line in {"1.", "2.", "S. No.", "Benefit"} for line in lines)


def test_table_rows_stay_in_the_section_that_introduces_them_and_are_never_headings():
    """The real failure: "95 Percentile and above" became heading number 95,
    leaving sections whose whole text was "Merit Scholarship 75% ..."."""
    text = "2. Faculty Of Management\nIntroduction to the scholarships.\n" + extract_html(
        _SCORE_TABLES.encode()
    )
    sections = split_into_sections(text, "Scholarship Policy")
    by_key = {s.section_key(): s for s in sections}
    # "2. CLAT SCORE" is a real heading and keeps its table rows.
    assert "2-clat_score" in by_key
    assert "TOP 2500" in by_key["2-clat_score"].text
    # The CUET rows stay under the heading above them; no band became a heading.
    assert "90 to 94.99 Percentile" in by_key["2-faculty_of_management"].text
    assert not any("percentile" in key for key in by_key)


def test_extract_html_spreads_a_merged_cell_over_every_row_it_covers():
    """The code of conduct's consequence cell spans several misconduct rows.
    rowspan says exactly which rows, so each row gets it (2026-11-20; these
    tables used to be left flat, with consequences between the misconducts)."""
    html = """<html><body><main><table>
      <tr><td colspan="2"><strong>LEVEL – 2</strong></td></tr>
      <tr><td><strong>MISCONDUCT</strong></td><td><strong>CONSEQUENCE</strong></td></tr>
      <tr><td>1. Misuse of ID cards.</td><td rowspan="2">Five days academic suspension</td></tr>
      <tr><td>2. Any complaint of misconduct.</td></tr>
    </table></main></body></html>"""
    assert extract_html(html.encode()).splitlines() == [
        "LEVEL – 2",
        "LEVEL – 2 | MISCONDUCT: 1. Misuse of ID cards. | CONSEQUENCE: Five days academic suspension.",
        "LEVEL – 2 | MISCONDUCT: 2. Any complaint of misconduct. | CONSEQUENCE: Five days academic suspension.",
    ]


def test_merged_table_title_row_becomes_a_section_heading():
    """"LEVEL – 1" ... "LEVEL – 6" each start a section, so a consequence is
    cited as code_of_conduct/level_2 - not "consequence-2"."""
    html = """<html><body><main><p>Intro text.</p><table>
      <tr><td colspan="2"><strong>LEVEL – 2</strong></td></tr>
      <tr><td><strong>MISCONDUCT</strong></td><td><strong>CONSEQUENCE</strong></td></tr>
      <tr><td>1. Misuse of ID cards.</td><td>Five days academic suspension</td></tr>
      <tr><td>2. Forgery.</td><td>Rustication</td></tr>
    </table></main></body></html>"""
    sections = split_into_sections("4. PUNISHMENTS\nSome text.\n" + extract_html(html.encode()), "Code")
    assert [s.section_key() for s in sections] == ["4-punishments", "level_2"]


def test_extract_html_multi_row_header_names_each_column_by_its_inner_levels():
    """The Employee Ward Scholarship header is three rows deep. Each value is
    named by the two innermost header levels, so "20 years above" is tied to
    "100%" - flat, the two were in separate columns of text."""
    html = """<html><body><main><table>
      <tr><td rowspan="3"><strong>Sub Category</strong></td>
          <td colspan="3"><strong>Prerequisites-Years of Service</strong></td></tr>
      <tr><td rowspan="2"><strong>Staff</strong></td>
          <td colspan="2"><strong>Concession Amount in Tuition Fee</strong></td></tr>
      <tr><td><strong>100%</strong></td><td><strong>75%</strong></td></tr>
      <tr><td>Availing EWS for the first time</td><td>Teaching staff</td>
          <td>20 years above</td><td>&gt; 15 &lt; 20 yrs</td></tr>
    </table></main></body></html>"""
    line = extract_html(html.encode()).splitlines()[-1]
    assert "Concession Amount in Tuition Fee - 100%: 20 years above" in line
    assert "Concession Amount in Tuition Fee - 75%: > 15 < 20 yrs" in line
    assert "Prerequisites-Years of Service - Staff: Teaching staff" in line


def test_extract_html_keeps_each_row_of_a_headerless_table_together():
    """A two-column list with no header row (the examination policy's robe
    colours) has no column names, but each row stays on one line so a colour
    is never separated from the person who wears it."""
    html = """<html><body><main><table>
      <tr><td>The Chancellor</td><td>Velvet (Blue with gold border)</td></tr>
      <tr><td>The Registrar</td><td>Velvet (Maroon)</td></tr>
    </table></main></body></html>"""
    assert extract_html(html.encode()).splitlines() == [
        "The Chancellor | Velvet (Blue with gold border).",
        "The Registrar | Velvet (Maroon).",
    ]


def test_extract_html_table_with_th_header_and_generic_label_above():
    """A <th> header row counts as a header even without a serial column, and
    a generic bold label ("Conditions:") is not used as the table's title."""
    html = """<html><body><main>
      <p><strong>6. Differently-Abled Scholarship</strong></p>
      <p><strong>Conditions:</strong></p>
      <p>Must apply every year within the stipulated date</p>
      <table>
        <tr><th>Sub Category</th><th>Conditions for renewal</th></tr>
        <tr><td>Visually impaired</td><td>Subject to maintaining a CGPA &gt; 7</td></tr>
      </table>
    </main></body></html>"""
    lines = extract_html(html.encode()).splitlines()
    assert (
        "Differently-Abled Scholarship | Sub Category: Visually impaired | "
        "Conditions for renewal: Subject to maintaining a CGPA > 7."
    ) in lines


# ---------------------------------------------------------------------------
# PDFs: page furniture and tables (rewritten 2026-11-20)
# ---------------------------------------------------------------------------


def _page(*lines: str, table: tuple[str, ...] = ()) -> list[tuple[str, bool]]:
    return [(ln, False) for ln in lines] + [(ln, True) for ln in table]


def test_page_furniture_is_a_line_at_a_page_edge_on_several_pages():
    """Modelled on a real failure: 'SRM INSTITUTE OF SCIENCE AND TECHNOLOGY'
    repeating once per page was being mistaken for a section heading."""
    pages = [
        _page("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", f"Real content on page {i}.", "more", "more", "end")
        for i in range(1, 4)
    ]
    furniture = _page_furniture(pages)
    cleaned = "\n".join(_strip_page_furniture(pages, furniture))
    assert "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY" not in cleaned
    assert "Real content on page 1." in cleaned
    assert "Real content on page 3." in cleaned


def test_repeated_lines_in_the_middle_of_pages_are_kept():
    """The real failure the old filter caused: any short line seen 3+ times
    was deleted, wherever it was - every grade point ("10", "9", ...),
    attendance codes "9" and "8", "First Class", every "%". Only lines at a
    page's top or bottom are candidates now."""
    pages = [
        _page("Header A", "Header B", "Header C", "First Class", "9", "%", "body", "body2", "Footer x", "Footer y", "Footer z")
        for _ in range(4)
    ]
    cleaned = _strip_page_furniture(pages, _page_furniture(pages))
    assert cleaned.count("First Class") == 4
    assert cleaned.count("9") == 4
    assert cleaned.count("%") == 4
    assert "Header A" not in cleaned


def test_page_number_is_removed_only_at_the_page_edge():
    pages = [_page("Some text.", "7", "More text.", "18")]
    assert _strip_page_furniture(pages, set()) == ["Some text.", "7", "More text."]


def test_table_rows_are_never_page_furniture():
    rows = ("Code: H.",)
    pages = [_page("a", "b", table=rows) for _ in range(4)]
    cleaned = _strip_page_furniture(pages, _page_furniture(pages))
    assert cleaned.count("Code: H.") == 4


def _table(rows, filled=None):
    return _PdfTable((0, 0, 1, 1), rows, filled or [[c or "" for c in r] for r in rows])


def test_pdf_table_rows_become_self_contained_lines():
    """The attendance-code table of R 7.2: each band now says its code."""
    t = _table([["Attendance rounded to", "Code"], ["95% and above", "H"], ["85 to 94%", "9"]])
    _prepare_pdf_tables([t])
    assert _pdf_table_lines(t, "") == [
        "Attendance rounded to: 95% and above | Code: H.",
        "Attendance rounded to: 85 to 94% | Code: 9.",
    ]


def test_pdf_table_continued_on_the_next_page_keeps_its_header():
    """The warden table runs onto the next page: no header there, and a
    fragment row holding only the '.in' a page break cut off an address."""
    first = _table(
        [
            ["S.No.", "Hostel Name", "Warden Mail ID", "CUG"],
            ["19", "Meenakshi", "warden.meenakshi.ktr@srmist.edu", "8056016617"],
        ]
    )
    second = _table([["", "", ".in", ""], ["20", "Kalpana", "warden.k@srmist.edu.in", "8056016618"]])
    _prepare_pdf_tables([first, second])
    assert _pdf_table_lines(first, "") == [
        "Hostel Name: Meenakshi | Warden Mail ID: warden.meenakshi.ktr@srmist.edu.in | CUG: 8056016617."
    ]
    assert _pdf_table_lines(second, "") == [
        "Hostel Name: Kalpana | Warden Mail ID: warden.k@srmist.edu.in | CUG: 8056016618."
    ]


def test_pdf_table_of_contents_is_dropped():
    rows = [["No", "Regulations", "Page No"]] + [[f"R {i}", f"Rule {i}", str(i)] for i in range(1, 8)]
    t = _table(rows)
    _prepare_pdf_tables([t])
    assert _pdf_table_lines(t, "") == []


def test_pdf_blank_template_is_written_as_its_column_names_once():
    """Bloom's-level assessment templates: '%' and '-' to be filled in. The
    only information is the weightage in the column names."""
    rows = [
        ["Level", "CLA-1 (45%)", "CLA-2 (15%)"],
        ["Level 1", "%", "-"],
        ["Level 2", "%", "-"],
    ]
    t = _table(rows)
    _prepare_pdf_tables([t])
    assert _pdf_table_lines(t, "") == ["Level | CLA-1 (45%) | CLA-2 (15%)."]


def test_pdf_numbered_categories_each_start_a_section():
    """Scholarship Categories PDF: "VIII | Employee Ward Scholarship | ..."
    rows get a "VIII. Employee Ward Scholarship" heading line, once."""
    rows = [
        ["Sl. No", "Category", "Tuition Fee"],
        ["VIII", "Employee Ward Scholarship", "100%"],
        ["VIII", "Employee Ward Scholarship", "75%"],
        ["IX", "Alumni Scholarship", "25%"],
    ]
    t = _table(rows)
    _prepare_pdf_tables([t])
    lines = _pdf_table_lines(t, "")
    assert lines == [
        "VIII. Employee Ward Scholarship",
        "Category: Employee Ward Scholarship | Tuition Fee: 100%.",
        "Category: Employee Ward Scholarship | Tuition Fee: 75%.",
        "IX. Alumni Scholarship",
        "Category: Alumni Scholarship | Tuition Fee: 25%.",
    ]
    keys = [s.section_key() for s in split_into_sections("\n".join(lines), "Categories")]
    assert keys == ["VIII-employee_ward_scholarship", "IX-alumni_scholarship"]


def test_row_line_writes_a_cell_spread_over_two_columns_once():
    assert row_line("", ["Category", "Category", "Fee"], ["Merit", "Merit", "100%"]) == (
        "Category: Merit | Fee: 100%."
    )


def test_extract_pdf_reads_a_real_ruled_table_row_by_row():
    """End to end on a PDF built here: a ruled table, read by pymupdf's
    table finder, comes out one row per line."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 60), "R 7.2 Attendance Percentage & Code:")
    xs, ys = [72, 250, 400], [80, 100, 120, 140]
    cells = [["Attendance rounded to", "Code"], ["95% and above", "H"], ["85 to 94%", "9"]]
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page.insert_text((xs[c] + 4, ys[r] + 14), text, fontsize=9)
    raw = doc.tobytes()
    doc.close()
    lines = extract_pdf(raw).splitlines()
    # The short line just above the table becomes its title.
    assert "R 7.2 Attendance Percentage & Code | Attendance rounded to: 85 to 94% | Code: 9." in lines


# ---------------------------------------------------------------------------
# split_into_sections
# ---------------------------------------------------------------------------


def test_promote_inline_headings_rescues_pdf_buried_headings():
    """The exact line from SRM's Academic Regulations PDF that motivated this:
    a stray table-cell "R" landed in front of the heading, and the body text
    stayed on the same line, so the heading was invisible to both detectors."""
    line = (
        "R 7.3 Minimum Attendance:  A student must strive to attend all the "
        "classes without fail."
    )
    out = promote_inline_headings(line).splitlines()
    assert "7.3 Minimum Attendance:" in out
    # nothing is discarded - the stray prefix and the body both survive
    assert out[0] == "R"
    assert out[-1].startswith("A student must strive")


def test_promote_inline_headings_leaves_line_start_headings_alone():
    text = "7.1 For all I Year Students:\nBody text here."
    assert promote_inline_headings(text) == text


def test_promote_inline_headings_ignores_undotted_numbers():
    """Strictness guard: "1 Hour of learning..." is a list item, not a
    heading. Requiring a dotted number keeps it from matching."""
    line = "1 Hour of learning from a Lecture Session per week (L)"
    assert promote_inline_headings(line) == line


def test_promote_inline_headings_loses_no_characters():
    """It only moves line breaks, so no content can go missing."""
    text = "prefix 2.4 Some Heading: and then the body continues here"
    before = "".join(text.split())
    after = "".join(promote_inline_headings(text).split())
    assert before == after


def test_inline_heading_promotion_produces_citable_sections():
    """End to end: the buried heading becomes a real, checkable citation
    instead of vanishing into a blob."""
    text = (
        "R 7.3 Minimum Attendance:  A student must maintain at least 75% attendance.\n"
        "R 7.4 Attendance Shortage:  The teacher shall prepare particulars.\n"
    )
    sections = split_into_sections(text, title="Academic Regulations")
    numbers = [s.number for s in sections]
    assert "7.3" in numbers
    minimum = next(s for s in sections if s.number == "7.3")
    assert minimum.section_key() == "7.3-minimum_attendance"


def test_split_detects_numbered_headings():
    text = (
        "4.1 Overview\n"
        "This section is an overview.\n"
        "4.2 Minimum Attendance\n"
        "Students must attend at least 75% of classes.\n"
    )
    sections = split_into_sections(text, title="Attendance Policy")
    numbers = [s.number for s in sections]
    assert "4.1" in numbers
    assert "4.2" in numbers
    minimum = next(s for s in sections if s.number == "4.2")
    assert "75% of classes" in minimum.text


def test_split_detects_caps_headings():
    text = (
        "REGISTRATION AND ENROLLMENT OF COURSES:\n"
        "Registrations are controlled by the Controller of Examinations.\n"
        "ADMISSION TO EXAMINATIONS:\n"
        "Registration for examination is mandatory.\n"
    )
    sections = split_into_sections(text, title="Examination Policy")
    headings = [s.heading for s in sections]
    assert "Registration And Enrollment Of Courses" in headings
    assert "Admission To Examinations" in headings


def test_split_rejects_enumerated_list_items_as_headings():
    """Real failure found running this against SRM's actual examination
    policy: '1. All examinations undertaken shall be cancelled, ...' was
    being mistaken for a numbered section heading. It is sentence-case
    prose, not Title Case, which is what should tell the two apart."""
    text = (
        "SOME REAL HEADING:\n"
        "Intro text before the list.\n"
        "1. All examinations undertaken shall be cancelled, per this rule.\n"
        "2. Another list item that is also just a sentence, not a heading.\n"
        "ANOTHER REAL HEADING:\n"
        "More content.\n"
    )
    sections = split_into_sections(text, title="Examination Policy")
    headings = [s.heading for s in sections]
    assert not any("All examinations undertaken" in h for h in headings)
    assert "Some Real Heading" in headings
    assert "Another Real Heading" in headings


def test_split_falls_back_to_single_section_with_no_headings():
    text = "Just a few plain sentences.\nNo headings anywhere in this text.\n"
    sections = split_into_sections(text, title="Untitled Policy")
    assert len(sections) == 1
    assert sections[0].heading == "Untitled Policy"
    assert sections[0].number is None
    assert "Just a few plain sentences." in sections[0].text


def test_split_empty_text_gives_no_sections():
    assert split_into_sections("", title="Empty Policy") == []
    assert split_into_sections("   \n  \n", title="Empty Policy") == []


def test_split_drops_headings_with_no_body_before_next_heading():
    """A heading immediately followed by another heading (nothing between
    them) should not produce an empty, useless section."""
    text = "FIRST HEADING:\nSECOND HEADING:\nReal content under the second one.\n"
    sections = split_into_sections(text, title="Policy")
    assert all(s.text.strip() for s in sections)


def test_split_gives_sections_valid_citation_ids():
    """Every produced section must actually satisfy CitationID - the whole
    point of normalize.py is to feed schema.py's contract, not just produce
    text that looks plausible."""
    text = (
        "4.2 Minimum Attendance\n"
        "Students must attend at least 75%% of classes.\n"
        "4.3 Exceptions\n"
        "Medical exceptions may apply.\n"
    )
    sections = split_into_sections(text, title="Attendance Policy")
    for section in sections:
        key = section.section_key()
        assert key  # non-empty
        from policyverify.schema import CitationID

        assert CitationID.is_valid(f"srm/attendance/{key}")


# ---------------------------------------------------------------------------
# build_document - the full assembly, schema-validated
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Roman-numeral headings and table-band guards
# ---------------------------------------------------------------------------


def test_roman_heading_detects_srm_examination_headings():
    assert _is_roman_heading("II. ADMISSION TO EXAMINATIONS") == ("II", "Admission To Examinations")
    # No space after the dot, as in the real page.
    assert _is_roman_heading("XII.INSTRUCTIONS TO THE CANDIDATES APPEARING FOR EXAMINATIONS")[0] == "XII"
    # A trailing full stop or colon is fine on an ALL CAPS heading.
    assert _is_roman_heading("XV. EVALUATION OF ANSWER BOOKLETS:") == ("XV", "Evaluation Of Answer Booklets")
    assert _is_roman_heading(
        "IV. ISSUE, COLLECTION OF EXAMINATION APPLICATION FORMS & PAYMENT OF EXAMINATION FEES."
    )[0] == "IV"
    assert _is_roman_heading("XXVIII. MISCELLANEOUS") == ("XXVIII", "Miscellaneous")
    # Title Case works too, and ")" as the delimiter.
    assert _is_roman_heading("III) Examination Time Table") == ("III", "Examination Time Table")


def test_roman_heading_allows_long_all_caps_headings():
    line = (
        "VII. COLLECTION OF INTERNAL ASSESSMENT MARKS FOR THE THEORY OR PRACTICAL/ PURELY "
        "INTERNAL COURSES AND EXTERNAL ASSESSMENT MARKS PRACTICAL / CLINICAL EXAMINATIONS"
    )
    assert len(line) > 100
    assert _is_roman_heading(line)[0] == "VII"


def test_roman_heading_never_matches_the_word_i():
    """The reason Roman numerals were not detected before: "I" is a word."""
    for line in [
        "I",
        "I.",
        "I am a student of this university.",
        "I. e. the student shall attend",
        "I.e. the student shall attend",
        "I declare the convocation open.",
        "I/we will not discuss or inform anything related to this pre-audit.",
        "I. The candidate shall be present at the hall.",
        "I. Am I late?",
    ]:
        assert _is_roman_heading(line) is None, line


def test_roman_heading_rejects_sentences_and_non_numerals():
    for line in [
        # Real line from SRM's attendance regulations: a sentence, not a heading.
        "VI.  The maximum number of such courses that a student will be permitted by the",
        "IIII. NOT A NUMERAL",
        "C. HOSTEL RULES",  # C, D, L, M are deliberately not accepted
        "D. Library Rules",
        "(ii). Disqualify him/her permanently or for a specified period.",
        "Velvet. GOLD BORDER",
    ]:
        assert _is_roman_heading(line) is None, line


def test_lone_roman_numeral_joins_the_caps_heading_below_it():
    text = "V.\nDETENTION/CANCELLATION OF CANDIDATURE\nIf a candidate does not earn ..."
    assert join_lone_roman_numerals(text).splitlines()[0] == (
        "V. DETENTION/CANCELLATION OF CANDIDATURE"
    )


def test_lone_numeral_is_not_joined_to_ordinary_text():
    for text in ["I\nAM HERE", "V.\nThe candidate shall sign.", "I.\nfollowed by prose"]:
        assert join_lone_roman_numerals(text) == text


def test_roman_headings_split_a_document_into_citable_sections():
    text = (
        "II. ADMISSION TO EXAMINATIONS\n"
        "Registration for examination is mandatory.\n"
        "I am told this line is ordinary text.\n"
        " V.\n"
        "DETENTION/CANCELLATION OF CANDIDATURE\n"
        "A candidate short of attendance may be detained.\n"
        "XX. GRIEVANCES\n"
        "Grievances go to the Controller of Examinations.\n"
    )
    sections = split_into_sections(text, "Examination Policy")
    assert [s.section_key() for s in sections] == [
        "II-admission_to_examinations",
        "V-detention_cancellation_of_candidature",
        "XX-grievances",
    ]
    assert "I am told this line is ordinary text." in sections[0].text


def test_numbered_heading_rejects_table_bands():
    """Real scholarship cells that used to become headings 95, 90, 85, 80."""
    for line in [
        "95 Percentile and above",
        "90 to 94.99 Percentile",
        "85 and 89.99 Percentile",
        "95 Percentage and Above",
        "95 th Percentile and above",
    ]:
        assert _is_numbered_heading(line) is None, line
    # Real numbered headings are unaffected.
    assert _is_numbered_heading("2. CLAT SCORE") == ("2", "CLAT SCORE")
    assert _is_numbered_heading("4.2 Minimum Attendance") == ("4.2", "Minimum Attendance")


def test_table_row_lines_are_never_headings_or_promoted():
    row = "SAT SCORE | Category: 1401 -1500 | Scholarship Name: Merit Scholarship 75%"
    assert _is_numbered_heading("3. " + row) is None
    assert _is_roman_heading("II. " + row) is None
    banded = "CUET | Percentile: 90 to 94.99 Percentile | Benefit: 75% Tuition Fee Waiver."
    assert promote_inline_headings(banded) == banded


def test_tight_numbered_heading_without_a_space():
    """"8.Employee ward scholarship (EWS)": no space after the dot on SRM's
    scholarship page. EWS and Alumni used to be cited as part of section 7."""
    assert _is_numbered_heading("8.Employee ward scholarship (EWS)") == (
        "8",
        "Employee ward scholarship (EWS)",
    )
    assert _is_numbered_heading("9.Alumni Scholarship") == ("9", "Alumni Scholarship")
    # A decimal or a sentence is still not a heading.
    assert _is_numbered_heading("8.5 marks are awarded") is None
    assert _is_numbered_heading("2.All examinations undertaken by the student shall be cancelled and recorded.") is None


def test_regulation_headings_are_split_out_whatever_their_case():
    """12 regulations were missed: no colon, a long title, mostly lower case,
    or "R7.1" with no space."""
    text = "\n".join(
        [
            "R 2.2 Category of Courses",
            "R 3.1 Course Registration for a semester",
            "R7.1 Attendance Definition: It is the physical presence of a student.",
            "R 1.3  SRM Joint Engineering Entrance Examination - UG (SRMJEEE - UG):  SRMJEEE (UG) is a test.",
        ]
    )
    out = split_regulation_headings(text).splitlines()
    assert out == [
        "2.2 Category of Courses:",
        "3.1 Course Registration for a semester:",
        "7.1 Attendance Definition:",
        "It is the physical presence of a student.",
        "1.3 SRM Joint Engineering Entrance Examination - UG (SRMJEEE - UG):",
        "SRMJEEE (UG) is a test.",
    ]
    sections = split_into_sections(text + "\nBody.", "Regulations")
    # 2.2 and 3.1 have no body in this snippet (the next line is a heading).
    assert [s.number for s in sections] == ["7.1", "1.3"]


def test_lone_section_number_joins_its_title():
    """Hostel rules: "16.0" then "GENERAL (Discipline)" - all of General
    (Discipline) used to be folded into Ragging."""
    text = "15.0\nRAGGING\nRagging is an offence.\n16.0\nGENERAL (Discipline)\n16.1 Be polite."
    assert "16.0 GENERAL (Discipline)" in join_lone_section_numbers(text).splitlines()
    # "15.0" is left alone: RAGGING is found as a heading already.
    assert "15.0" in join_lone_section_numbers(text).splitlines()


def test_wrapped_caps_heading_is_joined():
    text = "3. SRMIST STUDENT CONDUCT EXPECTATIONS AS AN INDIVIDUAL\nOR AS A GROUP:\nDO'S"
    assert join_wrapped_caps_headings(text).splitlines()[0] == (
        "3. SRMIST STUDENT CONDUCT EXPECTATIONS AS AN INDIVIDUAL OR AS A GROUP:"
    )


def test_document_key_names_a_second_document_of_the_same_type():
    """The Scholarship Categories PDF sits beside the scholarship web page:
    its own key keeps its document ID and citations apart."""
    doc = build_document(
        university="uni_a",
        university_name="University A",
        policy_type=PolicyType.SCHOLARSHIP,
        title="Scholarship Categories",
        source_url="https://example.edu/cat.pdf",
        source_format=SourceFormat.PDF,
        retrieved_at=__import__("datetime").date(2026, 11, 20),
        raw_bytes=b"x",
        text="I. Founder's Scholarship\nTuition Fee: 100%.\nII. Merit Scholarship\nTuition Fee: 100%.",
        key="scholarship_categories",
    )
    assert doc.doc_id == "uni_a/scholarship_categories"
    assert doc.policy_type == PolicyType.SCHOLARSHIP
    assert str(doc.citations()[0]) == "uni_a/scholarship_categories/I-founder_s_scholarship"


def test_build_document_produces_a_valid_document():
    from datetime import date

    text = (
        "4.2 Minimum Attendance\n"
        "Students must attend at least 75% of classes.\n"
        "4.3 Exceptions\n"
        "Medical exceptions may apply.\n"
    )
    doc = build_document(
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        title="Academic Regulations 2021",
        source_url="https://example.srmist.edu.in/regs.pdf",
        source_format=SourceFormat.PDF,
        retrieved_at=date(2026, 8, 14),
        raw_bytes=b"fake pdf bytes",
        text=text,
    )
    assert doc.doc_id == "srm/attendance"
    assert len(doc.sections) == 2
    assert doc.checksum  # sha256 of raw_bytes, non-empty
    # round-trips through JSON, same as scripts/ingest.py does when saving
    restored = doc.model_validate_json(doc.model_dump_json())
    assert restored.sections[0].heading == "Minimum Attendance"
    assert restored.sections[0].number == "4.2"
