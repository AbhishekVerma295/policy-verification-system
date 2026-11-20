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

from policyverify.ingest.extract import (
    TABLE_CELL_SEPARATOR,
    _strip_repeated_lines,
    extract_html,
)
from policyverify.ingest.normalize import (
    _is_numbered_heading,
    _is_roman_heading,
    build_document,
    join_lone_roman_numerals,
    promote_inline_headings,
    split_into_sections,
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


def test_extract_html_leaves_tables_with_merged_cells_as_they_were():
    """The code of conduct's consequence cell spans several misconduct rows;
    guessing which row it belongs to is not safe, so the table is left flat."""
    html = """<html><body><main><table>
      <tr><td>MISCONDUCT</td><td>CONSEQUENCE</td></tr>
      <tr><td>1. Misuse of ID cards.</td><td rowspan="2">Five days academic suspension</td></tr>
      <tr><td>2. Any complaint of misconduct.</td></tr>
    </table></main></body></html>"""
    text = extract_html(html.encode())
    assert TABLE_CELL_SEPARATOR not in text
    assert text.splitlines() == [
        "MISCONDUCT",
        "CONSEQUENCE",
        "1. Misuse of ID cards.",
        "Five days academic suspension",
        "2. Any complaint of misconduct.",
    ]


def test_extract_html_leaves_tables_without_a_clear_header_as_they_were():
    """A two-column list with no header row (the examination policy's robe
    colours) has no column names to attach to each value."""
    html = """<html><body><main><table>
      <tr><td>The Chancellor</td><td>Velvet (Blue with gold border)</td></tr>
      <tr><td>The Registrar</td><td>Velvet (Maroon)</td></tr>
    </table></main></body></html>"""
    text = extract_html(html.encode())
    assert TABLE_CELL_SEPARATOR not in text
    assert "The Chancellor\nVelvet (Blue with gold border)" in text


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
# _strip_repeated_lines - the PDF running-header fix
# ---------------------------------------------------------------------------


def test_strip_repeated_lines_removes_running_header():
    """Modelled on a real failure: 'SRM INSTITUTE OF SCIENCE AND TECHNOLOGY'
    repeating once per page in a PDF was being mistaken for a section
    heading downstream. A line appearing 3+ times is page furniture."""
    text = "\n".join(
        [
            "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY",
            "Real content on page 1.",
            "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY",
            "Real content on page 2.",
            "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY",
            "Real content on page 3.",
        ]
    )
    cleaned = _strip_repeated_lines(text, min_repeats=3)
    assert "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY" not in cleaned
    assert "Real content on page 1." in cleaned
    assert "Real content on page 3." in cleaned


def test_strip_repeated_lines_keeps_lines_below_threshold():
    """A line repeated only twice is not confidently boilerplate - keep it."""
    text = "Heading\nOnly twice\nbody text\nOnly twice"
    cleaned = _strip_repeated_lines(text, min_repeats=3)
    assert "Only twice" in cleaned


def test_strip_repeated_lines_ignores_long_lines():
    """A long line repeating is very unlikely to be a header/footer and more
    likely a real (if oddly duplicated) sentence - the length cap protects it."""
    long_line = "This is a genuinely long sentence that happens to repeat. " * 2
    text = "\n".join([long_line] * 4)
    cleaned = _strip_repeated_lines(text, min_repeats=3, max_len=50)
    assert long_line.strip() in cleaned


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


# ---------------------------------------------------------------------------
# PDF tables
# ---------------------------------------------------------------------------


def _pdf_with_table(rows, before="The code is shown as follows:", pages_of_furniture=0):
    """A real PDF with a ruled table, drawn with pymupdf, so the test runs
    the same table detection the corpus goes through."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 60), before, fontsize=10)
    x0, y0, col_w, row_h = 50, 80, 160, 20
    cols = len(rows[0])
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * col_w, y0 + r * row_h, x0 + (c + 1) * col_w, y0 + (r + 1) * row_h
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 4, rect.y1 - 6), cell, fontsize=9)
    page.insert_text((50, y0 + len(rows) * row_h + 30), "Text after the table.", fontsize=10)
    for _ in range(pages_of_furniture):
        extra = doc.new_page()
        extra.insert_text((50, 60), "SRM RUNNING HEADER", fontsize=10)
        extra.insert_text((50, 90), "Body text of another page.", fontsize=10)
    data = doc.tobytes()
    doc.close()
    assert cols  # rows are non-empty
    return data


def test_a_pdf_table_becomes_one_line_per_row_with_its_headers():
    """Real failure: the attendance code table (R7.2) came out one cell per
    line, so '85 to 94%' lost its code and the grade table lost which mark
    range went with which grade."""
    from policyverify.ingest.extract import extract_pdf

    text = extract_pdf(_pdf_with_table([
        ["Attendance rounded to", "Code"],
        ["95% and above", "H"],
        ["Below 75%", "L"],
    ]))
    assert "Attendance rounded to: 95% and above | Code: H." in text
    assert "Attendance rounded to: Below 75% | Code: L." in text
    # The text around the table is kept, in order.
    assert text.index("as follows") < text.index("95% and above") < text.index("after the table")


def test_a_cell_merged_down_is_copied_into_the_rows_it_spans():
    from policyverify.ingest.extract import _pdf_table_lines

    class _Row:
        def __init__(self, cells):
            self.cells = cells

    class _Table:
        def __init__(self, rows):
            self.rows = [_Row(r) for r in rows]

    # Cells are bounding boxes; one word sits in each box except the merged one.
    boxes = [[(0, r * 10, 50, r * 10 + 10), (50, r * 10, 100, r * 10 + 10)] for r in range(3)]
    boxes[2][0] = None  # merged down from row 1
    words = [
        (5, 2, 20, 8, "Category", 0, 0, 0), (55, 2, 70, 8, "Class", 0, 0, 1),
        (5, 12, 20, 18, "Regular", 0, 1, 0), (55, 12, 70, 18, "First", 0, 1, 1),
        (55, 22, 70, 28, "Second", 0, 2, 0),
    ]
    lines = _pdf_table_lines(_Table(boxes), words)
    assert lines == ["Category: Regular | Class: First.", "Category: Regular | Class: Second."]


def test_contents_pages_and_tables_without_a_header_are_left_alone():
    from policyverify.ingest.extract import extract_pdf

    toc = extract_pdf(_pdf_with_table([["Description", "Page"], ["Admission", "6"]]))
    assert " | " not in toc
    headerless = extract_pdf(_pdf_with_table([["", "Code"], ["95% and above", "H"]]))
    assert " | " not in headerless


def test_rendering_a_table_does_not_change_which_lines_count_as_page_furniture():
    """Real failure: rendering the hostel contacts table removed its serial
    numbers, so page numbers no longer repeated often enough to be stripped
    and contents entries stole real sections' citation IDs."""
    from policyverify.ingest.extract import extract_pdf

    text = extract_pdf(
        _pdf_with_table([["Name", "Phone"], ["Sannasi A", "8056016605"]], pages_of_furniture=3)
    )
    assert "SRM RUNNING HEADER" not in text
    assert "Name: Sannasi A | Phone: 8056016605." in text


def test_content_before_the_first_heading_is_kept_as_a_preamble():
    """Real failure: the plagiarism policy's Preamble and its DEFINITION of
    plagiarism came before the first detected heading and were dropped, so
    'what is plagiarism?' had no source in the index."""
    from policyverify.ingest.normalize import split_into_sections

    text = (
        "Plagiarism Policy\nPreamble:\n" + "Ethics and honesty matter in research. " * 6
        + "\nDefinition of Plagiarism: presenting another person's work as one's own.\n"
        "2. Invalid Sources (Misleading Citation)\nInvalid source attribution occurs when ...\n"
        "3. Duplication (Self Plagiarism)\nDuplication happens when ...\n"
    )
    sections = split_into_sections(text, "Plagiarism Policy")
    assert sections[0].heading == "Preamble"
    assert "Definition of Plagiarism" in sections[0].text


def test_a_bare_title_before_the_first_heading_is_not_a_preamble():
    from policyverify.ingest.normalize import split_into_sections

    text = "Examination Policy\n1. Registration Of Courses\nStudents register ...\n2. Award Of Degree\nDegrees ...\n"
    assert [s.heading for s in split_into_sections(text, "Examination Policy")] == [
        "Registration Of Courses", "Award Of Degree"]


def test_bracketed_glosses_do_not_decide_whether_a_line_is_a_heading():
    from policyverify.ingest.normalize import _is_numbered_heading

    # Rejected before - the plagiarism policy's type 1.
    assert _is_numbered_heading("1. Secondary sources (Inaccurate citation)")
    # Must stay a heading: counting the bracket's lowercase words broke it.
    assert _is_numbered_heading("1. Founder's Scholarship (applicable for 4 years)")
    # Still not a heading: sentence-case prose.
    assert not _is_numbered_heading("1. all examinations undertaken shall be cancelled")
