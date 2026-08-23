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

from policyverify.ingest.extract import _strip_repeated_lines, extract_html
from policyverify.ingest.normalize import build_document, split_into_sections
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
