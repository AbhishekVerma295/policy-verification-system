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

Known limitation: a short breadcrumb-adjacent line and a small footer
prompt ("Where could your journey at SRMIST take you?") can still leak
into the very start or end of the extracted text on some pages. This was
checked directly against srmist.edu.in on 2026-08-19 and left in
deliberately rather than chased further - it produces at most one small,
low-quality chunk once Phase 2 splits the document, which real policy
questions are very unlikely to retrieve.
"""

from __future__ import annotations

import re
from collections import Counter

import pymupdf
from bs4 import BeautifulSoup

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

_BLANK_RUN_RE = re.compile(r"\n{3,}")


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

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")
    return _clean_text(text)


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
