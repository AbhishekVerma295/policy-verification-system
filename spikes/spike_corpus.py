"""
SPIKE 1 of 3 - can we get clean text out of these policy documents?

WHAT A SPIKE IS
    A quick throwaway test you run BEFORE building on top of something, to
    check that the thing actually works. Nothing here is production code and
    none of it gets imported by the real system.

WHY THIS ONE MATTERS
    The corpus is the one decision in this project that is genuinely expensive
    to reverse. Everything downstream - chunking, citations, retrieval - is
    built on the assumption that we can reliably get clean, structured text
    out of these documents. If a university publishes policies as scanned
    images or badly-laid-out multi-column PDFs, you want to find that out in
    week 1, not in week 6 with the chunker already written.

HOW TO USE IT
    Pick 5 candidate universities. Find one publicly accessible policy page
    from each (attendance or academic integrity are usually easiest to find).
    Then:

        python spikes/spike_corpus.py <url1> <url2> <url3> <url4> <url5>

    Keep the 3-4 that score GOOD or OK. Drop the rest and find replacements.

    Choosing your universities based on which ones parse cleanly is a sensible
    engineering decision, not cheating. Nothing about this project requires any
    particular institution, so there is no reason to fight a bad PDF.

WHAT TO LOOK FOR
    - "headings found" should be more than a handful. Zero headings means we
      cannot split the document into sections, which breaks citations.
    - "numbered sections" is a bonus. If the document numbers its own sections
      ("4.2 Plagiarism") those numbers become the citation IDs, which is much
      nicer than generating slugs from headings.
    - "text ratio" below ~0.7 usually means the extractor produced garbage.
    - Always read the preview at the bottom with your own eyes. The numbers can
      look fine while the text is subtly scrambled.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field

import requests

# Matches "4", "4.2", "4.2.1", "A.1" at the start of a line - the kind of
# numbering that makes a really good citation ID.
NUMBERED_SECTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)*|[A-Z]\.\d+)[\.\)]?\s+\S")

TIMEOUT = 30
HEADERS = {
    # Some university sites block requests that do not look like a browser.
    "User-Agent": "Mozilla/5.0 (compatible; PolicyVerifySpike/0.1; academic use)"
}


@dataclass
class Report:
    """What we learned about one candidate document."""

    url: str
    ok: bool = False
    error: str = ""
    fmt: str = ""
    raw_bytes: int = 0
    text: str = ""
    headings: int = 0
    numbered: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def text_ratio(self) -> float:
        """Fraction of characters that are letters, digits, spaces or basic
        punctuation. A low value means the extractor emitted junk."""
        if not self.text:
            return 0.0
        good = sum(1 for c in self.text if c.isalnum() or c in " .,;:()-'\"%/\n")
        return good / len(self.text)

    @property
    def avg_line_len(self) -> float:
        """Very short average lines usually mean a PDF was extracted
        column-by-column and the sentences are shredded."""
        lines = [ln for ln in self.text.splitlines() if ln.strip()]
        if not lines:
            return 0.0
        return sum(len(ln) for ln in lines) / len(lines)

    def verdict(self) -> str:
        """A blunt GOOD / OK / POOR call, so the output is skimmable."""
        if not self.ok:
            return "FAILED"
        if self.chars < 500:
            return "POOR"
        if self.text_ratio < 0.70:
            return "POOR"
        if self.headings == 0 and self.numbered == 0:
            return "POOR"
        if self.headings >= 5 and self.chars > 2000 and self.text_ratio > 0.85:
            return "GOOD"
        return "OK"


def fetch(url: str) -> tuple[bytes, str]:
    """Download the URL. Returns (raw bytes, detected format)."""
    resp = requests.get(url, timeout=TIMEOUT, headers=HEADERS)
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").lower()
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        return resp.content, "pdf"
    return resp.content, "html"


def extract_html(raw: bytes) -> tuple[str, int]:
    """HTML to text. Returns (text, number of headings found)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw, "lxml")

    # Strip the furniture. Navigation and cookie banners are not policy text,
    # and leaving them in would pollute both the chunks and the search index.
    for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
        tag.decompose()

    headings = len(soup.find_all(["h1", "h2", "h3", "h4"]))

    # Prefer the main content region when the page marks one up.
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n")

    # Collapse the runs of blank lines that get_text tends to leave behind.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return text.strip(), headings


def extract_pdf(raw: bytes) -> tuple[str, int]:
    """PDF to text. Returns (text, number of heading-ish lines found).

    PDFs have no real concept of a heading, so we guess: a short line, not
    ending in a full stop, that is either numbered or title-case.
    """
    import pymupdf

    doc = pymupdf.open(stream=raw, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    text = "\n".join(pages)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    headings = 0
    for line in text.splitlines():
        s = line.strip()
        if 3 < len(s) < 90 and not s.endswith("."):
            if NUMBERED_SECTION_RE.match(s) or (s.istitle() and len(s.split()) <= 10):
                headings += 1
    return text, headings


def analyse(url: str) -> Report:
    """Download one document and report how usable it looks."""
    rep = Report(url=url)
    try:
        raw, fmt = fetch(url)
        rep.fmt = fmt
        rep.raw_bytes = len(raw)

        if fmt == "pdf":
            rep.text, rep.headings = extract_pdf(raw)
        else:
            rep.text, rep.headings = extract_html(raw)

        rep.numbered = sum(
            1 for ln in rep.text.splitlines() if NUMBERED_SECTION_RE.match(ln)
        )
        rep.ok = True

        # Human-readable warnings, which are usually more useful than the score.
        if rep.fmt == "pdf":
            rep.notes.append("PDF - prefer an HTML source if one exists")
        if rep.avg_line_len < 40 and rep.chars > 500:
            rep.notes.append("very short lines - text may be shredded by columns")
        if rep.numbered == 0:
            rep.notes.append("no numbered sections - citations will use heading slugs")
        if rep.chars > 0 and rep.text_ratio < 0.8:
            rep.notes.append("lots of odd characters - check the preview carefully")

    except requests.HTTPError as e:
        rep.error = f"HTTP {e.response.status_code}"
    except requests.RequestException as e:
        rep.error = f"network error: {type(e).__name__}"
    except Exception as e:  # extraction can fail in many creative ways
        rep.error = f"{type(e).__name__}: {e}"

    return rep


def print_report(rep: Report, index: int) -> None:
    print("\n" + "=" * 78)
    print(f"[{index}] {rep.url}")
    print("=" * 78)

    if not rep.ok:
        print(f"  VERDICT: FAILED  ({rep.error})")
        return

    print(f"  VERDICT: {rep.verdict()}")
    print(f"  format            : {rep.fmt}")
    print(f"  downloaded        : {rep.raw_bytes:,} bytes")
    print(f"  extracted text    : {rep.chars:,} characters")
    print(f"  headings found    : {rep.headings}")
    print(f"  numbered sections : {rep.numbered}")
    print(f"  text ratio        : {rep.text_ratio:.2f}   (want > 0.85)")
    print(f"  avg line length   : {rep.avg_line_len:.0f}   (want > 40)")

    for note in rep.notes:
        print(f"  ! {note}")

    preview = rep.text[:600].replace("\n", "\n    ")
    print("\n  --- first 600 characters (READ THIS, do not just trust the numbers) ---")
    print(f"    {preview}")


def main(urls: list[str]) -> int:
    if not urls:
        print(__doc__)
        print("\nERROR: give me some URLs to test.\n")
        print("  python spikes/spike_corpus.py <url1> <url2> ...\n")
        return 1

    print(f"\nTesting {len(urls)} candidate document(s)...")
    reports = [analyse(u) for u in urls]

    for i, rep in enumerate(reports, 1):
        print_report(rep, i)

    # Summary table, so the decision is easy to make at a glance.
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'#':<3} {'VERDICT':<8} {'FMT':<5} {'CHARS':>8} {'HEAD':>5} {'NUM':>5}  URL")
    for i, rep in enumerate(reports, 1):
        print(
            f"  {i:<3} {rep.verdict():<8} {rep.fmt or '-':<5} {rep.chars:>8,} "
            f"{rep.headings:>5} {rep.numbered:>5}  {rep.url[:34]}"
        )

    good = [r for r in reports if r.verdict() in ("GOOD", "OK")]
    print(f"\n  {len(good)} of {len(reports)} look usable.")
    if len(good) >= 3:
        print("  -> Enough to proceed. Put the best 3-4 into data/manifest.yaml.")
    else:
        print("  -> Not enough yet. Find more candidates and run this again.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
