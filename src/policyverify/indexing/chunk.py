"""
chunk.py - Document -> searchable Chunks.

A chunk is the unit the search engine returns and the verifier checks claims
against. Getting the boundaries right matters more than it sounds: a chunk
that spans two unrelated rules will happily "support" a claim that mixes them
together, and that is exactly the failure this whole project exists to catch.

THE RULE: never split across a section boundary.
Sections come from normalize.py and correspond to real headings in the real
document, so one section is one citable thing. Chunking respects that, which
is what lets a citation like `srm/attendance/4.2` mean something exact.

Two things this has to cope with, both measured from the real SRM corpus
rather than assumed (147 sections, 2026-08-19):

  Oversized sections.  74% of all the text in the corpus lives in just 14%
    of the sections, and the largest single section is ~42,000 characters -
    far too big to embed usefully. Those get split into overlapping windows.
    Every window keeps the SAME citation_id, because they are all genuinely
    part of that one section; they differ only in chunk_id. That split is
    the reason `chunk_id` and `citation_id` are separate fields in the
    schema.

  Table-of-contents noise.  29% of sections are under 120 characters, and
    inspecting them showed most are contents-page entries whose entire body
    is a page number ("Payment Of Fine" -> "12"). Those are dropped, since
    they contain no rule to cite and would only pollute search results.
"""

from __future__ import annotations

import re

from policyverify.config import ChunkingConfig, get_config
from policyverify.schema import Chunk, Document, Section

# A "word" for noise detection: a token containing at least one letter.
# Page numbers ("12"), bare punctuation and "S.No." style fragments do not
# clear this bar, which is exactly what we want to drop.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# A section whose body has fewer real words than this is treated as
# table-of-contents furniture rather than policy text.
#
# Deliberately set low, because the two kinds of mistake are not equally bad.
# Noise that survives the filter just sits in the index scoring badly, and the
# verifier catches it downstream. Real content that gets dropped can never be
# retrieved or cited by anything, ever - the question becomes permanently
# unanswerable and nothing downstream can recover it. So when in doubt, keep.
#
# Three clears the observed noise ("12", "13", "S.No.") while keeping short
# real rules: "Attend 75% of classes." is four words and must survive.
_MIN_REAL_WORDS = 3

# Prefer to break long text at a blank line, then at a sentence end. Both
# keep chunks readable; a hard character cut is the last resort.
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def is_noise_section(section: Section) -> bool:
    """True if this section is contents-page furniture rather than content.

    Detected by counting real words, not characters: the observed noise in
    the SRM corpus is short *because it is a page number*, and word count
    separates that from a short-but-real rule far more reliably than length.
    """
    return len(_WORD_RE.findall(section.text)) < _MIN_REAL_WORDS


def split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split text into <= max_chars windows, preferring natural boundaries.

    Overlap exists so that a rule sitting right on a window boundary is not
    cut in half and lost to both windows. It costs a little duplication and
    buys not silently losing the one sentence someone asked about.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # Break into the smallest natural units we can, largest-first:
    # paragraphs, then sentences for any paragraph still too long.
    units: list[str] = []
    for para in _PARAGRAPH_SPLIT_RE.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in _SENTENCE_SPLIT_RE.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= max_chars:
                units.append(sentence)
            else:
                # A single sentence longer than the window. Rare, but real
                # (dense legal prose with no full stops). Hard-cut it.
                for i in range(0, len(sentence), max_chars):
                    units.append(sentence[i : i + max_chars])

    # Greedily pack units into windows, carrying a tail of the previous
    # window forward as overlap.
    windows: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            windows.append(current)
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            # Start the overlap at a word boundary so it does not begin
            # mid-word, which reads as corrupt text to both a human and
            # the embedding model.
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            current = f"{tail}\n\n{unit}".strip() if tail else unit
        else:
            current = unit
    if current:
        windows.append(current)

    return windows


def chunk_document(doc: Document, config: ChunkingConfig | None = None) -> list[Chunk]:
    """Turn one Document into the chunks that will be indexed and searched."""
    cfg = config or get_config().chunking
    chunks: list[Chunk] = []

    # Citations are assigned at document level, not per section, because
    # uniqueness needs to see all the sections at once - see
    # Document.citations().
    citations = doc.citations()

    for section, citation_id in zip(doc.sections, citations, strict=True):
        if is_noise_section(section):
            continue

        citation = citation_id.render()
        section_path = " > ".join(section.path) if section.path else section.heading

        pieces = split_long_text(section.text, cfg.max_chars, cfg.overlap_chars)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    # Same citation for every piece of one section - they are
                    # all that section. Only the chunk_id distinguishes them.
                    chunk_id=f"{citation}#{i}",
                    text=piece,
                    citation_id=citation,
                    university=doc.university,
                    university_name=doc.university_name,
                    policy_type=doc.policy_type,
                    section_path=section_path,
                    source_url=doc.source_url,
                )
            )

    return chunks


def chunk_documents(docs: list[Document], config: ChunkingConfig | None = None) -> list[Chunk]:
    """Chunk several documents, preserving order."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, config))
    return out
