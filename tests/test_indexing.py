"""
Tests for Phase 2: chunking and the vector index.

The chunking tests are pure functions and run instantly. The store tests use
a fake embedder rather than loading a real model, so this whole file needs no
GPU, no network and no model download - the real index build is exercised by
scripts/build_index.py instead.

Several tests here encode failures found by running against the real SRM
corpus, not hypotheticals. Those are marked in their docstrings.
"""

from __future__ import annotations

from datetime import date

import pytest

from policyverify.config import ChunkingConfig, load_config
from policyverify.indexing.chunk import (
    chunk_document,
    is_noise_section,
    split_long_text,
)
from policyverify.indexing.store import (
    IndexMismatchError,
    VectorStore,
    embedding_text,
)
from policyverify.schema import Chunk, Document, PolicyType, Section, SourceFormat


def _doc(sections: list[Section], policy=PolicyType.ATTENDANCE) -> Document:
    return Document(
        doc_id=f"srm/{policy.value}",
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=policy,
        title="Test Policy",
        source_url="https://example.srmist.edu.in/policy",
        source_format=SourceFormat.HTML,
        retrieved_at=date(2026, 8, 19),
        checksum="a" * 64,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Noise detection
# ---------------------------------------------------------------------------


def test_noise_section_detects_contents_page_entries():
    """Real failure from SRM's hostel rules: contents-page entries whose whole
    body is a page number ("Payment Of Fine" -> "12")."""
    assert is_noise_section(Section(heading="Payment Of Fine", text="12"))
    assert is_noise_section(Section(heading="Mess", text="13"))
    assert is_noise_section(Section(heading="Contents", text="S.No."))


def test_noise_section_keeps_short_but_real_rules():
    """Word count, not character count, is the signal - a short real rule must
    survive."""
    section = Section(
        heading="Attendance", text="Students must attend at least 75% of classes."
    )
    assert not is_noise_section(section)


# ---------------------------------------------------------------------------
# split_long_text
# ---------------------------------------------------------------------------


def test_split_leaves_short_text_alone():
    assert split_long_text("Short text.", max_chars=1000, overlap_chars=100) == [
        "Short text."
    ]


def test_split_respects_max_chars():
    text = "\n\n".join(f"Paragraph number {i} with some filler words in it." for i in range(60))
    pieces = split_long_text(text, max_chars=300, overlap_chars=50)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 300 + 50, "window exceeded max_chars beyond its overlap"


def test_split_preserves_all_content():
    """Splitting must not silently drop a rule. Every source paragraph has to
    appear in at least one window."""
    paragraphs = [f"Rule {i}: something specific and unique to this rule." for i in range(30)]
    text = "\n\n".join(paragraphs)
    pieces = split_long_text(text, max_chars=200, overlap_chars=40)
    joined = " ".join(pieces)
    for para in paragraphs:
        assert para in joined, f"lost content: {para!r}"


def test_split_handles_a_single_sentence_longer_than_the_window():
    """Dense legal prose sometimes runs for hundreds of characters with no
    full stop. It still has to be split rather than crash or overflow."""
    text = "word " * 500  # one long run, no sentence boundaries
    pieces = split_long_text(text, max_chars=200, overlap_chars=20)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 200 + 20


def test_split_overlap_starts_at_a_word_boundary():
    text = "\n\n".join(f"Sentence {i} here with content." for i in range(40))
    pieces = split_long_text(text, max_chars=200, overlap_chars=50)
    for piece in pieces[1:]:
        assert not piece.startswith(" ")


def test_split_empty_text_gives_nothing():
    assert split_long_text("", max_chars=100, overlap_chars=10) == []
    assert split_long_text("   \n  ", max_chars=100, overlap_chars=10) == []


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------


def test_chunk_document_never_merges_two_sections():
    """The core rule. A chunk spanning two sections would happily 'support' a
    claim that mixes two unrelated rules together."""
    doc = _doc(
        [
            Section(number="1", heading="Attendance", text="Students must attend 75% of classes."),
            Section(number="2", heading="Exemptions", text="Medical exemptions may be granted."),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    for chunk in chunks:
        assert not ("75%" in chunk.text and "Medical" in chunk.text)


def test_chunk_document_splits_oversized_sections_keeping_one_citation():
    """A section too big to embed is split into several chunks, but they all
    remain the same citable section - only chunk_id differs."""
    long_text = "\n\n".join(f"Clause {i} of this long section." for i in range(200))
    doc = _doc([Section(number="4.2", heading="Long Section", text=long_text)])
    chunks = chunk_document(doc, ChunkingConfig(max_chars=500, min_chars=50, overlap_chars=50))

    assert len(chunks) > 1, "oversized section should have been split"
    citations = {c.citation_id for c in chunks}
    assert len(citations) == 1, "all pieces of one section share one citation"
    assert len({c.chunk_id for c in chunks}) == len(chunks), "chunk_ids must be unique"
    assert all(c.chunk_id.startswith(f"{chunks[0].citation_id}#") for c in chunks)


def test_chunk_document_drops_noise_sections():
    doc = _doc(
        [
            Section(heading="Payment Of Fine", text="12"),
            Section(heading="Real Rule", text="Students must pay fees before the deadline."),
        ]
    )
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert "fees" in chunks[0].text


def test_chunk_ids_unique_when_headings_repeat():
    """Real failure that broke the first index build: SRM's Code of Conduct has
    six sections all headed "Consequence", which produced duplicate chunk ids
    and Chroma rejected the whole batch."""
    doc = _doc(
        [
            Section(heading="Consequence", text="First consequence rule text here."),
            Section(heading="Consequence", text="Second consequence rule text here."),
            Section(heading="Consequence", text="Third consequence rule text here."),
        ],
        policy=PolicyType.CODE_OF_CONDUCT,
    )
    chunks = chunk_document(doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert len({c.citation_id for c in chunks}) == len(chunks)


def test_chunks_carry_everything_needed_to_cite_them():
    """A chunk comes back from the search engine without its document, so it
    has to already know how to cite itself."""
    doc = _doc([Section(number="4.2", heading="Attendance", text="Attend 75% of classes please.")])
    chunk = chunk_document(doc)[0]
    assert chunk.university == "srm"
    assert chunk.policy_type is PolicyType.ATTENDANCE
    assert chunk.citation_id.startswith("srm/attendance/4.2")
    assert chunk.source_url
    assert chunk.section_path


def test_chunk_metadata_is_chroma_safe():
    doc = _doc([Section(number="4.2", heading="Attendance", text="Attend 75% of classes please.")])
    meta = chunk_document(doc)[0].to_metadata()
    for key, value in meta.items():
        assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value)}"


# ---------------------------------------------------------------------------
# embedding_text - what gets embedded vs what gets verified
# ---------------------------------------------------------------------------


def test_embedding_text_adds_context_but_chunk_text_stays_raw():
    """Search benefits from the heading; verification must not see it. A claim
    has to be checked against what the university wrote, not against a string
    we assembled ourselves."""
    chunk = Chunk(
        chunk_id="srm/attendance/4.2#0",
        text="Students must attend at least 75% of classes.",
        citation_id="srm/attendance/4.2",
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Attendance Policy > Minimum Attendance",
        source_url="https://example.srmist.edu.in/a",
    )
    embedded = embedding_text(chunk)
    assert "Minimum Attendance" in embedded
    assert "SRM Institute" in embedded
    assert chunk.text in embedded
    # the stored text itself is untouched
    assert chunk.text == "Students must attend at least 75% of classes."


# ---------------------------------------------------------------------------
# The index/model mismatch guard
# ---------------------------------------------------------------------------


def test_search_refuses_when_no_index_exists(tmp_path):
    config = load_config()
    config.paths.index = str(tmp_path / "empty-index")
    store = VectorStore(config)
    with pytest.raises(IndexMismatchError, match="No index found"):
        store.check_ready()


def test_search_refuses_on_embedding_model_mismatch(tmp_path):
    """The bug this guard exists for is silent: embeddings from two different
    models are not comparable, but comparing them raises nothing. Search just
    quietly returns near-random passages."""
    import json

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_dim": 768,
                "chunk_count": 231,
                "document_count": 6,
                "built_at": "2026-08-19T00:00:00+00:00",
                "chunking": {},
            }
        ),
        encoding="utf-8",
    )

    config = load_config()
    config.paths.index = str(index_dir)
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    store = VectorStore(config)
    with pytest.raises(IndexMismatchError) as excinfo:
        store.check_ready()
    message = str(excinfo.value)
    # the message has to be actionable, not just "error"
    assert "bge-base" in message
    assert "MiniLM" in message
    assert "build_index" in message


def test_check_ready_passes_when_models_match(tmp_path):
    import json

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index_manifest.json").write_text(
        json.dumps(
            {
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_dim": 768,
                "chunk_count": 231,
                "document_count": 6,
                "built_at": "2026-08-19T00:00:00+00:00",
                "chunking": {},
            }
        ),
        encoding="utf-8",
    )
    config = load_config()
    config.paths.index = str(index_dir)
    config.embedding.model_name = "BAAI/bge-base-en-v1.5"

    manifest = VectorStore(config).check_ready()
    assert manifest.chunk_count == 231


def test_build_refuses_empty_chunk_list(tmp_path):
    config = load_config()
    config.paths.index = str(tmp_path / "index")
    with pytest.raises(ValueError, match="zero chunks"):
        VectorStore(config).build([])
