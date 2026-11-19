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


def test_noise_section_drops_short_titles_with_nothing_under_them():
    """Real case: the examination policy's part title, left as the whole body
    of a section once its sub-headings were detected."""
    assert is_noise_section(
        Section(heading="Section B", text="EXAMINATIONS – RULES AND REGULATIONS")
    )


def test_noise_section_drops_unfinished_fragments():
    """Real cases from the hostel rules' cover page and the regulations."""
    assert is_noise_section(
        Section(heading="Srmist", text="Hostels,\nSRM Nagar, Kattankulathur – 603")
    )
    assert is_noise_section(
        Section(
            heading="Rules And Regulations",
            text="(updated Jan 2025)\nFor Further Details Contact:\nManager,",
        )
    )
    assert is_noise_section(Section(heading="Matrix", text="And, every course has well defined:"))


def test_noise_section_keeps_short_real_rules_and_contacts():
    """The new short-section rules must not catch a real one-line rule, a
    contact block, or anything long - when in doubt, keep."""
    keep = [
        "The Chief Guest will deliver the Convocation address.",
        "warden.esqa.ktr@srmist.edu.in\n8056016636\n17",
        "Ragging is banned: any student found guilty will be expelled.",
    ]
    assert not is_noise_section(Section(heading="Convocation", text=keep[0]))
    assert not is_noise_section(Section(heading="Esq A", text=keep[1]))
    assert not is_noise_section(Section(heading="Ragging", text=keep[2]))
    # Anything of 60 characters or more is judged by word count alone.
    long_fragment = "The following rules apply to every student in every SRM hostel,"
    assert len(long_fragment) >= 60
    assert not is_noise_section(Section(heading="Hostel", text=long_fragment))
    # A long ALL CAPS body is a real (shouted) rule, not a title.
    shouted = "SMOKING AND ALCOHOL ARE STRICTLY PROHIBITED INSIDE THE HOSTEL PREMISES"
    assert not is_noise_section(Section(heading="Prohibited", text=shouted))


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


# ---------------------------------------------------------------------------
# get_by_citation ordering
# ---------------------------------------------------------------------------


class _StubCollection:
    """Returns rows in a deliberately shuffled order, like Chroma may."""

    def __init__(self, order):
        self.order = order

    def get(self, where=None):
        meta = {
            "citation_id": CITATION_FOR_ORDER,
            "university": "srm",
            "university_name": "SRM Institute of Science and Technology",
            "policy_type": "attendance",
            "section_path": "Policy > Section",
            "source_url": "https://example.srmist.edu.in/p",
        }
        return {
            "ids": [f"{CITATION_FOR_ORDER}#{i}" for i in self.order],
            "documents": [f"piece {i}" for i in self.order],
            "metadatas": [dict(meta) for _ in self.order],
        }


CITATION_FOR_ORDER = "srm/attendance/4.2-long_section"


def _store_with(order):
    from policyverify.indexing.store import VectorStore

    store = VectorStore(load_config())
    store._collection = _StubCollection(order)
    return store


def test_get_by_citation_returns_chunks_in_document_order():
    """Chroma promises nothing about row order, but the verifier scores these
    as premises in sequence and the eval drift check hashes them joined - both
    need a stable order."""
    store = _store_with([3, 0, 2, 1])
    chunks = store.get_by_citation(CITATION_FOR_ORDER)
    assert [c.chunk_id for c in chunks] == [
        f"{CITATION_FOR_ORDER}#{i}" for i in range(4)
    ]


def test_get_by_citation_sorts_numerically_not_lexically():
    """#10 must come after #9, which string sorting would get wrong."""
    store = _store_with([10, 2, 1])
    chunks = store.get_by_citation(CITATION_FOR_ORDER)
    assert [c.chunk_id.rsplit("#", 1)[1] for c in chunks] == ["1", "2", "10"]


def test_get_by_citation_is_stable_across_calls():
    first = _store_with([2, 0, 1]).get_by_citation(CITATION_FOR_ORDER)
    second = _store_with([1, 2, 0]).get_by_citation(CITATION_FOR_ORDER)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


# ---------------------------------------------------------------------------
# The cached keyword index and the similarity lookup
# ---------------------------------------------------------------------------


class _MemoryCollection:
    """Just enough of a Chroma collection, held in memory: add, and get by
    ids or everything. Rows come back reversed on purpose - Chroma promises
    no order, so nothing here may depend on one."""

    def __init__(self):
        self.rows: dict[str, tuple[str, list[float], dict]] = {}

    def add(self, ids, documents, embeddings, metadatas):
        for cid, doc, vec, meta in zip(ids, documents, embeddings, metadatas, strict=True):
            self.rows[cid] = (doc, list(vec), meta)

    def get(self, ids=None, where=None, include=None):
        keys = [k for k in self.rows if ids is None or k in ids][::-1]
        return {
            "ids": keys,
            "documents": [self.rows[k][0] for k in keys],
            "embeddings": [self.rows[k][1] for k in keys],
            "metadatas": [self.rows[k][2] for k in keys],
        }

    def count(self):
        return len(self.rows)


class _MemoryClient:
    def __init__(self):
        self.collection = None

    def delete_collection(self, name):
        self.collection = None

    def get_or_create_collection(self, name, metadata=None):
        if self.collection is None:
            self.collection = _MemoryCollection()
        return self.collection


class _FakeEmbedder:
    """Every chunk gets the same unit vector; the tests here are about the
    keyword index, not about embeddings."""

    def encode_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]


def _memory_store(tmp_path):
    config = load_config()
    config.paths.index = str(tmp_path / "index")
    store = VectorStore(config, embedder=_FakeEmbedder())
    store._client = _MemoryClient()
    return store


def _plain_chunk(name: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"srm/attendance/{name}#0",
        text=text,
        citation_id=f"srm/attendance/{name}",
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path=f"Policy > {name}",
        source_url="https://example.srmist.edu.in/p",
    )


def test_bm25_index_is_built_once_and_cached(tmp_path):
    store = _memory_store(tmp_path)
    store.build([_plain_chunk("a", "Students must attend classes.")])
    first = store.bm25_index()
    assert store.bm25_index() is first
    assert len(first) == 1


def test_rebuilding_the_index_replaces_the_cached_keyword_index(tmp_path):
    """A keyword index left over from the old corpus would keep returning
    sections that no longer exist - and nothing would raise."""
    store = _memory_store(tmp_path)
    store.build([_plain_chunk("old", "The condonation rule.")])
    assert store.bm25_index().search("condonation", k=5)

    store.build([_plain_chunk("new", "The hostel curfew rule.")])
    assert store.bm25_index().search("condonation", k=5) == []
    hits = store.bm25_index().search("curfew", k=5)
    assert [c.chunk_id for c, _ in hits] == ["srm/attendance/new#0"]


def test_all_chunks_is_sorted_whatever_order_chroma_returns(tmp_path):
    store = _memory_store(tmp_path)
    store.build([_plain_chunk(n, f"text {n}") for n in ("b", "c", "a")])
    assert [c.chunk_id for c in store.all_chunks()] == [
        "srm/attendance/a#0",
        "srm/attendance/b#0",
        "srm/attendance/c#0",
    ]


def test_similarities_match_each_vector_to_its_own_id():
    """The lookup must pair each returned embedding with ITS id - Chroma may
    return the rows in any order, so zipping against the request would mix
    up which chunk gets which similarity."""
    store = VectorStore(load_config())
    collection = _MemoryCollection()
    collection.add(
        ids=["x", "y", "z"],
        documents=["", "", ""],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]],
        metadatas=[{}, {}, {}],
    )
    store._collection = collection

    sims = store.similarities([1.0, 0.0], ["x", "y", "z"])
    assert sims["x"] == pytest.approx(1.0)
    assert sims["y"] == pytest.approx(0.0)
    assert sims["z"] == pytest.approx(0.6)
    assert store.similarities([1.0, 0.0], []) == {}


# ---------------------------------------------------------------------------
# Section summaries
# ---------------------------------------------------------------------------


class _FakeLLM:
    """Records prompts and returns a fixed summary, so no model is needed."""

    def __init__(self, reply="Summary: Students must attend 75% of classes."):
        self.reply = reply
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, json_mode: bool = True) -> str:
        assert json_mode is False, "summaries are prose, not JSON"
        self.prompts.append(prompt)
        return self.reply


class _DeadLLM:
    calls = 0

    def generate(self, prompt: str, *, json_mode: bool = True) -> str:
        from policyverify.llm import LLMError

        _DeadLLM.calls += 1
        raise LLMError("connection refused")


def test_default_chunks_are_small():
    """The original 1800-character windows held several unrelated rules each."""
    cfg = ChunkingConfig()
    assert cfg.max_chars <= 800
    assert cfg.overlap_chars < cfg.max_chars // 4


def test_every_piece_of_a_section_carries_the_same_section_summary():
    from policyverify.indexing.summarize import SectionSummarizer

    long_text = "\n\n".join(f"Clause {i} of the attendance rule." for i in range(100))
    doc = _doc([Section(number="4", heading="Attendance", text=long_text)])
    cfg = ChunkingConfig(max_chars=300, overlap_chars=30)
    llm = _FakeLLM()
    summarizer = SectionSummarizer(llm, cfg)
    chunks = chunk_document(doc, cfg, summarize=summarizer.summarise)

    assert len(chunks) > 1
    assert {c.summary for c in chunks} == {"Students must attend 75% of classes."}
    assert len(llm.prompts) == 1, "one summary per section, not per chunk"
    # The summary is context only - the chunk text is still the source text.
    assert all("75%" not in c.text for c in chunks)


def test_summary_goes_into_the_embedding_but_not_the_chunk_text():
    chunk = Chunk(
        chunk_id="srm/attendance/4.2#3",
        text="Clause 7 applies in that case.",
        citation_id="srm/attendance/4.2",
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Attendance > Condonation",
        source_url="https://example.srmist.edu.in/a",
        summary="Students below 75% attendance may apply for condonation.",
    )
    assert "condonation" in embedding_text(chunk)
    assert chunk.text == "Clause 7 applies in that case."
    assert chunk.to_metadata()["summary"] == chunk.summary


def test_long_sections_are_summarised_in_parts_then_combined():
    from policyverify.indexing.summarize import SectionSummarizer

    text = "\n\n".join(f"Rule {i} says something specific here." for i in range(60))
    cfg = ChunkingConfig(summary_input_chars=500)
    llm = _FakeLLM("A combined summary.")
    SectionSummarizer(llm, cfg).summarise(Section(heading="Long", text=text))
    assert len(llm.prompts) > 2
    assert "PART SUMMARIES" in llm.prompts[-1]
    assert all(len(p) < 500 + 1500 for p in llm.prompts[:-1]), "a part overflowed"


def test_summaries_are_cached_by_text(tmp_path):
    from policyverify.indexing.summarize import SectionSummarizer, SummaryCache

    section = Section(heading="Attendance", text="Students must attend 75% of classes.")
    cfg = ChunkingConfig()
    path = tmp_path / "summaries.json"

    first = SectionSummarizer(_FakeLLM(), cfg, SummaryCache(path), model_name="m")
    first.summarise(section)
    first.cache.save()

    llm = _FakeLLM()
    second = SectionSummarizer(llm, cfg, SummaryCache(path), model_name="m")
    assert second.summarise(section) == "Students must attend 75% of classes."
    assert llm.prompts == [], "a cached summary must not call the model"

    # Edited text, or a different model, means a fresh summary.
    second.summarise(Section(heading="Attendance", text="Students must attend 80%."))
    assert len(llm.prompts) == 1


def test_unreachable_model_falls_back_to_the_opening_and_stops_retrying():
    from policyverify.indexing.summarize import SectionSummarizer

    _DeadLLM.calls = 0
    summarizer = SectionSummarizer(_DeadLLM(), ChunkingConfig(summary_max_chars=60))
    text = "Students must attend 75% of classes. Medical leave counts. More text follows here."
    first = summarizer.summarise(Section(heading="A", text=text))
    summarizer.summarise(Section(heading="B", text="Second section has its own words."))

    assert first == "Students must attend 75% of classes. Medical leave counts."
    assert _DeadLLM.calls == 1, "after one failure the build must not keep waiting"
    assert summarizer.stats["fallback"] == 2


def test_clean_summary_strips_labels_and_cuts_at_a_sentence():
    from policyverify.indexing.summarize import clean_summary

    raw = 'Summary:  "First sentence here.\n Second one is longer than the limit allows."'
    assert clean_summary(raw, 30) == "First sentence here."


def test_chunks_are_tagged_from_text_and_heading_not_from_the_summary():
    """A model-written summary must not decide who a rule is for."""
    doc = _doc(
        [Section(heading="Guest Accommodation", text="Rooms may be booked 7 days ahead.")],
        policy=PolicyType.RESIDENCE,
    )
    chunk = chunk_document(
        doc, summarize=lambda *_: "Parents may stay for up to three days."
    )[0]
    assert "visitor" in chunk.entities  # from the heading
    assert "parent" not in chunk.entities  # only the summary said it
    assert chunk.to_metadata()["entities"] == "|visitor|"
