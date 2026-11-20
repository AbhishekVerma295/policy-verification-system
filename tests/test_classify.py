"""
Tests for audience classification and the audience filter.

No model and no vector store: sections are classified by a fake LLM, and
retrieval runs against small stand-in stores. The questions are the real
residence questions from the adversarial set, because those are the ones
where who is asking changes which passage answers them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from policyverify.config import load_config
from policyverify.indexing.bm25 import BM25Index
from policyverify.indexing.chunk import chunk_document
from policyverify.indexing.classify import (
    ALL_AUDIENCES,
    AudienceCache,
    SectionClassifier,
    classify_query,
    matches_audience,
    parse_audience,
)
from policyverify.llm import LLMError
from policyverify.retrieve import query_audience, retrieve
from policyverify.schema import (
    Chunk,
    Document,
    PolicyType,
    RetrievedChunk,
    Section,
    SourceFormat,
)

# ---------------------------------------------------------------------------
# Classifying questions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question, expected",
    [
        # Real items from eval/adversarial.jsonl
        ("Can parents stay overnight in the hostel?", {"parent_visitor"}),
        ("How long may a parent stay in the hostel as a guest?", {"parent_visitor"}),
        ("Can visitors stay overnight in the hostel?", {"parent_visitor"}),
        # Asked by a student about their guests: both.
        ("How many overnight guests is each hosteller allowed per month?",
         {"student", "parent_visitor"}),
        ("Can day scholars stay in the hostel as guests?", {"student", "parent_visitor"}),
        # First person means a student is asking.
        ("Can my parents visit me on Sunday?", {"student", "parent_visitor"}),
        # No audience word at all: a student's question.
        ("What is the minimum attendance requirement?", {"student"}),
        ("Are hostel fees refundable?", {"student"}),
        ("What are the duties of the warden?", {"staff"}),
    ],
)
def test_classify_query(question, expected):
    assert classify_query(question) == expected


def test_a_question_is_never_classified_as_for_nobody():
    for question in ("", "?", "asdf qwer"):
        assert classify_query(question)


# ---------------------------------------------------------------------------
# Parsing the model's labels, and the failure mode
# ---------------------------------------------------------------------------


def test_parse_audience_reads_the_labels():
    assert parse_audience('{"audience": ["student", "parent_visitor"]}') == {
        "student",
        "parent_visitor",
    }
    assert parse_audience('Sure! {"audience": "Staff"}') == {"staff"}


def test_parse_audience_rejects_anything_unusable():
    for raw in ("", "student", '{"audience": []}', '{"audience": ["alien"]}',
                '{"labels": ["student"]}', "{not json}"):
        assert parse_audience(raw) is None


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, *, json_mode=True):
        assert json_mode, "audience labels are JSON"
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _section(text="Visitors may meet residents in the lobby until 6 pm."):
    return Section(heading="Visitors To The Hostels", text=text)


def test_every_failure_means_for_everyone_never_for_nobody():
    """A label can only HIDE a passage, so uncertainty must not hide one."""
    everyone = sorted(ALL_AUDIENCES)
    assert SectionClassifier(None).classify(_section()) == everyone
    assert SectionClassifier(_FakeLLM("no idea")).classify(_section()) == everyone
    assert SectionClassifier(_FakeLLM('{"audience": []}')).classify(_section()) == everyone


def test_an_unreachable_model_is_asked_once_per_build():
    llm = _FakeLLM(LLMError("connection refused"))
    classifier = SectionClassifier(llm)
    classifier.classify(_section("one"))
    classifier.classify(_section("two"))
    assert llm.calls == 1
    assert classifier.stats["fallback"] == 2


def test_labels_are_cached_and_bad_replies_are_not(tmp_path):
    path = tmp_path / "audiences.json"
    first = SectionClassifier(
        _FakeLLM('{"audience": ["student", "parent_visitor"]}'),
        AudienceCache(path),
        model_name="m",
    )
    assert first.classify(_section()) == ["parent_visitor", "student"]
    first.cache.save()

    llm = _FakeLLM('{"audience": ["staff"]}')
    second = SectionClassifier(llm, AudienceCache(path), model_name="m")
    assert second.classify(_section()) == ["parent_visitor", "student"]
    assert llm.calls == 0

    # An unparseable reply is not cached, so the next build asks again.
    third = SectionClassifier(_FakeLLM("garbage"), AudienceCache(None), model_name="m")
    third.classify(_section("other text"))
    assert len(third.cache) == 0


# ---------------------------------------------------------------------------
# Tagging chunks
# ---------------------------------------------------------------------------


def _doc(sections):
    from datetime import date

    return Document(
        doc_id="srm/residence",
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.RESIDENCE,
        title="Hostel Rules",
        source_url="https://example.srmist.edu.in/hostel",
        source_format=SourceFormat.HTML,
        retrieved_at=date(2026, 8, 19),
        checksum="a" * 64,
        sections=sections,
    )


def test_every_piece_of_a_section_carries_its_audience():
    long_text = "\n\n".join(f"Visitor rule {i} applies here." for i in range(60))
    doc = _doc([Section(heading="Visitors", text=long_text)])
    from policyverify.config import ChunkingConfig

    chunks = chunk_document(
        doc,
        ChunkingConfig(max_chars=300, overlap_chars=30),
        classify=lambda *_: ["parent_visitor", "student"],
    )
    assert len(chunks) > 1
    assert all(c.audience == ["parent_visitor", "student"] for c in chunks)


def test_audience_metadata_has_one_flag_per_label():
    chunk = _chunk("visitors", "x", ["parent_visitor"])
    meta = chunk.to_metadata()
    assert meta["aud_parent_visitor"] is True
    assert meta["aud_student"] is False and meta["aud_staff"] is False
    assert meta["audience"] == "|parent_visitor|"


def test_an_unclassified_chunk_passes_every_filter():
    meta = _chunk("old", "x", []).to_metadata()
    assert meta["aud_student"] and meta["aud_parent_visitor"] and meta["aud_staff"]
    assert matches_audience([], {"staff"})


# ---------------------------------------------------------------------------
# Filtering retrieval
# ---------------------------------------------------------------------------


def _chunk(name, text, audience):
    return Chunk(
        chunk_id=f"srm/residence/{name}#0",
        text=text,
        citation_id=f"srm/residence/{name}",
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.RESIDENCE,
        section_path=f"Hostel Rules > {name}",
        source_url="https://example.srmist.edu.in/hostel",
        audience=audience,
    )


_CORPUS = [
    _chunk("guest_rooms", "Guest rooms can be booked 7 days in advance.", ["parent_visitor"]),
    _chunk("visitors", "Visitors may meet residents in the lobby.",
           ["parent_visitor", "student"]),
    _chunk("curfew", "Residents must return to the hostel by 9 pm.", ["student"]),
    _chunk("warden_duties", "The warden shall inspect the rooms weekly.", ["staff"]),
]
_COSINE = {c.chunk_id: s for c, s in zip(_CORPUS, (0.70, 0.69, 0.68, 0.67), strict=True)}


class _FilteringStore:
    """Hybrid-capable stand-in that honours `audience` like the real store."""

    def __init__(self, chunks=_CORPUS, classified=True):
        self.chunks = chunks
        self.classified = classified
        self.audiences_seen = []
        self._bm25 = BM25Index(chunks)

    def read_manifest(self):
        return SimpleNamespace(entity_weights={}, audience_classified=self.classified)

    def embed_query(self, question):
        return [1.0]

    def search(self, question, k=None, university=None, policy_type=None,
               query_embedding=None, audience=None):
        self.audiences_seen.append(audience)
        pool = [c for c in self.chunks if not audience or matches_audience(c.audience, audience)]
        pool.sort(key=lambda c: -_COSINE[c.chunk_id])
        return [RetrievedChunk(chunk=c, score=_COSINE[c.chunk_id], rank=i)
                for i, c in enumerate(pool[:k])]

    def bm25_index(self):
        return self._bm25

    def similarities(self, query_embedding, chunk_ids):
        return {cid: _COSINE[cid] for cid in chunk_ids}


def _config(top_k=2):
    config = load_config()
    config.retrieval.top_k = top_k
    config.retrieval.oversample_factor = 1
    config.retrieval.hybrid_candidate_pool = 10
    config.retrieval.entity_boost = False
    return config


def test_a_students_question_does_not_pull_passages_for_visitors_or_staff():
    results = retrieve("Must residents be back by 9 pm?", config=_config(), store=_FilteringStore())
    ids = [r.chunk.citation_id for r in results]
    assert "srm/residence/guest_rooms" not in ids
    assert "srm/residence/warden_duties" not in ids
    assert set(ids) == {"srm/residence/visitors", "srm/residence/curfew"}


def test_a_parents_question_gets_the_passages_written_for_parents():
    results = retrieve(
        "Can parents book a guest room?", config=_config(), store=_FilteringStore()
    )
    assert {r.chunk.citation_id for r in results} == {
        "srm/residence/guest_rooms",
        "srm/residence/visitors",
    }


def test_the_keyword_side_is_filtered_too():
    hits = BM25Index(_CORPUS).search("warden inspect rooms", k=5, audience={"student"})
    assert "srm/residence/warden_duties#0" not in [c.chunk_id for c, _ in hits]


def test_a_too_narrow_filter_is_filled_from_the_unfiltered_search():
    """Fewer passages is a worse failure than one for another reader."""
    results = retrieve(
        "What are the warden's duties?", config=_config(top_k=3), store=_FilteringStore()
    )
    ids = [r.chunk.citation_id for r in results]
    assert ids[0] == "srm/residence/warden_duties", "filtered results come first"
    assert len(ids) == 3
    assert [r.rank for r in results] == [0, 1, 2]


def test_no_filter_on_an_index_built_without_audience_labels():
    """Filtering on aud_* fields an old index does not have would match nothing."""
    store = _FilteringStore(classified=False)
    retrieve("Can parents book a guest room?", config=_config(), store=store)
    assert store.audiences_seen == [None]
    assert query_audience("Can parents stay?", _config(), store) is None


def test_the_filter_can_be_switched_off():
    config = _config(top_k=4)
    config.retrieval.audience_filter = False
    store = _FilteringStore()
    results = retrieve("Must residents be back by 9 pm?", config=config, store=store)
    assert store.audiences_seen == [None]
    # Nothing is filtered: every passage, whoever it is for, can come back.
    assert len(results) == 4
