"""
Tests for retrieval: the per-citation diversity cap, and hybrid search.

No vector store and no model here - diversify() is a pure function over
already-scored results, and the hybrid tests use a stand-in store whose
"embedding similarities" are written by hand next to a real BM25 index. That
is exactly the part worth testing: how the two rankings are merged, capped
and scored. Whether the embeddings are any good is a question for the
evaluation harness, not a unit test.
"""

from __future__ import annotations

import pytest

from policyverify.config import load_config
from policyverify.indexing.bm25 import BM25Index
from policyverify.retrieve import diversify, retrieve
from policyverify.schema import Chunk, PolicyType, RetrievedChunk

A = "srm/attendance/4.2-first_section"
B = "srm/attendance/4.3-second_section"


def _chunk(citation: str, piece: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{citation}#{piece}",
        text=f"Text of {citation} piece {piece}.",
        citation_id=citation,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Policy > Section",
        source_url="https://example.srmist.edu.in/p",
    )


def _results(spec: list[tuple[str, float]]) -> list[RetrievedChunk]:
    """Build scored results from (citation_id, score) pairs, best first."""
    out = []
    counts: dict[str, int] = {}
    for rank, (citation, score) in enumerate(spec):
        piece = counts.get(citation, 0)
        counts[citation] = piece + 1
        out.append(RetrievedChunk(chunk=_chunk(citation, piece), score=score, rank=rank))
    return out


def test_diversify_caps_chunks_from_one_section():
    """The real failure: one oversized section split into many chunks took
    every top slot and pushed the actually-relevant rule out of the results."""
    candidates = _results(
        [
            ("srm/examination/detention", 0.63),
            ("srm/examination/detention", 0.62),
            ("srm/examination/detention", 0.61),
            ("srm/attendance/7.3-minimum_attendance", 0.60),
            ("srm/attendance/7.4-shortage", 0.59),
        ]
    )
    kept = diversify(candidates, k=3, max_per_citation=2)

    citations = [r.chunk.citation_id for r in kept]
    assert citations.count("srm/examination/detention") == 2
    assert "srm/attendance/7.3-minimum_attendance" in citations


def test_diversify_keeps_the_highest_scoring_pieces():
    candidates = _results(
        [(A, 0.9), (A, 0.8), (A, 0.7), (B, 0.6)]
    )
    kept = diversify(candidates, k=3, max_per_citation=2)
    scores = [r.score for r in kept]
    assert 0.7 not in scores, "should have dropped the weakest piece of 'a'"
    assert scores == sorted(scores, reverse=True)


def test_diversify_fills_from_overflow_rather_than_returning_short():
    """A narrow corpus may not have k distinct sections. Returning fewer
    passages is worse than returning a slightly repetitive set - the answer
    would simply have less to stand on."""
    candidates = _results([(A, 0.9), (A, 0.8), (A, 0.7), (A, 0.6)])
    kept = diversify(candidates, k=3, max_per_citation=1)
    assert len(kept) == 3


def test_diversify_reassigns_rank_to_final_order():
    candidates = _results([(A, 0.9), (A, 0.8), (B, 0.7)])
    kept = diversify(candidates, k=3, max_per_citation=1)
    assert [r.rank for r in kept] == list(range(len(kept)))


def test_diversify_disabled_when_cap_is_zero():
    candidates = _results([(A, 0.9), (A, 0.8), (A, 0.7)])
    kept = diversify(candidates, k=2, max_per_citation=0)
    assert len(kept) == 2
    assert all(r.chunk.citation_id == A for r in kept)


def test_diversify_handles_empty_input():
    assert diversify([], k=5, max_per_citation=2) == []


def test_diversify_never_returns_more_than_k():
    candidates = _results([(f"srm/attendance/s{i}", 0.9 - i / 100) for i in range(20)])
    assert len(diversify(candidates, k=6, max_per_citation=2)) == 6


def test_diversify_preserves_input_order_not_score_order():
    """Hybrid search orders by the merged ranking while `score` stays the
    cosine similarity, so a lower-scoring passage may rightly come first.
    Re-sorting by score here would silently undo the merge."""
    candidates = _results([(A, 0.55), (B, 0.60), ("srm/attendance/c", 0.58)])
    kept = diversify(candidates, k=3, max_per_citation=2)
    assert [r.score for r in kept] == [0.55, 0.60, 0.58]


def test_diversify_cap_still_applies_to_a_fused_order():
    candidates = _results([(A, 0.5), (A, 0.9), (A, 0.8), (B, 0.4)])
    kept = diversify(candidates, k=3, max_per_citation=2)
    assert [r.chunk.citation_id for r in kept] == [A, A, B]
    assert [r.score for r in kept] == [0.5, 0.9, 0.4]


def test_diversify_overflow_fill_keeps_its_original_position():
    """A fill-in from overflow goes back where it ranked, not to the end."""
    candidates = _results([(A, 0.9), (A, 0.8), (B, 0.7)])
    kept = diversify(candidates, k=3, max_per_citation=1)
    assert [r.score for r in kept] == [0.9, 0.8, 0.7]


# ---------------------------------------------------------------------------
# Hybrid retrieval (embedding search + BM25, merged by RRF)
# ---------------------------------------------------------------------------


def _policy_chunk(name: str, text: str, university="srm",
                  policy=PolicyType.ATTENDANCE) -> Chunk:
    return Chunk(
        chunk_id=f"{university}/{policy.value}/{name}#0",
        text=text,
        citation_id=f"{university}/{policy.value}/{name}",
        university=university,
        university_name="Some University",
        policy_type=policy,
        section_path=f"Policy > {name}",
        source_url="https://example.edu/p",
    )


class _HybridStore:
    """A stand-in store with everything hybrid search needs, and no model.

    `cosine` is the similarity each chunk "has" to every question; the
    embedding search returns chunks in that order. The keyword side is the
    real BM25Index over the same chunks.
    """

    def __init__(self, chunks: list[Chunk], cosine: dict[str, float]):
        self.chunks = chunks
        self.cosine = cosine
        self.calls: dict[str, list] = {"search": [], "similarities": []}
        self._bm25 = BM25Index(chunks)

    @staticmethod
    def _allowed(chunk, university, policy_type):
        value = getattr(policy_type, "value", policy_type)
        return (not university or chunk.university == university) and (
            not value or chunk.policy_type.value == value
        )

    def embed_query(self, question):
        return [1.0]

    def search(self, question, k=None, university=None, policy_type=None,
               query_embedding=None):
        self.calls["search"].append(query_embedding)
        pool = [c for c in self.chunks if self._allowed(c, university, policy_type)]
        pool.sort(key=lambda c: -self.cosine[c.chunk_id])
        return [
            RetrievedChunk(chunk=c, score=self.cosine[c.chunk_id], rank=i)
            for i, c in enumerate(pool[:k])
        ]

    def bm25_index(self):
        return self._bm25

    def similarities(self, query_embedding, chunk_ids):
        self.calls["similarities"].append(list(chunk_ids))
        return {cid: self.cosine[cid] for cid in chunk_ids}


def _config(hybrid=True, pool=2, top_k=2, cap=2):
    config = load_config()
    config.retrieval.hybrid_search = hybrid
    config.retrieval.hybrid_candidate_pool = pool
    config.retrieval.top_k = top_k
    config.retrieval.oversample_factor = 1
    config.retrieval.max_chunks_per_citation = cap
    return config


# The marks failure in miniature: the pass rule shares the question's words
# but is only the embedding search's 4th choice, below the pool cut-off.
_CHUNKS = [
    _policy_chunk("attendance", "Students must attend 75% of classes."),
    _policy_chunk("degree", "Award of degree requires completing all credits."),
    _policy_chunk("diploma", "Eligibility for award of the diploma."),
    _policy_chunk("grades", "To pass a course a minimum of 50% marks is needed."),
]
_COSINE = {
    "srm/attendance/attendance#0": 0.61,
    "srm/attendance/degree#0": 0.60,
    "srm/attendance/diploma#0": 0.59,
    "srm/attendance/grades#0": 0.56,
}
_QUESTION = "What is minimum marks needed to pass a course ?"


def test_hybrid_rescues_a_passage_the_embedding_search_missed():
    store = _HybridStore(_CHUNKS, _COSINE)
    results = retrieve(_QUESTION, config=_config(), store=store)
    ids = [r.chunk.chunk_id for r in results]
    assert "srm/attendance/grades#0" in ids


def test_hybrid_off_is_the_embedding_search_exactly():
    store = _HybridStore(_CHUNKS, _COSINE)
    results = retrieve(_QUESTION, config=_config(hybrid=False), store=store)
    assert [r.chunk.chunk_id for r in results] == [
        "srm/attendance/attendance#0",
        "srm/attendance/degree#0",
    ]
    assert store.calls["similarities"] == []


def test_score_stays_the_cosine_similarity_for_keyword_only_finds():
    """abstain.py compares `score` with min_retrieval_score and the app shows
    it as a similarity, so a chunk only BM25 found must still carry its real
    cosine - looked up, not invented from the fused number."""
    store = _HybridStore(_CHUNKS, _COSINE)
    # pool=2 and k=2: the embedding side only sees attendance and degree,
    # so the pass rule can only arrive through the keyword side.
    results = retrieve(_QUESTION, k=2, config=_config(pool=2), store=store)
    assert "srm/attendance/grades#0" in [r.chunk.chunk_id for r in results]
    for r in results:
        assert r.score == _COSINE[r.chunk.chunk_id]
    # the keyword-only chunk was looked up, and only keyword-only ones were
    looked_up = [cid for call in store.calls["similarities"] for cid in call]
    assert "srm/attendance/grades#0" in looked_up
    assert "srm/attendance/attendance#0" not in looked_up


def test_hybrid_rank_follows_the_fused_order():
    store = _HybridStore(_CHUNKS, _COSINE)
    results = retrieve(_QUESTION, k=4, config=_config(pool=4), store=store)
    assert [r.rank for r in results] == list(range(len(results)))
    # The pass rule is top for keywords and present for embeddings, so it
    # wins the merge despite having the lowest cosine of the four.
    assert results[0].chunk.chunk_id == "srm/attendance/grades#0"
    assert results[0].score == min(r.score for r in results)


def test_hybrid_embeds_the_question_once_and_reuses_it():
    store = _HybridStore(_CHUNKS, _COSINE)
    retrieve(_QUESTION, config=_config(), store=store)
    assert store.calls["search"] == [[1.0]]


def test_hybrid_respects_the_per_section_cap():
    """Three pieces of one section, all strong keyword matches, must still
    leave room for a different section."""
    grades = _policy_chunk("grades", "pass a course with 50% marks")
    pieces = [grades.model_copy(update={"chunk_id": f"srm/attendance/grades#{i}"})
              for i in range(3)]
    chunks = [*pieces, _policy_chunk("other", "unrelated text about hostels")]
    cosine = {c.chunk_id: 0.5 for c in chunks}
    store = _HybridStore(chunks, cosine)
    results = retrieve("pass marks", k=3, config=_config(pool=10, cap=2), store=store)
    citations = [r.chunk.citation_id for r in results]
    assert citations.count("srm/attendance/grades") == 2
    assert "srm/attendance/other" in citations


@pytest.mark.parametrize(
    "filters",
    [{"university": "uni_b"}, {"policy_type": PolicyType.EXAMINATION},
     {"policy_type": "examination"}],
)
def test_filters_restrict_the_keyword_side_too(filters):
    """A keyword match from the wrong university is exactly the mixed-up
    answer the filter exists to prevent."""
    chunks = [
        *_CHUNKS,
        _policy_chunk("pass_b", "Pass a course with 40% marks.", university="uni_b"),
        _policy_chunk("pass_exam", "Pass a course exam with 45% marks.",
                      policy=PolicyType.EXAMINATION),
    ]
    cosine = {**_COSINE, chunks[-2].chunk_id: 0.3, chunks[-1].chunk_id: 0.3}
    store = _HybridStore(chunks, cosine)
    results = retrieve(_QUESTION, k=10, config=_config(pool=10), store=store, **filters)
    assert results
    for r in results:
        if "university" in filters:
            assert r.chunk.university == filters["university"]
        else:
            assert r.chunk.policy_type is PolicyType.EXAMINATION


class _SearchOnlyStore:
    """Like the stubs in test_pipeline.py: only search(), no keyword index."""

    def search(self, question, k=None, university=None, policy_type=None):
        return _results([(A, 0.9), (B, 0.8)])


def test_falls_back_to_embedding_search_for_a_search_only_store():
    results = retrieve("q", config=_config(hybrid=True), store=_SearchOnlyStore())
    assert [r.chunk.citation_id for r in results] == [A, B]
    assert [r.score for r in results] == [0.9, 0.8]
