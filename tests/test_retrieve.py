"""
Tests for retrieval, focused on the per-citation diversity cap.

No vector store and no model here - diversify() is a pure function over
already-scored results, which is exactly the part worth testing. Whether the
embeddings are any good is a question for the evaluation harness, not a unit
test.
"""

from __future__ import annotations

from policyverify.retrieve import diversify
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
