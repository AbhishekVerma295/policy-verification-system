"""
Tests for retrieval, focused on the per-citation diversity cap.

No vector store and no model here - diversify() is a pure function over
already-scored results, which is exactly the part worth testing. Whether the
embeddings are any good is a question for the evaluation harness, not a unit
test.
"""

from __future__ import annotations

import pytest

from policyverify.entities import entity_boost, entity_weights, extract_entities
from policyverify.retrieve import diversify, rerank_by_entities
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


# ---------------------------------------------------------------------------
# Entity extraction and re-ranking
# ---------------------------------------------------------------------------


def test_extracts_who_a_question_is_for():
    assert "parent" in extract_entities("How long may a parent stay in the hostel?")
    assert "parent" in extract_entities("Can my local guardian visit?")
    assert "day_scholar" in extract_entities("Can day scholars stay in the hostel as guests?")
    assert "hosteller" in extract_entities("How many guests is each hosteller allowed?")
    assert extract_entities("Are male visitors allowed in female hostels?") >= {
        "male",
        "female",
        "visitor",
    }


def test_extraction_avoids_the_traps_found_in_the_corpus():
    # "female" contains "male", "women" contains "men".
    assert "male" not in extract_entities("Female students only.")
    assert "male" not in extract_entities("Rules for women.")
    # Examination policy: "Chief Guest" and "guest faculty" are not visitors.
    assert "visitor" not in extract_entities("Greet the Chief Guest on arrival.")
    assert "visitor" not in extract_entities("Guest faculty may invigilate.")
    # UG/PG are only abbreviations in capitals.
    assert "pg" in extract_entities("For PG, the scribe must be qualified.")
    assert "pg" not in extract_entities("see pg 12")


def test_weights_make_common_entities_worth_almost_nothing():
    """Measured: "student" appears in most SRM chunks, "parent" in few."""
    chunks = [{"student"}] * 18 + [{"student", "parent"}] + [{"warden"}]
    w = entity_weights(chunks)
    assert w["student"] < 0.1
    assert w["parent"] == pytest.approx(1.0)
    assert w["parent"] > w["student"]


def test_boost_is_capped_and_uses_the_best_single_match():
    weights = {"parent": 1.0, "visitor": 0.8, "student": 0.02}
    assert entity_boost({"parent"}, {"parent"}, weights, weight=0.1, cap=0.05) == 0.05
    assert entity_boost({"student"}, {"student"}, weights, weight=0.1, cap=0.05) == 0.002
    # Matching more entities does not add up past the best one.
    both = entity_boost({"parent", "visitor"}, {"parent", "visitor"}, weights, 0.02, 1.0)
    assert both == 0.02
    assert entity_boost({"parent"}, {"warden"}, weights, 0.1, 0.05) == 0.0


def _tagged(citation: str, score: float, entities: list[str]) -> RetrievedChunk:
    chunk = _chunk(citation).model_copy(update={"entities": entities})
    return RetrievedChunk(chunk=chunk, score=score, rank=0)


def test_rerank_promotes_a_near_tie_that_names_the_right_people():
    weights = {"parent": 1.0, "hosteller": 0.6}
    candidates = [
        _tagged(A, 0.71, ["hosteller"]),  # topic match, wrong people
        _tagged(B, 0.69, ["parent"]),  # what the question asked about
    ]
    out = rerank_by_entities(
        "How long may a parent stay in the hostel?", candidates, weights, 0.1, 0.05
    )
    assert out[0].chunk.citation_id == B
    assert out[0].boost == 0.05 and out[0].matched_entities == ["parent"]
    # The similarity itself is untouched: abstention still reads it.
    assert out[0].score == 0.69


def test_rerank_cap_cannot_promote_a_clearly_worse_passage():
    weights = {"parent": 1.0}
    candidates = [_tagged(A, 0.80, []), _tagged(B, 0.60, ["parent"])]
    out = rerank_by_entities("Can parents visit?", candidates, weights, 0.1, 0.05)
    assert out[0].chunk.citation_id == A


def test_rerank_is_a_no_op_without_entities_or_weights():
    candidates = [_tagged(A, 0.7, ["parent"]), _tagged(B, 0.6, [])]
    assert rerank_by_entities("What is the fee?", candidates, {"parent": 1.0}, 0.1, 0.05) == candidates
    assert rerank_by_entities("Can parents visit?", candidates, {}, 0.1, 0.05) == candidates


def test_diversify_orders_by_score_plus_boost():
    boosted = _tagged(B, 0.69, ["parent"]).model_copy(update={"boost": 0.05})
    out = diversify([_tagged(A, 0.71, []), boosted], k=2, max_per_citation=2)
    assert [r.chunk.citation_id for r in out] == [B, A]
