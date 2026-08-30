"""
retrieve.py - finding the passages that might answer a question.

Thin layer over the vector store, with one behaviour the store deliberately
does not have: a cap on how many chunks any single section may contribute.

WHY THE CAP EXISTS
    Chunking splits oversized sections into several pieces (see
    indexing/chunk.py). Those pieces are near-identical in topic, so they
    score near-identically, so they arrive as a block and crowd everything
    else out of the results.

    Measured on the real corpus: asking "what is the minimum attendance
    requirement to sit the final examination?" returned three chunks of ONE
    examination section in the top 3, pushing the actual attendance rule
    (srm/attendance/7.3-minimum_attendance) down to rank 4. The answer would
    then be written without ever seeing the rule it was asked about.

    Capping per citation trades a little relevance for coverage. For a
    question-answering system that has to cite its sources, coverage is worth
    more: three chunks saying the same thing support no more claims than one.
"""

from __future__ import annotations

from policyverify.config import Config, get_config
from policyverify.indexing import VectorStore
from policyverify.schema import PolicyType, RetrievedChunk


def diversify(
    candidates: list[RetrievedChunk], k: int, max_per_citation: int
) -> list[RetrievedChunk]:
    """Take the best `k`, allowing at most `max_per_citation` from any section.

    Two passes. The first respects the cap strictly; if that leaves fewer than
    k results (a small corpus, or a narrow filter), the second pass fills the
    remaining slots from what was skipped rather than returning short. Fewer
    passages is a worse failure than a slightly repetitive set.
    """
    if max_per_citation <= 0:
        return candidates[:k]

    kept: list[RetrievedChunk] = []
    overflow: list[RetrievedChunk] = []
    counts: dict[str, int] = {}

    for candidate in candidates:
        citation = candidate.chunk.citation_id
        if counts.get(citation, 0) < max_per_citation:
            counts[citation] = counts.get(citation, 0) + 1
            kept.append(candidate)
        else:
            overflow.append(candidate)

    if len(kept) < k:
        kept.extend(overflow[: k - len(kept)])

    # Re-rank so `rank` reflects the final order the caller sees, not the
    # order the store happened to return.
    final = sorted(kept, key=lambda r: -r.score)[:k]
    return [
        RetrievedChunk(chunk=r.chunk, score=r.score, rank=i)
        for i, r in enumerate(final)
    ]


def retrieve(
    question: str,
    k: int | None = None,
    university: str | None = None,
    policy_type: PolicyType | str | None = None,
    config: Config | None = None,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Find the passages most likely to answer `question`."""
    config = config or get_config()
    store = store or VectorStore(config)
    top_k = k or config.retrieval.top_k

    # Over-fetch, then cap. Asking the store for exactly k and then filtering
    # would leave gaps, since the discarded chunks have nothing to replace
    # them with.
    oversampled = max(top_k * config.retrieval.oversample_factor, top_k)
    candidates = store.search(
        question,
        k=oversampled,
        university=university,
        policy_type=policy_type,
    )
    return diversify(candidates, top_k, config.retrieval.max_chunks_per_citation)
