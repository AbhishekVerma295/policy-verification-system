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
from policyverify.entities import entity_boost, extract_entities
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
    # order the store happened to return. rank_score is similarity plus any
    # entity boost - see rerank_by_entities().
    final = sorted(kept, key=lambda r: -r.rank_score)[:k]
    # model_copy rather than a fresh RetrievedChunk(...): it changes only the
    # rank and does not re-validate the chunk, which is already valid.
    return [r.model_copy(update={"rank": i}) for i, r in enumerate(final)]


def rerank_by_entities(
    question: str,
    candidates: list[RetrievedChunk],
    weights: dict[str, float],
    weight: float,
    cap: float,
) -> list[RetrievedChunk]:
    """Add a capped boost to candidates that name the same people as the
    question, then order by similarity + boost.

    Runs on the oversampled candidate list, before the per-section cap, so a
    passage just outside the top k can be promoted into it. `score` is left
    as the raw similarity. See entities.py for why.
    """
    query_entities = extract_entities(question)
    if not query_entities or not weights:
        return candidates
    out = []
    for c in candidates:
        chunk_entities = set(c.chunk.entities)
        boost = entity_boost(query_entities, chunk_entities, weights, weight, cap)
        out.append(
            c.model_copy(
                update={
                    "boost": boost,
                    "matched_entities": sorted(query_entities & chunk_entities),
                }
            )
        )
    return sorted(out, key=lambda r: -r.rank_score)


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
    rcfg = config.retrieval
    if rcfg.entity_boost:
        manifest = store.read_manifest()
        candidates = rerank_by_entities(
            question,
            candidates,
            weights=manifest.entity_weights if manifest else {},
            weight=rcfg.entity_boost_weight,
            cap=rcfg.entity_boost_cap,
        )
    return diversify(candidates, top_k, rcfg.max_chunks_per_citation)
