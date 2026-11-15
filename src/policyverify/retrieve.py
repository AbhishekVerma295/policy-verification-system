"""
retrieve.py - finding the passages that might answer a question.

Two behaviours live here rather than in the vector store: a cap on how many
chunks any single section may contribute, and hybrid search - merging the
embedding search with a keyword search.

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

WHY HYBRID SEARCH
    Embedding search alone is sensitive to wording here. All the chunks are
    university policy, so their similarity scores bunch into a narrow band
    (about 0.55-0.61) and a small rephrasing reorders them. Measured: for
    "What is minimum marks needed to pass a course ?" the section that
    literally says "To 'Pass' a course ... a minimum of 50%" ranked about 10th,
    so the answer was written from the attendance rule instead. A keyword
    search (BM25, see indexing/bm25.py) finds that section easily, because the
    words are right there; but keywords alone miss rewordings that share no
    words with the source. So we run both and merge.

HOW THE MERGE WORKS: RECIPROCAL RANK FUSION
    The two searches produce scores on unrelated scales (a cosine similarity
    around 0.6 versus a BM25 score around 10), so adding the scores would mean
    nothing. Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009)
    ignores the scores and uses only the RANKS: each search gives a passage

        1 / (rrf_k + rank)          (rank 1 = best; rrf_k = 60 by default)

    and the contributions are added. A passage both searches like beats one
    that only a single search ranks first, which is the behaviour we want:
    agreement between two different kinds of evidence is the strongest signal
    we have. It needs no tuning beyond rrf_k, and rrf_k = 60 is the paper's
    value.

WHAT `score` MEANS AFTER THE MERGE
    Unchanged: the cosine similarity of the passage to the question. The
    fused number is used only to decide the ORDER. This matters because
    abstain.py compares the best `score` with `min_retrieval_score`, and the
    app shows it as a similarity - neither would make sense of an RRF number
    like 0.03. A passage only the keyword search found gets its similarity
    computed from its stored embedding (VectorStore.similarities). `rank` is
    the position in the final, merged order, so results may come back with
    a lower-scoring passage above a higher-scoring one. That is intended.
"""

from __future__ import annotations

from collections.abc import Sequence

from policyverify.config import Config, get_config
from policyverify.indexing import VectorStore
from policyverify.schema import PolicyType, RetrievedChunk


def diversify(
    candidates: list[RetrievedChunk], k: int, max_per_citation: int
) -> list[RetrievedChunk]:
    """Take the first `k` of a best-first list, allowing at most
    `max_per_citation` from any section.

    Two passes. The first respects the cap strictly; if that leaves fewer than
    k results (a small corpus, or a narrow filter), the second pass fills the
    remaining slots from what was skipped rather than returning short. Fewer
    passages is a worse failure than a slightly repetitive set.

    The input ORDER is what counts as "best first", not `score`. For the
    embedding-only search the two are the same thing. For hybrid search they
    are not: the order comes from the merged ranking while `score` stays the
    cosine similarity (see the top of this module), so re-sorting by score
    here would silently undo the merge.
    """
    if max_per_citation <= 0:
        kept_positions = list(range(min(k, len(candidates))))
    else:
        kept_positions = []
        overflow: list[int] = []
        counts: dict[str, int] = {}

        for position, candidate in enumerate(candidates):
            citation = candidate.chunk.citation_id
            if counts.get(citation, 0) < max_per_citation:
                counts[citation] = counts.get(citation, 0) + 1
                kept_positions.append(position)
            else:
                overflow.append(position)

        if len(kept_positions) < k:
            kept_positions.extend(overflow[: k - len(kept_positions)])

        # Put any overflow fill-ins back where they ranked in the input,
        # rather than tacking them onto the end.
        kept_positions.sort()

    # Re-rank so `rank` reflects the final order the caller sees, not the
    # order the store happened to return.
    final = [candidates[p] for p in kept_positions[:k]]
    return [
        RetrievedChunk(chunk=r.chunk, score=r.score, rank=i)
        for i, r in enumerate(final)
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], rrf_k: int = 60
) -> list[str]:
    """Merge several best-first lists of ids into one, by RRF.

    Each list gives an id 1 / (rrf_k + rank), with rank counted from 1; the
    contributions are summed and the result sorted highest first. An id
    missing from a list simply gets nothing from it.

    Ties are broken deterministically: first by the best rank the id reached
    in any list (a first place beats two middling ones), then by the id
    itself. Without that, equal sums would come out in whatever order a dict
    happened to produce, and the answer could change between runs.
    """
    fused: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue  # a duplicate within one list counts once
            seen.add(item)
            fused[item] = fused.get(item, 0.0) + 1.0 / (rrf_k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(fused, key=lambda item: (-fused[item], best_rank[item], item))


def _supports_hybrid(store) -> bool:
    """Does this store have everything hybrid search needs?

    Tests (and anything else) may hand retrieve() a stand-in store that only
    implements `search()`. Those keep working, on the embedding-only path,
    rather than failing on a missing method.
    """
    return all(
        callable(getattr(store, name, None))
        for name in ("embed_query", "bm25_index", "similarities")
    )


def _hybrid_candidates(
    question: str,
    store: VectorStore,
    pool: int,
    rrf_k: int,
    university: str | None,
    policy_type: PolicyType | str | None,
) -> list[RetrievedChunk]:
    """Both searches, merged by RRF, best first, every `score` a cosine."""
    query_vector = store.embed_query(question)

    dense = store.search(
        question,
        k=pool,
        university=university,
        policy_type=policy_type,
        query_embedding=query_vector,
    )
    # The same filters, applied the same way: they restrict which chunks are
    # eligible, so the keyword side can never bring back a passage from a
    # university or policy the embedding side was told to exclude.
    keyword = store.bm25_index().search(
        question, k=pool, university=university, policy_type=policy_type
    )

    by_id = {r.chunk.chunk_id: r for r in dense}
    keyword_chunks = {chunk.chunk_id: chunk for chunk, _ in keyword}

    order = reciprocal_rank_fusion(
        [[r.chunk.chunk_id for r in dense], [chunk.chunk_id for chunk, _ in keyword]],
        rrf_k=rrf_k,
    )

    # Chunks only the keyword side found have no similarity yet. Look them up
    # from their stored embeddings so `score` means the same for every result.
    missing = [cid for cid in order if cid not in by_id]
    similarity = store.similarities(query_vector, missing)

    out: list[RetrievedChunk] = []
    for position, cid in enumerate(order):
        if cid in by_id:
            score = by_id[cid].score
            chunk = by_id[cid].chunk
        else:
            score = similarity.get(cid, 0.0)
            chunk = keyword_chunks[cid]
        out.append(RetrievedChunk(chunk=chunk, score=score, rank=position))
    return out


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
    cfg = config.retrieval
    top_k = k or cfg.top_k

    # Over-fetch, then cap. Asking the store for exactly k and then filtering
    # would leave gaps, since the discarded chunks have nothing to replace
    # them with.
    oversampled = max(top_k * cfg.oversample_factor, top_k)

    if cfg.hybrid_search and _supports_hybrid(store):
        # Never pull fewer candidates than the embedding-only path would.
        pool = max(cfg.hybrid_candidate_pool, oversampled)
        candidates = _hybrid_candidates(
            question, store, pool, cfg.rrf_k, university, policy_type
        )
    else:
        candidates = store.search(
            question,
            k=oversampled,
            university=university,
            policy_type=policy_type,
        )
    return diversify(candidates, top_k, cfg.max_chunks_per_citation)
