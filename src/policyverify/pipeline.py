"""
pipeline.py - the whole system, end to end, in one function.

    question -> retrieve -> generate -> verify -> relevance gate -> decide -> Answer

This is the only entry point anything outside the library should need. The
CLI, the Streamlit UI and the evaluation harness all call `answer_question()`,
which is what guarantees they cannot drift apart: a result seen in the UI is
the result the harness measured, because it came from the same code.

Everything expensive is injectable. The store, the language model and the NLI
checker can all be passed in, which lets callers load a model once and reuse
it across many questions - and lets tests substitute fakes and run the whole
pipeline in milliseconds with nothing downloaded.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from policyverify.abstain import decide
from policyverify.compose import compose_answer
from policyverify.config import Config, get_config
from policyverify.generate import generate_answer_draft
from policyverify.indexing import VectorStore
from policyverify.listing import find_list, list_topic
from policyverify.llm import LLMBackend, get_llm
from policyverify.retrieve import retrieve
from policyverify.schema import (
    Answer,
    ClaimVerdict,
    DraftAnswer,
    PolicyType,
    RetrievedChunk,
    Timings,
)
from policyverify.situation import apply_to_situation, mask_user_figures
from policyverify.verify import verify_claims
from policyverify.verify.relevance import (
    RelevanceReport,
    assess_relevance,
    cross_encoder_for,
    similarity_for_store,
)

# The relevance gate as the pipeline sees it: (question, verdicts) -> report.
# A plain callable so tests can pass a one-line fake.
RelevanceFn = Callable[[str, list[ClaimVerdict]], RelevanceReport]


class _Stopwatch:
    """Wall-clock timing for one stage, in milliseconds."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


def _default_relevance(store, llm: LLMBackend | None, config: Config) -> RelevanceFn:
    """The real relevance gate: similarity from the store's embedder, the
    judge from the same language model that wrote the claims."""
    similarity = similarity_for_store(store)
    judge = llm or get_llm(config)
    cfg = config.relevance
    # Like the similarity, the cross-encoder is a model that needs the real
    # environment: a stand-in store with no embedder (the tests) keeps the
    # original gate, so nothing here downloads or loads a model for it.
    cross_encoder = (
        cross_encoder_for(cfg.cross_encoder_model, cfg.cross_encoder_device)
        if cfg.judge == "cross_encoder" and similarity is not None
        else None
    )

    own_names: list[str] = []
    if cross_encoder is not None and callable(getattr(store, "all_chunks", None)):
        # The institution's own names, so "SRM ... will use the best software"
        # is not mistaken for naming the software (relevance.names_something_new).
        names = {c.university_name for c in store.all_chunks()}
        names |= {c.university for c in store.all_chunks()}
        own_names = sorted(names)

    def gate(question: str, verdicts: list[ClaimVerdict]) -> RelevanceReport:
        return assess_relevance(
            question, verdicts, llm=judge, similarity=similarity, config=config,
            cross_encoder=cross_encoder, own_names=own_names,
        )

    return gate


def answer_question(
    question: str,
    k: int | None = None,
    university: str | None = None,
    policy_type: PolicyType | str | None = None,
    config: Config | None = None,
    store=None,
    llm: LLMBackend | None = None,
    nli=None,
    on_stage: Callable[[str], None] | None = None,
    relevance: RelevanceFn | None = None,
) -> Answer:
    """Answer one policy question, with every claim verified.

    Returns an `Answer` whether or not the system decided to answer. An
    abstention is a result, not an error: it carries the reason and the claims
    that were considered and rejected.

    `on_stage`, if given, is called with "retrieve", "generate", "verify" and
    "decide" as each stage starts - so an interface can show where a slow
    question has got to. It only observes; it cannot change the result.

    `relevance`, if given, replaces the relevance gate (see
    verify/relevance.py): a callable taking the question and the verified
    claims and returning a `RelevanceReport`. Tests pass a fake; normally it
    is built from the store's embedder and the language model.
    """
    config = config or get_config()
    timings = Timings()
    # One store for retrieval, verification and the relevance gate. Left to
    # themselves each would build its own and reload the embedding model.
    store = store or VectorStore(config)

    def stage(name: str) -> None:
        if on_stage is not None:
            on_stage(name)

    # --- 1. retrieve -----------------------------------------------------
    stage("retrieve")
    watch = _Stopwatch()
    chunks: list[RetrievedChunk] = retrieve(
        question,
        k=k,
        university=university,
        policy_type=policy_type,
        config=config,
        store=store,
    )
    timings.retrieve_ms = watch.stop()

    # --- 1b. a list question is answered from the document's list --------
    # "What are the types of plagiarism?" needs all ten numbered sections, not
    # the four most similar passages (see listing.py). The list's sections
    # become the passages, and its items the claims, quoted from the text.
    listing = None
    all_chunks = getattr(store, "all_chunks", None)
    if chunks and callable(all_chunks) and list_topic(question):
        document = chunks[0].chunk.citation_id.rsplit("/", 1)[0]
        listing = find_list(question, all_chunks(), document)
        if listing is not None:
            cited = {c.citation_ids[0] for c in listing.claims}
            order = [c.citation_ids[0] for c in listing.claims]
            pieces = sorted(
                (c for c in all_chunks() if c.citation_id in cited),
                key=lambda c: (order.index(c.citation_id), c.chunk_id),
            )
            chunks = [
                RetrievedChunk(chunk=c, score=1.0, rank=i) for i, c in enumerate(pieces)
            ]

    # --- 2. generate -----------------------------------------------------
    stage("generate")
    watch = _Stopwatch()
    if listing is not None:
        draft, fabricated = DraftAnswer(claims=listing.claims), []
    else:
        draft, fabricated = generate_answer_draft(question, chunks, llm=llm, config=config)
    timings.generate_ms = watch.stop()

    # --- 3. verify -------------------------------------------------------
    # Two questions are asked of every claim: is it TRUE of the passage it
    # cites (the NLI checks), and does it ANSWER the question (the relevance
    # gate). The gate runs here, inside the verify stage, and its time is
    # recorded both on its own and as part of verify_ms.
    stage("verify")
    watch = _Stopwatch()
    verdicts = verify_claims(draft.claims, chunks, nli=nli, store=store, config=config)
    gate_watch = _Stopwatch()
    if listing is not None:
        # Each item is an entry of the very list the question asked for, so
        # it answers the question by construction; only its truth is checked.
        verdicts = [
            v.model_copy(update={"checks": v.checks.model_copy(
                update={"relevant": True, "relevance_source": "list"})})
            for v in verdicts
        ]
        report = RelevanceReport(verdicts=verdicts)
    else:
        if relevance is None:
            relevance = _default_relevance(store, llm, config)
        report = relevance(question, verdicts)
    verdicts = report.verdicts
    timings.relevance_ms = gate_watch.stop()
    timings.verify_ms = watch.stop()

    # --- 4. decide -------------------------------------------------------
    stage("decide")
    decision = decide(verdicts, retrieved=chunks, config=config)

    answer = Answer(
        question=question,
        list_intro=listing.intro if listing is not None and not decision.abstained else None,
        university_filter=university,
        relevance_judge_failed=report.judge_failed,
        claims_kept=decision.kept,
        claims_removed=decision.removed,
        abstained=decision.abstained,
        reason=decision.reason,
        timings=timings,
        config_fingerprint=config.fingerprint(),
    )

    # The asker's figure against a verified threshold, worked out in code.
    # Only from KEPT claims: a rule that failed verification decides nothing.
    if not answer.abstained:
        note = apply_to_situation(
            question, answer.claims_kept,
            passages=[(r.chunk.citation_id, r.chunk.text) for r in chunks],
        )
        if note is not None:
            answer.situation = note.message
            answer.situation_citation = note.citation

    # A written answer from the kept claims, every sentence checked against
    # them (compose.py). Not for a list answer: the list is the answer.
    # Not for a question about the asker's OWN attendance either: there the
    # answer turns on one comparison, which situation.py makes exactly, and
    # a written yes/no can get it backwards. Measured: from a correct fact the
    # composer wrote "No, you will not be detained" for 74 of 100 classes, and
    # the NLI check accepted it (0.91) - the fact's "will be allowed to sit
    # ... after satisfying the required attendance" reads as "not detained".
    own_situation = mask_user_figures(question) != question
    if (
        config.answer.compose and not answer.abstained and listing is None
        and not own_situation and llm is not None and nli is not None
    ):
        watch = _Stopwatch()
        composed = compose_answer(
            question, answer.claims_kept, llm, nli, threshold=config.nli.entailment_threshold
        )
        timings.compose_ms = watch.stop()
        if composed is not None:
            answer.composed = composed.text
            answer.composed_dropped = len(composed.dropped_sentences)

    # A fabricated citation is a finding about the model, so it is surfaced in
    # the answer rather than left only in the logs.
    if fabricated:
        note = (
            f"The model cited {len(fabricated)} section(s) that do not exist: "
            f"{', '.join(fabricated)}."
        )
        answer.reason = f"{answer.reason} {note}" if answer.reason else note

    return answer
