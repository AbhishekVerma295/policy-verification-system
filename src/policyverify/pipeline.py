"""
pipeline.py - the whole system, end to end, in one function.

    question -> retrieve -> generate -> verify -> decide -> Answer

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

from policyverify.abstain import decide
from policyverify.config import Config, get_config
from policyverify.generate import generate_answer_draft
from policyverify.llm import LLMBackend
from policyverify.retrieve import retrieve
from policyverify.schema import Answer, PolicyType, RetrievedChunk, Timings
from policyverify.verify import verify_claims


class _Stopwatch:
    """Wall-clock timing for one stage, in milliseconds."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


def answer_question(
    question: str,
    k: int | None = None,
    university: str | None = None,
    policy_type: PolicyType | str | None = None,
    config: Config | None = None,
    store=None,
    llm: LLMBackend | None = None,
    nli=None,
) -> Answer:
    """Answer one policy question, with every claim verified.

    Returns an `Answer` whether or not the system decided to answer. An
    abstention is a result, not an error: it carries the reason and the claims
    that were considered and rejected.
    """
    config = config or get_config()
    timings = Timings()

    # --- 1. retrieve -----------------------------------------------------
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

    # --- 2. generate -----------------------------------------------------
    watch = _Stopwatch()
    draft, fabricated = generate_answer_draft(question, chunks, llm=llm, config=config)
    timings.generate_ms = watch.stop()

    # --- 3. verify -------------------------------------------------------
    watch = _Stopwatch()
    verdicts = verify_claims(draft.claims, chunks, nli=nli, store=store, config=config)
    timings.verify_ms = watch.stop()

    # --- 4. decide -------------------------------------------------------
    decision = decide(verdicts, retrieved=chunks, config=config)

    answer = Answer(
        question=question,
        university_filter=university,
        claims_kept=decision.kept,
        claims_removed=decision.removed,
        abstained=decision.abstained,
        reason=decision.reason,
        timings=timings,
        config_fingerprint=config.fingerprint(),
    )

    # A fabricated citation is a finding about the model, so it is surfaced in
    # the answer rather than left only in the logs.
    if fabricated:
        note = (
            f"The model cited {len(fabricated)} section(s) that do not exist: "
            f"{', '.join(fabricated)}."
        )
        answer.reason = f"{answer.reason} {note}" if answer.reason else note

    return answer
