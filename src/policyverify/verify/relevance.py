"""
relevance.py - does a verified claim actually answer the question that was asked?

THE GAP THIS CLOSES
    Verification (verifier.py) proves a claim is TRUE of the passage it cites.
    It never asks whether the claim is what the user wanted to know. Asked
    "Who is the current Vice Chancellor?", the system retrieved a passage about
    a welcome address and produced "The Vice-Chancellor will deliver the
    welcome address": true, cited, supported - and no answer at all. The
    question had no answer in the corpus, so the right result was to decline.

    This module is the second half of the check: after verification and before
    the keep/abstain decision, each SUPPORTED claim is judged relevant or not.
    A claim that is true but off topic is not kept (abstain.py moves it to the
    removed list with its own reason), and if nothing relevant is left the
    system declines and says the policies do not contain what was asked.

TWO SIGNALS, USED IN ORDER OF COST
    1. SIMILARITY (free, no model call). The cosine similarity between the
       question and each claim, computed with the same embedding model and the
       same query prefix that retrieval uses. Clearly high -> relevant;
       clearly low -> not relevant.
    2. THE LANGUAGE MODEL (one short yes/no call per answer). Whenever the best
       claim lands between the two thresholds - which is most answers, see
       below. The model is asked, with a JSON schema forcing a
       {"answered": true/false} reply, whether at least one statement states
       the specific fact the question is about.

    Why the model is needed and similarity is not enough: on the answers
    replayed while designing this, true-but-off-topic claims that repeat the
    question's own words ("the late fee is calculated from the date of
    payment", asked for the fee amount) scored as high as 0.81, while good
    answers (a false-premise correction, a claim that lacks the question's
    wording) scored as low as 0.57. The two ranges overlap, so the high
    "accept without asking" threshold is set above the highest off-topic score
    seen, and the model decides everything in between. Thresholds are in
    config.yaml's `relevance` section, chosen on the tuning and fresh
    questions only, never the held-out half.

    The wording of the model's prompt matters a great deal for a 4B model, and
    it was chosen by measurement, not guesswork: a plain "does a statement give
    the information asked?" prompt wrongly blocked 7 of 52 good answers (6 of
    them false-premise corrections, because a "Why ...?" question is not
    answered by a bare fact). Worked examples from an unrelated policy, one of
    them a false-premise correction, took that to 0 of 52.

FALSE PREMISES
    "Why does a B.Tech need only 150 credits?" is answered correctly by "A
    B.Tech requires 160 credits". That claim is about the very thing the
    question named, so its similarity is high and it is accepted without asking
    the model. The judge prompt also says outright that a statement giving the
    true rule behind a mistaken question counts as an answer.

WHEN THE JUDGE FAILS
    The model may be unreachable or return something unusable. The gate then
    falls back to the similarity alone, against a single cut-off, and the
    result is flagged (`RelevanceReport.judge_failed`, copied onto the Answer)
    so a fallback is never silent. If the similarity cannot be computed either
    (no embedding model), the claims are left exactly as verification found
    them and marked "unchecked": failing open there keeps the system usable,
    and the flag says so.

Nothing here loads a model by itself: the language model and the embedding
model are both passed in, so tests substitute fakes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from policyverify.config import Config, get_config
from policyverify.llm import LLMBackend
from policyverify.schema import ClaimVerdict, VerdictStatus

# The reply the judge is forced to give. Constraining decoding to this schema
# gave 0 unparseable replies in 67 trials, where plain "JSON mode" let the
# model invent its own fields.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"answered": {"type": "boolean"}},
    "required": ["answered"],
}

JUDGE_PROMPT = """A student asked a question about university policy. Below are statements taken from the policy.

QUESTION: {question}

STATEMENTS:
{statements}

Decide whether the statements contain the specific fact the question is about.
- true: a statement states that fact (a number, rule, time, person, yes/no ruling). If the question is built on a wrong assumption (often a "Why ..." question), a statement that gives the real rule or figure about that same thing counts as true, even though it does not explain "why".
- false: the statements only describe the general topic, or something nearby, but never state the specific fact asked about.

Examples (an unrelated policy, so none of these is a real answer):
Q: How much is the library fine per day? Statement: Library fines are charged for overdue books. -> false (no amount given)
Q: Why is the library fine Rs 50 per day? Statement: The library fine is Rs 5 per day. -> true (states the real amount, correcting the question)
Q: Why must students pay a Rs 900 laboratory deposit? Statement: A refundable laboratory deposit of Rs 500 is collected at admission. -> true (states the real deposit)
Q: What are the library opening hours? Statement: Books can be borrowed for 14 days. -> false (different fact)

Return JSON only: {{"answered": true}} or {{"answered": false}}
"""


class Similarity(Protocol):
    """Anything that can say how close each claim is to the question."""

    def scores(self, question: str, claims: list[str]) -> list[float]:
        """Cosine similarity of `question` to each of `claims`, in order."""
        ...


class EmbedderSimilarity:
    """Question-to-claim similarity using the retrieval store's own embedder.

    Uses `store.embedder`: the same bge model, the same query prefix on the
    question and none on the claims, so these numbers are directly comparable
    with the retrieval scores the rest of the system reports. Both vectors are
    normalised, so the dot product is the cosine.
    """

    def __init__(self, embedder) -> None:
        self.embedder = embedder

    def scores(self, question: str, claims: list[str]) -> list[float]:
        if not claims:
            return []
        query = self.embedder.encode_query(question)
        vectors = self.embedder.encode_documents(claims)
        return [sum(float(a) * float(b) for a, b in zip(query, v, strict=True)) for v in vectors]


def similarity_for_store(store) -> Similarity | None:
    """A `Similarity` for `store`, or None when the store has no embedder.

    Test stores that only implement `search()` have none, and the gate then
    runs on the model alone (or leaves claims unchecked).
    """
    embedder = getattr(store, "embedder", None)
    if embedder is None:
        return None
    if not all(callable(getattr(embedder, n, None)) for n in ("encode_query", "encode_documents")):
        return None
    return EmbedderSimilarity(embedder)


@dataclass
class RelevanceReport:
    """What the gate decided for one answer.

    `verdicts` holds a copy of every input verdict, in the input order, with
    the relevance fields filled in for the supported ones. Nothing is dropped
    here: acting on the decision (moving claims to the removed list) is
    abstain.py's job, which keeps the no-claim-lost rule in one place.
    """

    verdicts: list[ClaimVerdict]
    # True when the judge was needed but unusable and similarity decided.
    judge_failed: bool = False
    # True when the judge was actually consulted for this answer.
    judge_called: bool = False
    notes: list[str] = field(default_factory=list)


def _candidates(verdicts: list[ClaimVerdict], min_claim_score: float) -> list[int]:
    """Positions of the claims the gate has to judge: those that would be
    kept, i.e. SUPPORTED with enough confidence. Anything else is already
    removed by verification and is not relevant to this check."""
    return [
        i
        for i, v in enumerate(verdicts)
        if v.status is VerdictStatus.SUPPORTED and v.score >= min_claim_score
    ]


# Put at the front of the explanation of a claim the gate withheld, so the
# removed-claims panel says plainly why a TRUE claim is not shown as an answer.
OFF_TOPIC_NOTE = "Supported by the policy, but it does not answer the question asked."


def _mark(
    verdict: ClaimVerdict, relevant: bool, similarity: float | None, source: str
) -> ClaimVerdict:
    checks = verdict.checks.model_copy(
        update={
            "relevant": relevant,
            "relevance_similarity": None if similarity is None else round(similarity, 4),
            "relevance_source": source,
        }
    )
    update: dict = {"checks": checks}
    if not relevant:
        update["explanation"] = f"{OFF_TOPIC_NOTE} {verdict.explanation}".strip()
    return verdict.model_copy(update=update)


def ask_judge(llm: LLMBackend, question: str, statements: list[str]) -> bool | None:
    """One yes/no call. True = answered, False = not, None = unusable.

    Any failure - the model unreachable, a reply that is not the expected
    JSON - is returned as None rather than raised, because the caller has a
    defined fallback and a crash here would lose an otherwise good answer.
    """
    listing = "\n".join(f"- {s}" for s in statements)
    prompt = JUDGE_PROMPT.format(question=question, statements=listing)
    try:
        reply = llm.generate(prompt, json_mode=True, schema=JUDGE_SCHEMA)
        value = json.loads(reply).get("answered")
    except Exception:  # noqa: BLE001 - every failure has the same, recorded, fallback
        return None
    return value if isinstance(value, bool) else None


def assess_relevance(
    question: str,
    verdicts: list[ClaimVerdict],
    llm: LLMBackend | None = None,
    similarity: Similarity | None = None,
    config: Config | None = None,
) -> RelevanceReport:
    """Judge whether the supported claims answer `question`.

    The decision is made for the ANSWER first and then applied to its claims:

      best similarity >= similarity_accept  -> on topic, no model asked
      best similarity <  similarity_reject  -> off topic, no model asked
      in between                             -> ask the model once

    On topic: every candidate claim is marked relevant except the ones that
    are themselves clearly off topic (below `similarity_reject`), which are
    tangents the answer can do without. Off topic: every candidate is marked
    not relevant, and abstain.py will decline the question.
    """
    config = config or get_config()
    cfg = config.relevance
    report = RelevanceReport(verdicts=list(verdicts))
    if not cfg.enabled:
        return report

    idx = _candidates(verdicts, config.abstention.min_claim_score)
    if not idx:
        return report
    texts = [verdicts[i].claim.text for i in idx]

    sims: list[float] | None = None
    if similarity is not None:
        try:
            sims = similarity.scores(question, texts)
        except Exception:  # noqa: BLE001 - embedding trouble must not lose the answer
            sims = None
            report.notes.append("similarity could not be computed")

    def finish(relevant_flags: list[bool], source: str) -> RelevanceReport:
        out = list(verdicts)
        for k, i in enumerate(idx):
            out[i] = _mark(verdicts[i], relevant_flags[k], None if sims is None else sims[k], source)
        report.verdicts = out
        return report

    def ask() -> bool | None:
        report.judge_called = True
        shown = texts if sims is None else [
            t for _, t in sorted(zip(sims, texts, strict=True), reverse=True)
        ]
        return ask_judge(llm, question, shown[: cfg.judge_max_claims]) if llm is not None else None

    # --- no similarity: the model alone ---------------------------------
    if sims is None:
        verdict = ask()
        if verdict is None:
            report.judge_failed = True
            report.notes.append("no similarity and no usable judge: claims left unchecked")
            return finish([True] * len(idx), "unchecked")
        return finish([verdict] * len(idx), "judge")

    best = max(sims)
    if best >= cfg.similarity_accept:
        # On topic. Tangents far below the best claim are still dropped.
        return finish([s >= cfg.similarity_reject for s in sims], "similarity")
    if best < cfg.similarity_reject:
        return finish([False] * len(idx), "similarity")

    # --- borderline: ask the model --------------------------------------
    verdict = ask()
    if verdict is None:
        report.judge_failed = True
        report.notes.append("judge unusable: similarity alone decided")
        on_topic = best >= cfg.similarity_fallback
        return finish([on_topic and s >= cfg.similarity_reject for s in sims], "similarity_fallback")
    if verdict:
        return finish([s >= cfg.similarity_reject for s in sims], "judge")
    return finish([False] * len(idx), "judge")
