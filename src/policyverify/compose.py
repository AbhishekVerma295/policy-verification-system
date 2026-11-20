"""
compose.py - a written answer, built only from facts that passed checking.

WHY
The verified claims are the trustworthy part of an answer, but a list of
quoted sentences does not read like an answer: there is no direct reply to
the question, nothing joins the facts up. People expect what a chat
assistant gives them. This module writes that answer - a direct first
sentence, then a short explanation - WITHOUT giving up what the system is
for.

HOW IT STAYS HONEST
1. The model writing the answer is shown ONLY the claims that passed
   verification, numbered, and the question. Not the passages: it cannot
   reach for anything that was not already proved.
2. It must mark each sentence with the facts it rests on ("[2]").
3. Every sentence it writes is then checked by the same NLI model that
   checks claims: the sentence must be ENTAILED by the facts it cites (all
   facts, if it cites none). A sentence that is not is removed, and the
   number removed is recorded.
4. If nothing survives, there is no written answer and the page falls back
   to the list of verified facts. The written answer can therefore only ever
   say less than the facts, never more.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from policyverify.llm import LLMBackend, LLMError
from policyverify.schema import ClaimVerdict
from policyverify.situation import mask_user_figures

COMPOSE_PROMPT = """You are a helpful assistant answering a student's question about their university's rules.

Write a short answer using ONLY the numbered facts below. They have been checked against the policy.
- First sentence: answer the question directly, in a full sentence. For a yes/no question start with "Yes," or "No," and continue the SAME sentence with the reason.
- Then one to three more sentences that explain, using the facts.
- Do not add anything the facts do not say: no advice, no outside knowledge.
- Keep every qualifier a fact uses ("normally", "only", "at least", "cannot exceed").
- Put the number of each fact you use at the end of the sentence, like [1] or [1][2].

Example (an unrelated library, not a real answer):
FACTS:
[1] Books can be borrowed for 14 days.
[2] A late fee of Rs 5 per day is charged.
QUESTION: Can I keep a library book for a month?
ANSWER: No, books can only be borrowed for 14 days [1]. If you keep a book longer, a late fee of Rs 5 per day is charged [2].

FACTS:
{facts}
QUESTION: {question}
ANSWER:"""

# A sentence ends at . ! or ? - plus the fact markers the model writes AFTER
# the full stop ("... normally. [1] Refund is ..."), which belong to it.
_SENTENCE_RE = re.compile(r".+?[.!?](?:\s*\[\d+\])*(?=\s+[A-Z\"'(]|\s*$)", re.DOTALL)
_REF_RE = re.compile(r"\[(\d+)\]")
_MAX_SENTENCES = 5
_LEADING_CONNECTIVE_RE = re.compile(
    r"^(?:however|but|also|additionally|moreover|so|therefore|thus|furthermore)\s*,?\s+",
    re.IGNORECASE,
)


@dataclass
class ComposedAnswer:
    text: str
    kept_sentences: list[str] = field(default_factory=list)
    dropped_sentences: list[str] = field(default_factory=list)


def _clean(reply: str) -> str:
    text = reply.strip().strip('"').strip()
    text = re.sub(r"^(answer|response)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def compose_answer(
    question: str,
    kept: list[ClaimVerdict],
    llm: LLMBackend,
    nli,
    threshold: float = 0.5,
) -> ComposedAnswer | None:
    """A short written answer from the kept claims, every sentence checked."""
    if not kept:
        return None
    facts = [v.claim.text for v in kept]
    listing = "\n".join(f"[{i}] {f}" for i, f in enumerate(facts, start=1))
    try:
        reply = llm.generate(
            # The asker's own attendance figures are masked here too: the model
            # once wrote "No, you will not get detained" for 74 of 100 classes.
            # The comparison is situation.py's job.
            COMPOSE_PROMPT.format(facts=listing, question=mask_user_figures(question).strip()),
            json_mode=False,
        )
    except LLMError:
        return None  # the verified facts are still shown; this is an extra
    text = _clean(reply)
    if not text:
        return None

    result = ComposedAnswer(text="")
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)] or [text]
    for sentence in sentences[:_MAX_SENTENCES]:
        if not result.kept_sentences:
            # This would open the answer: it must not start on a connective
            # pointing back at a removed sentence ("However, ..."). Stripped
            # BEFORE checking, so what is checked is exactly what is shown.
            sentence = _LEADING_CONNECTIVE_RE.sub("", sentence)
            sentence = sentence[:1].upper() + sentence[1:]
        refs = [int(n) for n in _REF_RE.findall(sentence) if 1 <= int(n) <= len(facts)]
        bare = _REF_RE.sub("", sentence).strip()
        if len(bare.split()) < 2:
            continue  # a stray fragment
        premise = " ".join(facts[n - 1] for n in dict.fromkeys(refs)) if refs else " ".join(facts)
        try:
            verdict = nli.check(premise, bare)
            supported = verdict.entailment >= threshold
        except Exception:  # noqa: BLE001 - an unchecked sentence is never shown
            supported = False
        if supported:
            # Keep the model's own fact markers; add one if it gave none.
            result.kept_sentences.append(sentence if refs else f"{bare} [1]")
        else:
            result.dropped_sentences.append(bare)
    if not result.kept_sentences:
        return None
    result.text = " ".join(result.kept_sentences)
    return result


__all__ = ["COMPOSE_PROMPT", "ComposedAnswer", "compose_answer"]
