"""
listing.py - answering "what are the types of X?" from the whole list.

WHY
Retrieval hands the model the few passages most similar to the question.
That is right for a question about one rule and wrong for a question about a
list: the plagiarism policy names ten types, one section each, and with four
passages the model could only ever list four - asked "what are the types of
plagiarism?" it answered with three. A list question is about the document's
STRUCTURE, so it is answered from the structure.

HOW
1. The question is a list question: "types / kinds / forms / categories of
   ...", "list all ...".
2. The document is the one the best retrieved passages come from.
3. The list is that document's run of sections numbered 1, 2, 3 ... in a
   row (at least three). The plagiarism policy's ten types are numbered 1-10.
   Documents whose numbering restarts or skips (the scholarship tables
   number their rows) give no clean run, and the normal path answers.
4. The question's topic word must appear in the list's headings or text, so
   "types of scholarship" is never answered with the types of plagiarism.

Each item becomes a claim: the first sentence of its section, quoted
exactly, citing that section. Nothing is written by a model, so nothing can
be invented; the claims still go through verification like any other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from policyverify.schema import Chunk, Claim

_KIND_RE = re.compile(
    r"\b(?P<kind>types?|kinds?|forms?|categories|category|varieties|classes)\s+of\s+"
    r"(?P<topic>[a-z][a-z -]{2,40})",
    re.IGNORECASE,
)
_LIST_RE = re.compile(r"\blist\s+(?:all\s+)?(?:the\s+)?(?P<topic>[a-z][a-z -]{2,40})", re.IGNORECASE)
_MIN_ITEMS = 3
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# Words that end the topic ("kinds of plagiarism DOES the policy describe")
# or say nothing about it.
_STOP = {
    "the", "a", "an", "of", "in", "at", "for", "on", "to", "and", "or", "srm",
    "university", "policy", "are", "is", "there", "does", "do", "did", "that",
    "which", "described", "describe", "mentioned", "listed", "given", "under",
}


@dataclass
class ListAnswer:
    intro: str
    claims: list[Claim]
    headings: dict[str, str]  # citation -> section heading, for display


def _match(question: str):
    return _KIND_RE.search(question) or _LIST_RE.search(question)


def list_topic(question: str) -> str | None:
    """The thing a list question asks about ("plagiarism"), or None.

    The topic runs from "of" to the first word that ends it, so "kinds of
    plagiarism does the policy describe" is about "plagiarism" and "types
    of punishment for ragging" about "punishment" ... "ragging".
    """
    match = _match(question)
    if not match:
        return None
    words = re.findall(r"[a-z]+", match.group("topic").lower())
    if match.re is _LIST_RE and words and words[0] in {"types", "kinds", "forms"}:
        return None  # "list all the types of X" - the kind pattern handles it
    kept: list[str] = []
    for word in words:
        if word in {"does", "do", "did", "is", "are", "that", "which"}:
            break  # the topic is over
        if word not in _STOP:
            kept.append(word)
    return " ".join(kept[:3]) or None


def _number(citation: str) -> int | None:
    slug = citation.rsplit("/", 1)[-1]
    match = re.match(r"(\d+)-", slug)
    return int(match.group(1)) if match else None


def _first_sentence(text: str) -> str:
    flat = " ".join(text.split())
    return _SENTENCE_RE.split(flat, maxsplit=1)[0].strip()


def find_list(
    question: str, all_chunks: list[Chunk], document_hint: str
) -> ListAnswer | None:
    """The numbered list in `document_hint` that the question asks for, or None."""
    topic = list_topic(question)
    if topic is None:
        return None

    sections: dict[int, tuple[str, str, str]] = {}  # number -> (citation, heading, text)
    pieces: dict[str, list[Chunk]] = {}
    for chunk in all_chunks:
        if chunk.citation_id.rsplit("/", 1)[0] == document_hint:
            pieces.setdefault(chunk.citation_id, []).append(chunk)
    repeated: set[int] = set()
    for citation, chunks in pieces.items():
        number = _number(citation)
        if number is None:
            continue
        if number in sections:
            repeated.add(number)
            continue
        first = min(chunks, key=lambda c: c.chunk_id)
        heading = first.section_path.split(" > ")[-1]
        sections[number] = (citation, heading, first.text)

    run = []
    n = 1
    while n in sections:
        run.append(sections[n])
        n += 1
    if len(run) < _MIN_ITEMS:
        return None
    # A number used twice within the run means the document numbers more
    # than one thing (the scholarship policy numbers its table rows): not one
    # list, so not one we can present as complete.
    if any(n in repeated for n in range(1, len(run) + 1)):
        return None

    # Every word of the question's topic must be what the list is about.
    body = " ".join(f"{h} {t}" for _, h, t in run).lower()
    if not all(re.search(rf"\b{re.escape(w[:6])}", body) for w in topic.split()):
        return None

    claims = [Claim(text=_first_sentence(text), citation_ids=[c]) for c, _, text in run]
    headings = {c: h for c, h, _ in run}
    return ListAnswer(
        intro=f"The policy lists {len(run)} {topic_label(question, topic)}:",
        claims=claims,
        headings=headings,
    )


def topic_label(question: str, topic: str) -> str:
    """'types of plagiarism' as the question put it."""
    match = _KIND_RE.search(question)
    kind = match.group("kind").lower() if match else "items"
    kind = kind if kind.endswith("s") else kind + "s"
    return f"{kind} of {topic}"


__all__ = ["ListAnswer", "find_list", "list_topic"]
