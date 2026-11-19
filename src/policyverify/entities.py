"""
entities.py - who a question is about, who a passage is about, and a capped
boost when they match.

WHY
Vector search matches on topic. It is weak on WHO. "How long may a parent
stay in the hostel?" and "How many overnight guests may a hosteller have?" are
the same topic to the embedding model, but they are answered by different
rules for different people. When a question names who it is for, a passage
that names the same people should rank higher.

HOW
1. Extraction is a fixed lexicon, not a model. The entities that matter here
   are a small, known set of roles (parent, warden, hosteller, UG...). A
   lexicon is deterministic, instant, and runs identically on questions and
   on chunks, so the two sides always agree on what "parent" means.
   The lexicon was built from the real SRM corpus (counts measured
   2026-11-19 across all 6 policies), not from guesses.

2. Weights come from the corpus, not from hand-tuning. Each entity's weight
   is its IDF over the indexed chunks, scaled to [0, 1]. "student" appears
   in most chunks, so matching it says almost nothing and earns almost no
   boost. "parent" appears in a few, so a match is real evidence.

3. The boost is capped. It may reorder passages whose similarity is already
   close. It must not lift an off-topic passage over a relevant one just
   because it names the same people. See RetrievalConfig.entity_boost_cap.

4. Chunks are tagged from their source text and heading only, never from
   their model-written summary. A summary that wrongly mentions "parents"
   must not be able to boost a chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

# canonical entity -> pattern. Matched case-insensitively on word boundaries,
# except where a pattern needs case (UG/PG are only abbreviations in capitals).
#
# Notes from the corpus:
#   - "guest" in the examination policy is "Chief Guest" (a ceremony) and
#     "guest faculty" - not a hostel visitor. Excluded by lookarounds.
#   - "resident" is only used for hostel residents in this corpus.
_LEXICON: dict[str, str] = {
    "student": r"students?",
    "parent": r"parents?|guardians?|father|mother|local guardians?",
    "hosteller": r"hostell?ers?|hostel residents?|residents?|boarders?|inmates?",
    "day_scholar": r"day[- ]scholars?",
    "visitor": r"visitors?|(?<!chief )guests?(?! faculty| lecture)",
    "warden": r"wardens?|deputy wardens?|chief wardens?",
    "faculty": r"faculty|teachers?|professors?|faculty advisors?|course teachers?",
    "staff": r"staff|employees?",
    "first_year": r"first[- ]year|freshers?|i year",
    "research_scholar": r"research scholars?|ph\.?\s?d\.?(?: scholars?)?",
    "female": r"girls?|women|female|ladies",
    "male": r"boys?|men|male|gents",
    "alumni": r"alumni|alumnus",
    "sports": r"sports? (?:persons?|quota|achievers?)|sportspersons?",
    "dean": r"deans?",
    "hod": r"hods?|heads? of (?:the )?departments?",
    "registrar": r"registrar",
    "controller_of_exams": r"controller of examinations?|coe",
    "disciplinary_committee": r"disciplinary committee|discipline committee",
}

# Case-sensitive: lowercase "ug"/"pg" are not these abbreviations.
_CASED: dict[str, str] = {
    "ug": r"UG|[Uu]ndergraduates?|B\.\s?Tech",
    "pg": r"PG|[Pp]ostgraduates?|M\.\s?Tech|MBA",
}

_PATTERNS: dict[str, re.Pattern] = {
    **{k: re.compile(rf"\b(?:{v})\b", re.IGNORECASE) for k, v in _LEXICON.items()},
    **{k: re.compile(rf"\b(?:{v})\b") for k, v in _CASED.items()},
}

ENTITIES: tuple[str, ...] = tuple(sorted(_PATTERNS))

# Chroma metadata must be a flat primitive, so a chunk's entity set is stored
# as one delimited string. Leading and trailing delimiters make a substring
# test unambiguous ("|hod|" never matches inside another name).
_SEP = "|"


def extract_entities(text: str) -> set[str]:
    """Every canonical entity mentioned in `text`."""
    return {name for name, pattern in _PATTERNS.items() if pattern.search(text)}


def encode_entities(entities: Iterable[str]) -> str:
    items = sorted(set(entities))
    return f"{_SEP}{_SEP.join(items)}{_SEP}" if items else ""


def decode_entities(value: str | None) -> set[str]:
    return {e for e in (value or "").split(_SEP) if e}


def entity_weights(chunk_entities: Iterable[set[str]]) -> dict[str, float]:
    """IDF of each entity over the chunks, scaled to [0, 1].

    An entity in every chunk gets 0. One in a single chunk gets 1. Computed at
    index time and saved in the index manifest, so search uses the weights of
    the corpus it is actually searching.
    """
    sets = list(chunk_entities)
    n = len(sets)
    if n <= 1:
        return {}
    df = Counter(e for s in sets for e in s)
    top = math.log(n)
    return {e: round(math.log(n / count) / top, 4) for e, count in sorted(df.items())}


def entity_boost(
    query_entities: set[str],
    chunk_entities: set[str],
    weights: dict[str, float],
    weight: float,
    cap: float,
) -> float:
    """The re-ranking bonus for one chunk: weight * matched IDF, capped.

    Uses the single best-matching entity, not a sum. A chunk that names five
    roles from a long question should not outrank one that names the role the
    question is really about. Summing would also make the cap the usual
    outcome rather than the limit.
    """
    matched = query_entities & chunk_entities
    if not matched:
        return 0.0
    best = max(weights.get(e, 0.0) for e in matched)
    return round(min(cap, weight * best), 4)


__all__ = [
    "ENTITIES",
    "decode_entities",
    "encode_entities",
    "entity_boost",
    "entity_weights",
    "extract_entities",
]
