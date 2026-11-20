"""
bm25.py - keyword search over the chunks, to sit beside the embedding search.

WHY A SECOND KIND OF SEARCH
    The vector store finds passages by meaning: the question and every chunk
    are turned into embeddings, and the closest ones win. That is good at
    paraphrase, but on this corpus it is touchy about wording. Every chunk is
    a university policy, so their similarity scores bunch together (roughly
    0.55-0.61 on the real index) and a small change in phrasing reorders them.

    Measured: for "What is minimum marks needed to pass a course ?" the section
    that literally says "To 'Pass' a course ... a minimum of 50%" ranked about
    10th by embeddings alone, so the generator never saw it and answered with
    the attendance rule instead. A keyword search has no trouble with that
    question - "pass", "course" and "marks" are right there in the text.

    So we run both and merge their rankings (see retrieve.py). Each covers the
    other's blind spot: keywords catch exact terms the embedding blurs, and the
    embedding catches rewordings that share no words with the source.

WHAT BM25 IS
    BM25 (Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25
    and Beyond", 2009) is the standard keyword-ranking formula behind most
    search engines. For each query word that appears in a chunk it adds:

        idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len))

    - idf is high for RARE words and near zero for words in every chunk, so
      "condonation" counts for far more than "students".
    - tf is how often the word appears in this chunk. The k1 term makes that
      saturate: the tenth mention adds much less than the first.
    - b scales for chunk length, so a long chunk does not win just because it
      has more words in which to contain the query terms by chance.

    It is about forty lines of arithmetic, and the corpus is a few hundred
    chunks, so it is written here in plain Python rather than adding a
    dependency for it.

WHAT GETS INDEXED
    The chunk's section path (its headings) plus its text. The headings are
    included for the same reason the embedding text includes them (see
    store.py): "Method of Awarding Letter Grades" is often the best keyword
    match for a question, even when the body text words it differently.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from policyverify.schema import Chunk, PolicyType

# Words so common in questions that matching them says nothing about which
# passage is relevant. "What is the ..." matches nearly every chunk; without
# this list those matches would add a little noise to every score. BM25's idf
# already pushes such words towards zero, but question words like "what",
# "how" and "can" are rarer in policy text than in questions, so idf alone
# would still give them some weight. Kept deliberately small: domain words
# such as "not", "minimum" or "maximum" carry meaning here and must stay.
STOP_WORDS = frozenset(
    """
    a an the of to in on for and or is are be by with at as it this that from
    what how many much does do can i my me which who when why will shall may
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric words, minus stop words.

    Punctuation splits words ("50%" -> "50", "re-examination" -> "re",
    "examination"), which is what we want: the question and the policy may
    hyphenate or punctuate the same idea differently. No stemming, so "pass"
    and "passed" are different words - a known limitation, accepted because
    the embedding search on the other side of the merge handles word forms.
    """
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOP_WORDS]


def indexed_text(chunk: Chunk) -> str:
    """What the keyword index sees for one chunk: its headings, then its text."""
    return f"{chunk.section_path} {chunk.text}"


def _policy_value(policy_type: PolicyType | str | None) -> str | None:
    if policy_type is None:
        return None
    return policy_type.value if isinstance(policy_type, PolicyType) else policy_type


class BM25Index:
    """An in-memory BM25 index over a fixed list of chunks.

    Built once and then only read, so it needs no locking and no updates: if
    the underlying chunks change, build a new one (VectorStore does this for
    you whenever the index is rebuilt).
    """

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = list(chunks)

        # Pre-compute everything that does not depend on the query.
        self._lengths: list[int] = []
        # word -> list of (position in self.chunks, how often it appears).
        # An "inverted index": scoring a query only touches the chunks that
        # actually contain one of its words, not all of them.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        for i, chunk in enumerate(self.chunks):
            words = tokenize(indexed_text(chunk))
            self._lengths.append(len(words))
            for word, tf in Counter(words).items():
                self._postings.setdefault(word, []).append((i, tf))

        n = len(self.chunks)
        self._avg_len = (sum(self._lengths) / n) if n else 0.0
        # The "+ 1 inside the log" form (as used by Lucene) keeps idf positive
        # even for a word found in more than half the chunks. The original
        # formula goes negative there, which would make matching a common
        # word *lower* a chunk's score - surprising and unhelpful.
        self._idf: dict[str, float] = {
            word: math.log(1 + (n - len(posts) + 0.5) / (len(posts) + 0.5))
            for word, posts in self._postings.items()
        }

    def __len__(self) -> int:
        return len(self.chunks)

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score of every chunk that shares at least one word with `query`.

        Keyed by position in self.chunks. Chunks sharing no words are absent
        rather than scored 0, which keeps the result small.
        """
        out: dict[int, float] = {}
        if not self.chunks:
            return out
        # set(): a word repeated in the question counts once. "pass ... pass"
        # should not mean twice as much as "pass".
        for word in set(tokenize(query)):
            posts = self._postings.get(word)
            if not posts:
                continue
            idf = self._idf[word]
            for i, tf in posts:
                norm = 1 - self.b + self.b * self._lengths[i] / self._avg_len
                out[i] = out.get(i, 0.0) + idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out

    def search(
        self,
        query: str,
        k: int,
        university: str | None = None,
        policy_type: PolicyType | str | None = None,
        audience: set[str] | None = None,
    ) -> list[tuple[Chunk, float]]:
        """The `k` best keyword matches, best first, as (chunk, bm25 score).

        The filters mean exactly what they mean in VectorStore.search: they
        restrict WHICH chunks can be returned, they do not re-rank. Word
        statistics (idf, average length) still come from the whole corpus, in
        the same way the embedding side compares against absolute vectors -
        so filtering never changes how a chunk scores, only whether it is
        eligible.

        Ties are broken by chunk_id so the same query always gives the same
        order. An unstable order would make retrieval, and so the whole
        answer, vary between runs for no reason.
        """
        policy = _policy_value(policy_type)
        hits = []
        for i, score in self.scores(query).items():
            chunk = self.chunks[i]
            if university and chunk.university != university:
                continue
            if policy and chunk.policy_type.value != policy:
                continue
            # Same rule as the aud_* filter in VectorStore.search: a chunk
            # with no audience labels is for everyone.
            if audience and chunk.audience and not (set(chunk.audience) & audience):
                continue
            hits.append((chunk, score))
        hits.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
        return hits[:k]
