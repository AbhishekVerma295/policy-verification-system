"""
citation.py - is the cited section real, and was it actually shown to the model?

TWO DIFFERENT FAILURES, KEPT SEPARATE
    Most systems collapse these into one "bad citation" bucket. They have
    different causes and deserve different treatment:

    FABRICATED - the citation names a section that does not exist anywhere in
        the corpus. The model invented it. Nothing can support this claim,
        because the thing it points at is not real.

    MISUSED - the section exists, but does not actually support the claim.
        The model cited something real and got the attribution wrong. The
        claim might even be true; the sourcing is not.

    Telling them apart is most of the value here. A fabricated citation is a
    model failure with no possible remedy; a misused one is a grounding
    failure that better retrieval or prompting could fix. Reporting both as
    "bad citation" would hide which problem you actually have.

    (A third, rarer case: the section exists but was never shown to the model.
    Since the model can only see the retrieved passages, this generally means
    it mangled one real ID into another. Recorded separately rather than
    lumped in with fabrication, because the cause is different.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from policyverify.schema import Chunk, CitationID, RetrievedChunk


@dataclass
class CitationCheck:
    """What we know about one citation the model produced."""

    citation_id: str
    well_formed: bool
    exists: bool
    was_retrieved: bool
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def fabricated(self) -> bool:
        """True if this points at nothing real in the corpus."""
        return not self.exists

    def describe(self) -> str:
        if not self.well_formed:
            return f"{self.citation_id!r} is not a valid citation format"
        if not self.exists:
            return f"{self.citation_id} does not exist in the corpus (fabricated)"
        if not self.was_retrieved:
            return f"{self.citation_id} exists but was not among the retrieved passages"
        return f"{self.citation_id} resolves to a real section"


def check_citation(
    citation_id: str,
    retrieved: list[RetrievedChunk],
    store=None,
) -> CitationCheck:
    """Resolve one citation against the retrieved passages and the corpus.

    The retrieved passages are checked first because that is the common case
    and costs nothing. The store is consulted only when a citation is not
    among them - that is what distinguishes "invented" from "real but not
    shown to the model", and it requires looking at the whole corpus rather
    than just this question's results.
    """
    well_formed = CitationID.is_valid(citation_id)
    if not well_formed:
        return CitationCheck(
            citation_id=citation_id,
            well_formed=False,
            exists=False,
            was_retrieved=False,
        )

    matching = [r.chunk for r in retrieved if r.chunk.citation_id == citation_id]
    if matching:
        return CitationCheck(
            citation_id=citation_id,
            well_formed=True,
            exists=True,
            was_retrieved=True,
            chunks=matching,
        )

    # Not retrieved. It may still be a real section elsewhere in the corpus.
    corpus_chunks: list[Chunk] = []
    if store is not None:
        try:
            corpus_chunks = store.get_by_citation(citation_id)
        except Exception:
            # A store problem must not be reported as a fabricated citation -
            # that would blame the model for our own failure.
            corpus_chunks = []

    return CitationCheck(
        citation_id=citation_id,
        well_formed=True,
        exists=bool(corpus_chunks),
        was_retrieved=False,
        chunks=corpus_chunks,
    )


def check_citations(
    citation_ids: list[str],
    retrieved: list[RetrievedChunk],
    store=None,
) -> list[CitationCheck]:
    """Check every citation on a claim."""
    return [check_citation(cid, retrieved, store) for cid in citation_ids]
