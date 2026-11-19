"""
Verification - the heart of the project.

Three independent checks run on every claim the language model produces:

  nli.py       Does the cited passage actually prove this claim?
  numeric.py   Do the numbers agree? ("75%" vs "80%" is the error that hurts.)
  citation.py  Does the cited section exist, and was it shown to the model?

verifier.py combines them into one ClaimVerdict per claim.

relevance.py is a fourth, separate question asked after those three: does a
claim that IS true of its passage actually answer the question that was
asked? See its docstring.

Why not just ask the language model to check its own work: a model grading
itself tends to repeat its own mistakes, because the same misunderstanding
that produced the wrong claim also produces the wrong grade. A separate NLI
model answers one narrow question - "does text A prove text B?" - and it
answers it independently.

The citation checks are deliberately two separate things. A section that does
not exist is a fabricated citation. A section that exists but does not support
the claim is a misused citation. Those are different failures with different
causes, and telling them apart is most of the value here.

nli.py scores a claim against each sentence of the cited passage (and short
runs of neighbouring sentences), not against whole paragraphs - NLI models
are only reliable at sentence size. An exact quote of the passage is
accepted without the model. The sentence that decided each verdict is
recorded on CheckResults so the interface can show it.

Built in Phase 4. Measured on real retrieved passages: 7/7, including a
planted "80% attendance" claim correctly refuted at 1.00 confidence against
the real 75% rule.
"""

from policyverify.verify.citation import CitationCheck, check_citation, check_citations
from policyverify.verify.nli import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    NLIChecker,
    NLIResult,
    evidence_units,
    find_exact_quote,
    split_sentences,
    split_windows,
)
from policyverify.verify.numeric import (
    Number,
    NumericCheck,
    check_numbers,
    extract_numbers,
)
from policyverify.verify.relevance import (
    EmbedderSimilarity,
    RelevanceReport,
    assess_relevance,
    similarity_for_store,
)
from policyverify.verify.verifier import verify_claim, verify_claims

__all__ = [
    "CONTRADICTION",
    "ENTAILMENT",
    "NEUTRAL",
    "CitationCheck",
    "EmbedderSimilarity",
    "NLIChecker",
    "NLIResult",
    "Number",
    "NumericCheck",
    "RelevanceReport",
    "assess_relevance",
    "check_citation",
    "check_citations",
    "check_numbers",
    "evidence_units",
    "extract_numbers",
    "find_exact_quote",
    "similarity_for_store",
    "split_sentences",
    "split_windows",
    "verify_claim",
    "verify_claims",
]
