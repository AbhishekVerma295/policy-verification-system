"""
policyverify - a policy question-answering system that shows its work.

The core library. It knows nothing about Streamlit, nothing about the command
line, and nothing about how it is being displayed. That is deliberate: the UI,
the CLI and the evaluation harness all import from here, so all three run the
exact same code and cannot drift apart.

Phase 0 exposes only the data contracts. Later phases add the pipeline.
"""

from policyverify.schema import (
    Answer,
    CheckResults,
    Chunk,
    CitationID,
    Claim,
    ClaimVerdict,
    Document,
    DraftAnswer,
    PolicyType,
    RetrievedChunk,
    Section,
    SourceFormat,
    Timings,
    VerdictStatus,
)

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "CheckResults",
    "Chunk",
    "CitationID",
    "Claim",
    "ClaimVerdict",
    "Document",
    "DraftAnswer",
    "PolicyType",
    "RetrievedChunk",
    "Section",
    "SourceFormat",
    "Timings",
    "VerdictStatus",
    "__version__",
]
