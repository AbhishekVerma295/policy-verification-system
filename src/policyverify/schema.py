"""
schema.py - the data shapes that every other module agrees on.

This is the most important file in the project. Everything else imports from
here, which is exactly why it gets written before any other code.

Two contracts live in this file:

  1. CitationID - how we point at one specific part of one specific policy
  2. The models - Document, Chunk, Claim, ClaimVerdict, Answer

Why Pydantic: it checks data at the boundary. When the language model returns
JSON with a missing field or the wrong type, you get a clear error at the
moment it happens, instead of a confusing crash three functions later.

Nothing in this file talks to a network, a database, or a model. It is pure
data definitions, so it is fast to import and easy to test.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Contract 1: the citation ID
# ---------------------------------------------------------------------------
#
# A citation ID answers the question "where exactly did this come from?".
# Format:  {university}/{policy}/{section}
# Example: uni_a/academic_integrity/4.2
#
# It has to be three things at once:
#   - readable by a human (so you can eyeball whether it looks right)
#   - checkable by a machine (so we can prove the section actually exists)
#   - stable (so the same passage always gets the same ID)

# university and policy are "slugs": lowercase letters, digits, underscores.
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# A section is either the document's own numbering ("4.2", "A.1.3") or, when
# the document does not number its sections, a slug made from the heading
# ("plagiarism"). Both are allowed.
_SECTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CitationID(BaseModel):
    """One specific section of one specific policy at one specific university.

    Use `CitationID.parse("uni_a/academic_integrity/4.2")` to read one from a
    string, and `str(cid)` or `cid.render()` to turn it back into a string.

    Frozen means it cannot be changed after it is created. That is deliberate:
    a citation that could be edited in place would be very easy to corrupt by
    accident, and a citation you cannot trust is worse than no citation.
    """

    model_config = ConfigDict(frozen=True)

    university: str
    policy: str
    section: str

    @field_validator("university", "policy")
    @classmethod
    def _check_slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(
                f"{v!r} is not a valid slug - use lowercase letters, digits "
                f"and underscores, starting with a letter (e.g. 'uni_a')"
            )
        return v

    @field_validator("section")
    @classmethod
    def _check_section(cls, v: str) -> str:
        if not _SECTION_RE.match(v):
            raise ValueError(
                f"{v!r} is not a valid section - use the document's own "
                f"numbering like '4.2', or a heading slug like 'plagiarism'"
            )
        return v

    @classmethod
    def parse(cls, raw: str) -> CitationID:
        """Turn 'uni_a/academic_integrity/4.2' into a CitationID.

        Raises ValueError if the string is not exactly three parts. We are
        strict on purpose: a citation we cannot parse is a citation we cannot
        verify, and silently accepting it would defeat the whole project.
        """
        parts = raw.strip().split("/")
        if len(parts) != 3:
            raise ValueError(
                f"citation {raw!r} must have exactly 3 parts separated by '/' "
                f"({{university}}/{{policy}}/{{section}}), got {len(parts)}"
            )
        university, policy, section = (p.strip() for p in parts)
        return cls(university=university, policy=policy, section=section)

    @classmethod
    def is_valid(cls, raw: str) -> bool:
        """True if `raw` can be parsed. Never raises - use this for filtering."""
        try:
            cls.parse(raw)
            return True
        except (ValueError, TypeError):
            return False

    def render(self) -> str:
        """Turn this back into 'uni_a/academic_integrity/4.2'."""
        return f"{self.university}/{self.policy}/{self.section}"

    def __str__(self) -> str:  # so f"{cid}" and print(cid) do the right thing
        return self.render()


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class PolicyType(StrEnum):
    """The kinds of policy we collect from each university.

    StrEnum means these behave as plain strings everywhere: they serialize to
    strings in JSON, Chroma accepts them as metadata, and printing one gives
    "attendance" rather than "PolicyType.ATTENDANCE".
    """

    ATTENDANCE = "attendance"
    EXAMINATION = "examination"
    ACADEMIC_INTEGRITY = "academic_integrity"
    SCHOLARSHIP = "scholarship"
    RESIDENCE = "residence"
    CODE_OF_CONDUCT = "code_of_conduct"


class SourceFormat(StrEnum):
    """What the original document was, before we turned it into text."""

    HTML = "html"
    PDF = "pdf"


class VerdictStatus(StrEnum):
    """The result of checking one claim against the passages it cited.

    NEUTRAL is not a failure of the checker - it is a real and useful answer.
    It means "this passage neither proves nor disproves the claim", which is
    exactly the situation where the system should consider staying quiet.
    """

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEUTRAL = "NEUTRAL"


# ---------------------------------------------------------------------------
# Documents (produced by Phase 1: ingestion)
# ---------------------------------------------------------------------------


class Section(BaseModel):
    """One numbered or titled part of a policy document.

    `number` is whatever the document itself calls this section ("4.2"). It is
    None when the document does not number its sections, in which case the
    citation ID falls back to a slug built from the heading.
    """

    number: str | None = None
    heading: str
    text: str
    # Breadcrumb of headings from the top of the document down to this one,
    # e.g. ["Academic Integrity", "Offences", "Plagiarism"]. This is what makes
    # a chunk understandable on its own once it has been pulled out of context.
    path: list[str] = Field(default_factory=list)
    level: int = 1

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("section text cannot be empty")
        return v

    def section_key(self) -> str:
        """The third part of a citation ID for this section.

        Uses the document's own numbering when it has any, because that is what
        a reader would quote. Falls back to a slug of the heading otherwise.
        """
        if self.number:
            return self.number
        slug = re.sub(r"[^a-z0-9]+", "_", self.heading.lower()).strip("_")
        return slug or "unnamed"


class Document(BaseModel):
    """One policy document from one university, after cleaning.

    Everything upstream of this (PDF quirks, HTML soup, encoding problems) has
    been dealt with by the time a Document exists. Everything downstream only
    ever sees this shape, and does not care where the document came from.
    """

    doc_id: str  # "{university}/{policy_type}"
    university: str  # slug, e.g. "uni_a"
    university_name: str  # display name, e.g. "University A"
    policy_type: PolicyType
    title: str
    source_url: str
    source_format: SourceFormat
    retrieved_at: date
    checksum: str  # sha256 of the raw bytes, so we can prove what we parsed
    sections: list[Section] = Field(default_factory=list)

    @field_validator("university")
    @classmethod
    def _check_slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(f"university {v!r} must be a slug like 'uni_a'")
        return v

    def citation_for(self, section: Section) -> CitationID:
        """Build the citation ID for one section of this document."""
        return CitationID(
            university=self.university,
            policy=self.policy_type.value,
            section=section.section_key(),
        )


# ---------------------------------------------------------------------------
# Chunks (produced by Phase 2: chunking + indexing)
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A searchable piece of a document, with everything needed to cite it.

    A chunk has to carry its own identity. Once the search engine hands it back
    to us, we no longer have the document it came from, so anything we need in
    order to cite it correctly has to already be attached.
    """

    chunk_id: str
    text: str
    citation_id: str  # rendered CitationID, e.g. "uni_a/attendance/4.2"
    university: str
    university_name: str
    policy_type: PolicyType
    section_path: str  # " > ".join(section.path)
    source_url: str

    @field_validator("citation_id")
    @classmethod
    def _check_citation(cls, v: str) -> str:
        CitationID.parse(v)  # raises if malformed
        return v

    def citation(self) -> CitationID:
        """The parsed citation ID for this chunk."""
        return CitationID.parse(self.citation_id)

    def to_metadata(self) -> dict[str, str]:
        """Flatten to the metadata dict the vector store keeps beside the text.

        Chroma only accepts flat primitives (str, int, float, bool) as metadata,
        which is why this is a deliberate conversion rather than just dumping
        the model. These fields are what we filter on later - most importantly
        `university`, which is our main defence against mixing up two schools.
        """
        return {
            "chunk_id": self.chunk_id,
            "citation_id": self.citation_id,
            "university": self.university,
            "university_name": self.university_name,
            "policy_type": self.policy_type.value,
            "section_path": self.section_path,
            "source_url": self.source_url,
        }


class RetrievedChunk(BaseModel):
    """A chunk plus how well it matched the question."""

    chunk: Chunk
    score: float
    rank: int


# ---------------------------------------------------------------------------
# Claims and verdicts (Phases 3 and 4)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """One single factual statement, with the passages it says back it up.

    "Single" is the important word. "Students must attend 75% of classes and
    may appeal to the Dean" is two claims, not one, and has to be split - you
    cannot give one verdict to a sentence where half is right and half is wrong.
    """

    text: str
    citation_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("claim text cannot be empty")
        return v.strip()

    @field_validator("citation_ids")
    @classmethod
    def _check_citations(cls, v: list[str]) -> list[str]:
        # We keep malformed IDs rather than dropping them. A made-up citation
        # is a finding, not noise - the verifier reports it as a fabrication.
        return [c.strip() for c in v if c and c.strip()]


class DraftAnswer(BaseModel):
    """Exactly what the language model is required to return.

    This is the structural contract from Phase 3. We hand the JSON schema of
    this model to the model itself, then validate its reply against it.

    The alternative - letting the model write prose and then trying to pull
    claims back out of it with regexes - breaks constantly, because prose has
    no rules. Making the format structural is what turns "usually works" into
    "works or fails loudly".
    """

    claims: list[Claim] = Field(default_factory=list)

    @classmethod
    def json_schema_for_prompt(cls) -> dict:
        """The JSON schema to show the model, so it knows the required shape."""
        return cls.model_json_schema()


class CheckResults(BaseModel):
    """The individual checks that were run on one claim.

    Kept separate from the final verdict on purpose. When something looks wrong
    you want to know *which* check objected, and later on this is what lets you
    compare the checks against each other.
    """

    # Does the cited passage actually prove the claim? (the NLI model)
    nli_label: str | None = None
    nli_score: float | None = None

    # Do the numbers agree? ("75%" vs "80%" is the error that matters here)
    numeric_ok: bool | None = None
    numeric_detail: str | None = None

    # Does the cited section exist at all? A False here means the model
    # invented a citation, which is a different and more serious failure than
    # citing something real that happens not to support the claim.
    citation_exists: bool = False
    citation_supports: bool | None = None


class ClaimVerdict(BaseModel):
    """One claim, after checking, with the evidence and the reasoning."""

    claim: Claim
    status: VerdictStatus
    score: float = Field(ge=0.0, le=1.0)
    evidence_chunk_ids: list[str] = Field(default_factory=list)
    checks: CheckResults = Field(default_factory=CheckResults)
    # Plain-language explanation, shown in the UI next to the claim.
    explanation: str = ""


# ---------------------------------------------------------------------------
# The final answer (Phases 5 and 8)
# ---------------------------------------------------------------------------


class Timings(BaseModel):
    """How long each stage took, in milliseconds.

    Split by stage rather than one total, because "it is slow" is not
    actionable but "verification is 80% of the time" tells you where to look.
    """

    retrieve_ms: float = 0.0
    generate_ms: float = 0.0
    verify_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.retrieve_ms + self.generate_ms + self.verify_ms


class Answer(BaseModel):
    """What the system gives back for one question.

    Note that removed claims are *kept*, not deleted. That list is the whole
    transparency feature: the user gets to see what the system started to say
    and then decided it could not stand behind.
    """

    question: str
    university_filter: str | None = None

    claims_kept: list[ClaimVerdict] = Field(default_factory=list)
    claims_removed: list[ClaimVerdict] = Field(default_factory=list)

    abstained: bool = False
    reason: str | None = None  # why we abstained, in plain language

    timings: Timings = Field(default_factory=Timings)
    # Which model and settings produced this, so a saved run can be understood
    # months later without guessing.
    config_fingerprint: dict[str, str] = Field(default_factory=dict)

    @property
    def hallucination_count(self) -> int:
        """How many claims we generated but could not stand behind."""
        return len(self.claims_removed)

    def all_citations(self) -> list[str]:
        """Every citation ID used by a claim we kept."""
        seen: list[str] = []
        for verdict in self.claims_kept:
            for cid in verdict.claim.citation_ids:
                if cid not in seen:
                    seen.append(cid)
        return seen
