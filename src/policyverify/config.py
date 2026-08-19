"""
config.py - every setting the system has, loaded from config.yaml.

One place for all the knobs. This matters more than it looks:

  - You can change behaviour without editing code, which means you can tune
    the abstention threshold without risking a typo in a function.
  - Every saved run records which settings produced it, so a result from six
    weeks ago can still be explained.
  - During a demo you can change a number, re-run, and show the difference.

The settings are validated by Pydantic, so a typo in config.yaml produces a
clear message naming the bad field instead of a confusing failure later on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# The project root is two levels up from this file:
#   <root>/src/policyverify/config.py  ->  <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class PathsConfig(BaseModel):
    """Where things live on disk. All relative to the project root."""

    manifest: str = "data/manifest.yaml"
    raw: str = "data/raw"
    processed: str = "data/processed"
    index: str = "data/index"
    runs: str = "eval/results"

    def resolve(self, name: str) -> Path:
        """Turn one of these settings into a real absolute path."""
        return PROJECT_ROOT / getattr(self, name)


class EmbeddingConfig(BaseModel):
    """The model that turns text into vectors for searching.

    Runs on CPU on purpose. It is small and fast enough there, and it keeps all
    8 GB of VRAM free for Qwen, which actually needs it.
    """

    model_name: str = "BAAI/bge-base-en-v1.5"
    device: str = "cpu"
    batch_size: int = 32
    # bge models work noticeably better when the query (not the document) is
    # prefixed with a short instruction. This is the prefix the authors suggest.
    query_prefix: str = "Represent this sentence for searching relevant passages: "


class ChunkingConfig(BaseModel):
    """How documents get split into searchable pieces.

    We split on headings rather than at a fixed character count, so a chunk is
    always a whole section. That is what makes a citation like '4.2' mean
    something precise. Sections longer than max_chars are split further, but
    only as a fallback.
    """

    max_chars: int = 1800
    min_chars: int = 120
    overlap_chars: int = 150


class RetrievalConfig(BaseModel):
    """How we find passages that might answer the question."""

    top_k: int = 6
    # Filtering by university is our main defence against mixing up two
    # schools' policies, which is the single most likely wrong answer here.
    enable_university_filter: bool = True


class LLMConfig(BaseModel):
    """Which language model writes the draft answer.

    `backend` is the one swappable piece in the system. Today it is Ollama
    running locally; the point of naming it here is that swapping it later is a
    config change rather than a code change.

    temperature is 0 so the same question gives the same answer. This is not a
    style preference - a system whose output changes between runs cannot be
    evaluated or debugged.
    """

    backend: str = "ollama"
    model: str = "qwen3:4b"
    temperature: float = 0.0
    # Hard ceiling so one bad question cannot hang the whole app.
    timeout_seconds: int = 120
    num_ctx: int = 8192


class NLIConfig(BaseModel):
    """The fact-checker: does this passage actually prove this claim?

    Also on CPU, for the same reason as the embedding model. The exact
    checkpoint is chosen by running spikes/spike_nli.py - do not assume this
    default is the best one until you have measured it.
    """

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str = "cpu"
    # An entailment score below this is not treated as support.
    entailment_threshold: float = 0.5


class VerificationConfig(BaseModel):
    """Settings for the checks run on each claim."""

    # Percentages and counts must match exactly. Dates get a little slack
    # because documents phrase them inconsistently ("14 days" vs "two weeks").
    numeric_exact_match: bool = True
    # If a claim cites a section that does not exist, that is a fabricated
    # citation - always a failure, never a judgement call.
    fail_on_missing_citation: bool = True


class AbstentionConfig(BaseModel):
    """When the system should stay quiet instead of answering.

    Tune `min_claim_score` on the tuning half of the adversarial set only, then
    report numbers on the held-out half. Tuning and reporting on the same
    questions would only prove it works on the questions you tuned it for.
    """

    min_claim_score: float = 0.5
    # If fewer than this many claims survive checking, abstain entirely rather
    # than giving a half-answer.
    min_supported_claims: int = 1
    # If the best retrieved passage is weaker than this, the corpus probably
    # does not cover the question at all.
    min_retrieval_score: float = 0.25


class Config(BaseModel):
    """The whole configuration."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    nli: NLIConfig = Field(default_factory=NLIConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)

    def fingerprint(self) -> dict[str, str]:
        """A short summary of the settings that affect results.

        Saved with every run so that a set of numbers can always be traced back
        to the settings that produced them.
        """
        return {
            "llm_model": self.llm.model,
            "llm_backend": self.llm.backend,
            "llm_temperature": str(self.llm.temperature),
            "embedding_model": self.embedding.model_name,
            "nli_model": self.nli.model_name,
            "top_k": str(self.retrieval.top_k),
            "min_claim_score": str(self.abstention.min_claim_score),
        }


def load_config(path: Path | str | None = None) -> Config:
    """Read config.yaml. Falls back to the defaults above if it is missing.

    Every default is defined in this file, so the system still runs with no
    config.yaml at all - the file only overrides what it mentions.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return Config()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Config.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """The shared config, read from disk once.

    Cached because loading it repeatedly would be wasteful and could produce
    two different configs inside a single run. Call `get_config.cache_clear()`
    in a test if you need to reload it.
    """
    return load_config()
