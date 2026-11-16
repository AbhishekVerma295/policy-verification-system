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

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

# The project root is two levels up from this file:
#   <root>/src/policyverify/config.py  ->  <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
# Environment variable that overrides llm.model - see load_config().
LLM_MODEL_ENV = "POLICYVERIFY_LLM_MODEL"


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

    # At most this many chunks from any one section may appear in the results.
    #
    # Without a cap, a single oversized section crowds everything else out: it
    # was split into many chunks, they all score similarly, and they take every
    # slot. Measured on the real corpus - asking "what is the minimum
    # attendance requirement to sit the final examination?" filled the whole
    # top 3 with chunks of one examination section and pushed the actual
    # attendance rule down to rank 4.
    max_chunks_per_citation: int = 2
    # How many extra candidates to pull before applying that cap, as a
    # multiple of top_k. Needs to be > 1 or capping would just leave gaps.
    oversample_factor: int = 4

    # Hybrid search: run a keyword (BM25) search alongside the embedding
    # search and merge the two rankings. Embeddings alone are touchy about
    # wording on this corpus - "What is minimum marks needed to pass a
    # course ?" ranked the section that says "To 'Pass' a course ... 50%"
    # about 10th. See indexing/bm25.py and retrieve.py. False = embedding
    # search only, exactly as before.
    hybrid_search: bool = True
    # The constant k in Reciprocal Rank Fusion, the merge rule: each search
    # gives a passage 1 / (rrf_k + its rank) and the two are added. A larger
    # value flattens the difference between rank 1 and rank 10, so agreement
    # between the two searches matters more than either one's top pick. 60 is
    # the value from the original paper (Cormack et al., SIGIR 2009) and is
    # the usual default.
    rrf_k: int = 60
    # How many candidates each search contributes before merging. Large
    # enough that a passage one search ranks badly can still be rescued by
    # the other; the per-section cap and top_k are applied after the merge.
    hybrid_candidate_pool: int = 50
    # BM25's two knobs. k1: how quickly repeating a word stops adding to the
    # score (higher = repetition keeps counting for longer). b: how much
    # long passages are penalised for having more words (0 = not at all,
    # 1 = fully). 1.5 and 0.75 are the textbook defaults.
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


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
    # Hard ceiling so one bad question cannot hang the whole app. Generous,
    # because on a machine with no GPU a normal answer takes over a minute.
    timeout_seconds: int = 300
    num_ctx: int = 8192
    # The most tokens the model may write in one reply. A normal answer is a
    # few hundred; the prompt asks for at most 6 claims. Without a cap, a
    # model in JSON mode can fall into emitting output forever - measured: one
    # adversarial question ran for 25 minutes and never returned. It was 800,
    # which cut real replies off mid-claim (an answer of many long claims runs
    # past it); 1500 leaves room for six full claims. A reply that is still
    # cut off is no longer thrown away: generate.py keeps every claim that was
    # completed before the cut.
    max_tokens: int = 1500
    # The most characters of passage text put in one prompt. Whole passages
    # are shown (never cut in the middle); if the total would be larger than
    # this, the lowest-ranked passages are left out instead. 16000 characters
    # is roughly 4000 tokens, which with the instructions and a full-length
    # reply (max_tokens) still fits inside num_ctx.
    max_passage_chars: int = 16000

    # Qwen3 is a hybrid reasoning model: it can emit "thinking" tokens before
    # its answer, and Ollama returns those in a SEPARATE field. Combined with
    # JSON mode the whole reply lands in `thinking` and `response` comes back
    # empty - the model looks broken while actually working fine.
    #
    # Turning thinking off is what makes structured output usable here. It is
    # a setting rather than a hardcoded value because a non-Qwen backend will
    # not need it. Measured: 3/3 valid JSON with this off, 0/3 with it on.
    disable_thinking: bool = True
    # How many times to re-ask when the model returns unparseable output.
    # One retry, then fail loudly - silently accepting malformed output is
    # exactly the kind of quiet wrongness this project exists to prevent.
    max_parse_retries: int = 1
    # How long Ollama keeps the model in memory after a request. Its default
    # is 5 minutes, after which the next question pays the full model load
    # again - a long pause in an interactive session.
    keep_alive: str = "30m"


class NLIConfig(BaseModel):
    """The fact-checker: does this passage actually prove this claim?

    Also on CPU, for the same reason as the embedding model. The exact
    checkpoint is chosen by running spikes/spike_nli.py - do not assume this
    default is the best one until you have measured it.
    """

    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    device: str = "cpu"
    # An entailment score below this is not treated as support. Checked with
    # scripts/calibrate.py on the 44 tuning claims (Nov 2026): the scores are
    # nearly all near 0 or near 1 (19 below 0.2, 23 above 0.8, 2 between),
    # so any value from 0.3 to 0.5 accepts exactly the same claims, and
    # higher values only start trading true claims for the single false one
    # in the 0.6-0.7 band. Left at 0.5; see abstention.min_claim_score.
    entailment_threshold: float = 0.5
    # How strongly one cited section must contradict a claim, while another
    # cited section supports it, before the claim is withheld as "the sources
    # disagree". Stricter than entailment_threshold on purpose: a weak
    # contradiction from a loosely related section is common and harmless,
    # and treating it as a conflict would throw away well-supported claims.
    # Only applies across DIFFERENT citation IDs - see verifier.py.
    conflict_threshold: float = 0.9
    # The claim is scored against single sentences of the passage and against
    # runs of up to this many neighbouring sentences. 1 = sentences only.
    # Measured on the tuning claims: 1 missed a true claim that joins two
    # sentences (recall 87.5%); 2 and 3 both reached 91.7%. 3 is kept because
    # it is what the SummaC-style probe validated, and it cost under 1 s/claim.
    max_window_sentences: int = 3
    # How many (sentence, claim) pairs the model scores at once. Batching
    # amortises the per-call overhead across many short pairs.
    batch_size: int = 16


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

    # A SUPPORTED claim is kept only if its score is at least this. Checked
    # with scripts/calibrate.py on the tuning claims (Nov 2026) and left at
    # 0.5: 23 of the 25 accepted tuning claims score above 0.8; raising it to
    # 0.7 would drop the one false claim accepted (scored 0.64) but also a
    # true one (0.53), and replaying the latest tuning run, it would turn one
    # correct answer (num-016, best claim 0.69) into a refusal while the one
    # hallucinated answer scored 0.96 and would stay. One claim either way is
    # not evidence enough to move it. Equal to nli.entailment_threshold, so
    # in practice it is that threshold that decides.
    min_claim_score: float = 0.5
    # If fewer than this many claims survive checking, abstain entirely rather
    # than giving a half-answer.
    min_supported_claims: int = 1
    # RETIRED - no longer read by anything. It was "decline if the best
    # retrieved passage scores below this"; measured, it never fired and no
    # value separates answerable from unanswerable questions (see the note
    # in abstain.decide). Kept only so config.yaml files that still set it
    # load without error; the line can be deleted from config.yaml.
    min_retrieval_score: float = 0.25


class RelevanceConfig(BaseModel):
    """The relevance gate: does a verified claim answer the question asked?

    Verification proves a claim is TRUE of its passage; it never asks whether
    the claim is what the user wanted. "Who is the current Vice Chancellor?"
    -> "The Vice-Chancellor will deliver the welcome address" is true, cited
    and on a nearby topic, and no use as an answer. The gate computes how
    close the verified claims are to the question and, unless that is clear
    either way, asks the language model one yes/no question.

    All three thresholds were chosen on the TUNING questions and the fresh
    questions only - never the held-out half - so they are somewhat
    optimistic for those same questions. See verify/relevance.py.

    Re-checked in Nov 2026 by replaying the latest tuning run: every answer
    that reached the gate had a best similarity between 0.565 and 0.811, so
    the language model decided all of them and neither cut-off was used.
    Two unanswerable questions the judge rightly rejected scored 0.75 and
    0.81, so lowering `similarity_accept` would let them through; raising
    `similarity_reject` past 0.59 would block a correct answer. The fallback
    was never needed in that run, so there is nothing to tune it on. All
    three are left as they were.
    """

    # False = gate off: every verified claim counts as relevant, exactly as
    # before the gate existed.
    enabled: bool = True
    # Cosine similarity (question with the retrieval query prefix, against the
    # claim text) at or above which the answer is on topic without asking the
    # model. Deliberately high: on the replayed answers, off-topic claims that
    # merely repeat the question's words (e.g. "the late fee is calculated
    # from the date of payment", asked for the fee amount) reached 0.81, so
    # anything under about 0.85 still needs the model's opinion.
    similarity_accept: float = 0.85
    # Similarity below which the claim is off topic without asking the model.
    # No good answer scored under 0.574; the lowest off-topic ones were far
    # lower still, so this saves a model call and a mistake on the clear cases.
    similarity_reject: float = 0.55
    # Between the two thresholds the model is asked. If it cannot be used
    # (unreachable, unusable reply) the answer is judged on similarity alone
    # against this single cut-off, and the Answer records that the fallback
    # happened. Low on purpose: similarity cannot tell a true-but-off-topic
    # claim from a good one in the 0.6-0.8 band, and a cut-off of 0.65
    # blocked 5 of 52 good answers while catching no more than 0.60 did.
    similarity_fallback: float = 0.60
    # Longest claim list shown to the model, most similar first, so the prompt
    # stays short.
    judge_max_claims: int = 6


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
    relevance: RelevanceConfig = Field(default_factory=RelevanceConfig)

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
        config = Config()
    else:
        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        config = Config.model_validate(raw)

    # One override from the environment: which language model to run. The
    # right model depends on the machine (a laptop with no GPU wants a smaller
    # one), and that should not require editing a committed file.
    #     set POLICYVERIFY_LLM_MODEL=qwen2.5:3b
    model = os.environ.get(LLM_MODEL_ENV, "").strip()
    if model:
        config.llm.model = model
    return config


@lru_cache(maxsize=1)
def get_config() -> Config:
    """The shared config, read from disk once.

    Cached because loading it repeatedly would be wasteful and could produce
    two different configs inside a single run. Call `get_config.cache_clear()`
    in a test if you need to reload it.
    """
    return load_config()
