# Policy Verification System

A question-answering system for university policies that **shows its work**.

Ask it *"What is the minimum attendance requirement?"* and it does not simply
answer. It breaks its answer into separate factual claims, checks each one
against the policy text it actually retrieved, shows the exact section behind
every claim, and stays quiet when it cannot find real evidence.

The point is the checking, not the answering. An ordinary chatbot will happily
tell you a policy says 80% when it really says 75%, and you would have no way
to know. This system is built to catch exactly that — and it does:

```
CLAIM  "A student must maintain at least 75% attendance"   SUPPORTED  1.00
                                                 -> srm/attendance/7.3-minimum_attendance

CLAIM  "A student must maintain at least 80% attendance"   REFUTED    1.00
       The cited policy text contradicts this claim.

CLAIM  "Students who miss classes must pay a fine"         NEUTRAL    0.00
       The cited text neither supports nor contradicts this.
```

Everything runs locally. No API keys, no cloud services, no per-token costs.

**Status:** Phases 0–4 complete. Retrieval, generation and verification all
work end-to-end. Next is Phase 5 (abstention). See [Roadmap](#roadmap).

---

## Contents

- [Why this is interesting](#why-this-is-interesting)
- [How it works](#how-it-works)
- [Setup](#setup)
- [Usage](#usage)
- [Architecture](#architecture)
- [The corpus](#the-corpus)
- [Design decisions](#design-decisions)
- [Evaluation plan](#evaluation-plan)
- [Roadmap](#roadmap)
- [Development](#development)

Further reading in this repository:

| Document | What it covers |
|---|---|
| [`CONCEPTS.md`](CONCEPTS.md) | The ideas behind the system — RAG, embeddings, NLI, abstention — explained from scratch |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Current state module by module, every finding, and what is not built yet |
| [`data/manifest.yaml`](data/manifest.yaml) | The corpus: every source URL, with the reasoning for each pick |

---

## Why this is interesting

Answering questions over documents is a mostly-solved problem. Knowing when
the answer is *wrong* is not. This project treats verification as the main
event rather than a postscript.

**Claim-level checking.** Answers are decomposed into individual factual
claims, and each gets its own verdict. A sentence where half is right and half
is wrong cannot honestly receive a single grade.

**Two citation failures, measured separately.** A citation pointing at a
section that does not exist (*fabrication*) is a different bug from one
pointing at a real section that does not support the claim (*misuse*). The
first is unrecoverable; the second is a grounding problem better retrieval
could fix. Most systems report both as "bad citation" and lose that
distinction.

**Abstention as a feature.** Refusing to answer is a correct output. So is
over-refusal being counted as a failure — a system that abstains on everything
scores perfectly on hallucination rate and is useless.

**Numbers are where the damage is.** "75% attendance" versus "80% attendance"
is the difference between sitting an exam and being barred from it, and the
two sentences are otherwise identical. A deterministic numeric guard runs
alongside the model for exactly this reason.

---

## How it works

```
question
   │
   ▼
[1] RETRIEVE ──── find relevant policy passages
   │              Chroma vector search + per-section diversity cap
   ▼
[2] GENERATE ──── draft the answer as structured claims, each with citations
   │              Qwen3 4B via Ollama, JSON validated against a schema
   ▼
[3] VERIFY ────── check every claim independently:
   │                • does the cited passage prove it?      NLI model
   │                • do the numbers agree?                 deterministic
   │                • does the cited section exist?         deterministic
   ▼
[4] DECIDE ────── keep supported claims, remove the rest,       ← Phase 5
   │              abstain entirely if too little survives
   ▼
answer + evidence + what was removed and why
```

Steps 1–3 are built and working. Step 4 is next.

---

## Setup

| Requirement | Notes |
|---|---|
| Python | **3.11 or 3.12** — not 3.13+, PyTorch lags new releases |
| GPU | NVIDIA, ~6 GB+ VRAM (developed on an RTX 4060 laptop, 8 GB) |
| [Ollama](https://ollama.com) | runs Qwen locally |
| Disk | ~5 GB for models |

The embedding and fact-checking models run on **CPU by design**, so the whole
GPU stays free for Qwen. Measured peak with all three loaded: 3.9 GB of 8 GB.

```bash
# 1. virtual environment on Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. CPU-only torch first — ~200 MB instead of ~2.5 GB for the CUDA build.
#    Qwen runs through Ollama, so torch never needs the GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. everything else
pip install -r requirements.txt

# 4. the language model
ollama pull qwen3:4b
```

Verify the install:

```bash
pytest
```

**137 tests** should pass in under a second. They need no GPU, no model and no
network — deliberately, so the suite runs anywhere.

---

## Usage

Build the corpus and index (first run downloads the embedding model):

```bash
python scripts/ingest.py        # fetch 6 policy documents → 192 sections
python scripts/build_index.py   # chunk and embed          → 258 chunks
```

Ask a question (needs Ollama running):

```bash
python scripts/ask.py "What is the minimum attendance requirement?"
python scripts/ask.py "Can a parent stay overnight?" --policy residence
python scripts/ask.py "..." -k 8 --show-passages
```

Output is per claim, with a verdict, a confidence, the citation, and a
plain-language explanation of *why* that verdict was reached.

---

## Architecture

```
src/policyverify/        the core library — no UI, no web framework
  schema.py              ★ data contracts everything else depends on
  config.py              every setting, loaded from config.yaml
  ingest/                documents in, clean Document objects out
    fetch.py               download from the manifest, checksum them
    extract.py             PDF/HTML → plain text
    normalize.py           text → sections, by detecting headings
  indexing/
    chunk.py               Document → Chunks, on section boundaries
    store.py               Chroma wrapper + index/model mismatch guard
  retrieve.py            vector search + per-section diversity cap
  llm.py                 the one place a language model is called
  generate.py            prompt building, JSON claim parsing
  verify/                ★ the heart of the project
    nli.py                 does the passage prove the claim?
    numeric.py             do the numbers agree?
    citation.py            is the citation real? was it shown to the model?
    verifier.py            combines the three into one verdict

scripts/                 ingest.py, build_index.py, ask.py
spikes/                  throwaway experiments that de-risked decisions
tests/                   137 tests, no GPU or network needed
data/manifest.yaml       the corpus recipe (URLs + checksums, not the PDFs)
config.yaml              every tunable knob in one place
```

**The core library imports no Streamlit and no web framework.** Streamlit
re-runs its whole script on every interaction, so logic living there would be
untestable. Keeping it out means the future UI, the CLI, and the evaluation
harness all run identical code and cannot drift apart.

### The citation contract

Every citation is `{university}/{policy}/{section}`, for example:

```
srm/attendance/7.3-minimum_attendance
```

The section part combines the document's own numbering with a slug of its
heading. Both halves are needed: real documents restart their numbering (SRM's
scholarship page has several independent numbered lists, so "2" appears six
times meaning six different things) and repeat their headings (the Code of
Conduct has six separate sections all titled "Consequence"). A citation
pointing at six different passages makes *"does the cited section support this
claim?"* unanswerable — so `Document.citations()` guarantees uniqueness,
adding an occurrence suffix where needed.

---

## The corpus

**One institution — [SRM Institute of Science and Technology, Kattankulathur](https://www.srmist.edu.in/)
— 6 documents**, one per policy type: attendance, examinations, academic
integrity, scholarships, residence, and conduct. Currently **192 sections**
and **258 indexed chunks**.

Every source was run through `spikes/spike_corpus.py` before adoption and
scored well on extraction quality. At this size, **curating by hand beats
writing a scraper** — no crawler, no robots.txt handling, no brittle
site-specific parser, and you can inspect every extraction yourself.

The repository stores a **manifest** (source URLs, access dates, checksums)
rather than the documents. That sidesteps the copyright question and makes the
corpus rebuildable from source, which is better engineering than a folder of
unverifiable binaries. See [`data/manifest.yaml`](data/manifest.yaml).

---

## Design decisions

**Why a separate NLI model instead of asking Qwen to check itself?**
A model grading its own work repeats its own mistakes — the same misreading
that produced the bad claim produces a confident bad grade. An NLI model
answers one narrow question ("does text A prove text B?") independently, runs
on CPU, and is deterministic.

**Why structured JSON claims instead of prose?**
Generating a paragraph and then pulling claims back out of it with regexes
fails constantly, because prose has no rules. Making the format structural
turns "usually works" into "works, or fails loudly". You cannot verify a claim
you could not reliably identify.

**Why can the numeric guard veto support but never grant it?**
Matching digits is not proof that a passage *means* what the claim says. Only
the NLI model can grant support; only the numeric check can overrule it on
numbers.

**Why does support require every check to agree?**
Wrongly marking a claim supported puts a false statement in front of a student
with a citation attached. Wrongly withholding a true one is visible and
recoverable. The asymmetry is deliberate.

**Why are long passages windowed rather than truncated?**
The NLI model caps at 512 tokens and chunks run to ~1,950 characters. Letting
the tokenizer truncate could silently discard the very sentence that supports
the claim — precisely the silent failure this project exists to prevent.

**Why temperature 0?**
So the same question gives the same answer. A system whose output changes
between runs cannot be debugged or evaluated.

**Why record which embedding model built the index?**
Rebuild with a different model and forget, and search silently returns
nonsense with no error at all. It is the most common bug in systems like this,
so the store refuses to search a mismatched index rather than guessing.

---

## Evaluation plan

Five metrics, no more:

| Metric | Question it answers |
|---|---|
| Claim verification P/R | Are "supported" claims really supported? |
| Citation accuracy | Does the cited section exist *and* support the claim? |
| Hallucination rate | What share of claims are not backed by the evidence? |
| Abstention quality | Does it refuse when it should — and **not** when it shouldn't? |
| Latency | Split across retrieve / generate / verify |

To be measured against a hand-built adversarial set of 60–80 questions across
five trap categories: cross-regulation confusion (SRM publishes several dated
regulation versions, so the same question has a different right answer by year
or programme), numeric traps, unanswerable-but-plausible, negation and
exceptions, and false premises.

The set will be split into a **tuning half** and a **held-out half**.
Thresholds get tuned on the first and results reported on the second — tuning
and reporting on the same questions would only prove it works on the questions
it was tuned for.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations, data contracts, spikes, corpus | ✅ done |
| 1 | Ingestion — documents to `Document` objects | ✅ done |
| 2 | Chunking and the vector index | ✅ done |
| 3 | Retrieval and structured claim generation | ✅ done |
| 4 | **The verifier** — NLI, numeric guards, citation checks | ✅ done |
| 5 | Abstention and transparent correction | next |
| 6 | Adversarial question set | |
| 7 | Streamlit interface | |
| 8 | Evaluation harness | |
| 9 | Hardening, error analysis, demo | |

A phase-by-phase record of what was built, what broke, and what was measured
lives in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

---

## Development

```bash
pytest                    # 137 tests, no GPU or network
python -m ruff check .    # lint
python -m ruff check . --fix
```

Tests use a `FakeLLM` fixture and a stub NLI checker, so nothing in the suite
downloads a model or calls Ollama. Model *quality* is measured by the
evaluation harness against the adversarial set, not asserted in unit tests —
those check the plumbing and the decision logic.

### Spikes

A **spike** is a quick throwaway experiment that checks something works
*before* you build on top of it. These are kept for the record; they are not
part of the pipeline.

```bash
python spikes/spike_corpus.py <url>   # will this document extract cleanly?
python spikes/spike_nli.py            # which NLI checkpoint is best?
python spikes/spike_vram.py           # does everything fit in 8 GB?
```

---

## License

MIT for the code. The policy documents themselves are **not** redistributed —
only their URLs — and remain the property of SRM Institute of Science and
Technology.
