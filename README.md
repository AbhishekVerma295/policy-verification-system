# Policy Verification System

A question-answering system for university policies that **shows its work**.

Ask it something like *"What is the minimum attendance requirement?"* and it does
not just answer. It breaks its answer into separate factual claims, checks each
one against the actual policy text it retrieved, shows you the exact section
behind every claim, and **stays quiet when it cannot find real evidence**.

The point is the checking, not the answering. An ordinary chatbot will happily
tell you a policy says 80% when it really says 75%, and you would have no way to
know. This one is built to catch exactly that.

**Status:** Phase 0 complete — foundations, spikes, and corpus verified. See [Roadmap](#roadmap).

---

## Why this is interesting

Answering questions over documents is a solved-ish problem. Knowing when the
answer is *wrong* is not. This project treats verification as the main event:

- **Claim-level checking.** Answers are decomposed into individual factual
  claims. Each gets its own verdict, because a sentence where half is right and
  half is wrong cannot honestly get a single grade.
- **Two different citation failures, measured separately.** A citation pointing
  at a section that does not exist (fabrication) is a different bug from one
  pointing at a real section that does not support the claim (misuse). Most
  systems conflate them.
- **Abstention as a feature.** Refusing to answer is a correct output. So is
  refusing *too often* being counted as a failure — a system that abstains on
  everything scores perfectly on hallucination rate and is useless.
- **Cross-regulation traps.** The corpus is one institution ([SRM Institute of
  Science and Technology, Kattankulathur](https://www.srmist.edu.in/)) that
  publishes several dated, programme-specific regulation documents with
  genuinely different rules. "75% under the 2021 regulations, a different
  number for an earlier cohort" is the realistic, genuinely harmful error the
  system has to get right — the same shape of trap a multi-university corpus
  would give, without needing multiple institutions.

---

## How it works

```
question
   |
   v
[1] retrieve      find relevant policy passages    (Chroma, filterable by source)
   |
   v
[2] generate      draft an answer as structured    (Qwen3 via Ollama)
   |               claims, each with citations
   v
[3] verify        check every claim independently:
   |                - does the cited passage prove it?   (NLI model)
   |                - do the numbers agree?              (deterministic)
   |                - does the cited section exist?      (deterministic)
   v
[4] decide        keep supported claims, remove the rest,
   |               abstain entirely if too little survives
   v
answer + evidence + what was removed and why
```

Everything runs locally. No API keys, no cloud services, no per-token costs.

---

## Requirements

| | |
|---|---|
| Python | **3.11 or 3.12** (not 3.13+ — PyTorch lags new releases) |
| GPU | NVIDIA with ~6 GB+ VRAM (developed on an RTX 4060 laptop, 8 GB) |
| [Ollama](https://ollama.com) | for running Qwen locally |
| Disk | ~5 GB for models |

The embedding and fact-checking models run on **CPU by design**, so the whole
GPU stays free for Qwen.

---

## Setup

```bash
# 1. virtual environment on Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. CPU-only torch first (~200 MB instead of ~2.5 GB for the CUDA build —
#    Qwen runs through Ollama, so torch never needs the GPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. everything else
pip install -r requirements.txt

# 4. the language model
ollama pull qwen3:4b
```

Verify it worked:

```bash
pytest
```

36 tests should pass in well under a second. They need no GPU and no model —
that is deliberate, so the suite runs anywhere.

---

## Run the spikes first

A **spike** is a quick throwaway test that checks something works *before* you
build on top of it. All three exist to catch expensive problems in week 1
rather than week 6.

```bash
# 1. Can we get clean text out of these documents?
#    Test 5 candidate universities, keep the 3-4 that parse cleanly.
python spikes/spike_corpus.py <url1> <url2> <url3> <url4> <url5>

# 2. Is an NLI model good enough to be our fact-checker?
#    Runs 20 hand-written policy claim/passage pairs past 3 candidate models.
python spikes/spike_nli.py

# 3. Does everything fit in 8 GB of VRAM at once?
python spikes/spike_vram.py
```

**Choosing your universities based on which ones parse cleanly is a sensible
engineering decision, not cheating.** Nothing about this project requires any
particular institution, so there is no reason to fight a badly-built PDF.

---

## Project layout

```
src/policyverify/        the core library — no UI, no web framework
  schema.py              ★ the data contracts everything else depends on
  config.py              every setting, loaded from config.yaml
  ingest/                Phase 1: documents in, clean text out
  indexing/              Phase 2: chunking and the vector store
  retrieve.py            Phase 3: finding relevant passages
  llm.py                 Phase 3: the one place the model is called
  generate.py            Phase 3: asking for structured claims
  verify/                Phase 4: the fact-checking — the heart of the project
  abstain.py             Phase 5: deciding when to stay quiet
  pipeline.py            Phase 5: wiring it all together

app/                     Phase 7: Streamlit UI (imports the core, holds no logic)
scripts/                 command-line entry points
spikes/                  throwaway week-1 experiments
eval/                    the adversarial question set and results
tests/                   fast tests, no GPU required
data/manifest.yaml       the corpus recipe (URLs + checksums, not the PDFs)
config.yaml              every knob in one place
```

**The core library imports no Streamlit and no web framework.** Streamlit
re-runs its whole script on every click, so any logic living there would be
untestable. Keeping it out means the UI, the CLI and the evaluation harness all
run identical code and cannot drift apart.

---

## The corpus

**One institution — [SRM Institute of Science and Technology, Kattankulathur]
(https://www.srmist.edu.in/) — 6 documents**, one per policy type: attendance,
examinations, academic integrity, scholarships, residence and conduct. See
[`data/manifest.yaml`](data/manifest.yaml) for the exact sources and the
reasoning behind each pick.

This started as a 3–4 institution plan; narrowing to one institution was a
deliberate scope decision, not a shortcut. All 6 sources were run through
[`spikes/spike_corpus.py`](spikes/spike_corpus.py) before being adopted and
scored well on every extraction-quality check.

At this size, **curating by hand beats writing a scraper** — no crawler, no
robots.txt handling, no brittle site-specific parser. You download a handful of
files and inspect every extraction with your own eyes.

The repository stores a **manifest** — source URLs, access dates and checksums —
rather than the documents. That sidesteps the copyright question and makes the
corpus rebuildable from source, which is better engineering than a folder of
unverifiable binaries.

---

## Evaluation

Five metrics, no more:

| Metric | Question it answers |
|---|---|
| Claim verification P/R | Are "supported" claims really supported? |
| Citation accuracy | Does the cited section exist *and* support the claim? |
| Hallucination rate | What share of claims are not backed by the evidence? |
| Abstention quality | Does it refuse when it should — and **not** when it shouldn't? |
| Latency | Split across retrieve / generate / verify |

Measured against a hand-built adversarial set of 60–80 questions across five
trap categories: cross-regulation confusion (same institution, different
rule under a different year or programme — SRM publishes several dated
regulation versions, so this is a real, not contrived, trap), numeric traps,
unanswerable-but-plausible, negation and exceptions, and false premises.

The set is split into a **tuning half** and a **held-out half**. Thresholds are
tuned on the first and results reported on the second — tuning and reporting on
the same questions would only prove it works on the questions it was tuned for.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations, data contracts, three spikes, corpus locked | ✅ done |
| 1 | Ingestion — documents to clean `Document` objects | next |
| 2 | Chunking and the vector index | |
| 3 | Retrieval and structured generation | |
| 4 | **The verifier** — NLI, numeric guards, citation checks | |
| 5 | Abstention and transparent correction | |
| 6 | Adversarial question set | |
| 7 | Streamlit interface | |
| 8 | Evaluation harness | |
| 9 | Hardening, error analysis, demo | |

**Phase 0 spike results:**

- **Corpus** — all 6 SRM sources scored GOOD on extraction quality (`spikes/spike_corpus.py`, 2026-08-14)
- **NLI** — `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` won: 90% overall, **100% on numeric traps** (the "75% vs 80%" errors that matter most here). Known weak spot to carry into Phase 4: scope confusion — e.g. mistaking a postgraduate-only rule for one that applies to all students. Config already set to this model.
- **VRAM** — peak 3.9 GB of 8 GB with Qwen3 4B, embedder and NLI all loaded together. 4.25 GB headroom; qwen3:8b is a safe upgrade later.

---

## Design notes

A few decisions worth explaining, since they are the ones people ask about:

**Why a separate NLI model instead of asking Qwen to check itself?** A model
grading its own work tends to repeat its own mistakes — the same misreading that
produced the bad claim produces a confident bad grade. An NLI model answers one
narrow question ("does text A prove text B?") independently.

**Why structured JSON claims instead of prose?** Generating prose and then
parsing claims back out of it with regexes breaks constantly, because prose has
no rules. Making the format structural turns "usually works" into "works, or
fails loudly."

**Why temperature 0?** So the same question gives the same answer. A system
whose output changes between runs cannot be debugged or evaluated.

**Why record which embedding model built the index?** Rebuild with a different
model and forget, and search silently returns nonsense with no error at all. It
is the most common bug in systems like this one.

---

## License

MIT for the code. The policy documents themselves are **not** redistributed —
only their URLs — and remain the property of SRM Institute of Science and
Technology.
