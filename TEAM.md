# Team Split — three people

How the project is divided between three members: which files each person
owns, what each person presents in the demo, what each person still has to
build, and which sections of the paper each person writes.

The split follows the pipeline, so each person owns one stage end to end and
the hand-offs between people are the data contracts in `schema.py`:

```
 question
    │
    ▼
┌──────────────────────┐   ┌───────────────────────────┐   ┌──────────────────────────┐
│ MEMBER 1             │   │ MEMBER 2                  │   │ MEMBER 3                 │
│ Data & Retrieval     │──▶│ Generation & Verification │──▶│ Evaluation & Interface   │
│                      │   │                           │   │                          │
│ documents → sections │   │ passages → claims         │   │ Answer → screen          │
│ sections → chunks    │   │ claims → verdicts         │   │ Answer → metrics         │
│ question → passages  │   │ verdicts → Answer         │   │ adversarial question set │
└──────────────────────┘   └───────────────────────────┘   └──────────────────────────┘
   RetrievedChunk[]            Answer                          run logs, UI, paper
```

---

## Member 1 — Data & Retrieval

**Owns: getting the right policy text in front of the model.**

| Area | Files |
|---|---|
| Corpus definition | `data/manifest.yaml` |
| Ingestion | `src/policyverify/ingest/fetch.py`, `extract.py`, `normalize.py` |
| Chunking and index | `src/policyverify/indexing/chunk.py`, `store.py` |
| Retrieval | `src/policyverify/retrieve.py` |
| Scripts | `scripts/ingest.py`, `scripts/build_index.py`, `spikes/spike_corpus.py` |
| Tests | `tests/test_ingest.py`, `tests/test_indexing.py`, `tests/test_retrieve.py` |

**Must be able to explain in the demo**

- Why chunks follow section boundaries (a citation must point at one exact section).
- How citation IDs are made unique (`srm/attendance/7.3-minimum_attendance`), and
  the two real bugs behind that design: repeated numbering in the scholarship
  page, and 52 PDF headings buried mid-line.
- The per-section diversity cap in retrieval, and the measured case that motivated it.
- Why the index refuses to search if the embedding model changed.

**Still to build**

1. **Hybrid retrieval** — add BM25 keyword search beside the dense search and
   merge the two rankings. Dense-only search misses exact terms and identifiers.
2. **Roman-numeral headings** — `II. ADMISSION TO EXAMINATIONS` is not detected
   as a heading; add a pattern without false positives on the word "I".
3. **Second regulation document** — add another dated SRM regulation PDF to
   `data/manifest.yaml`. This makes the original "cross-regulation" trap possible.
4. **Heading noise** — a few address fragments and contents lines still become sections.
5. Re-check for a plagiarism policy newer than 2017.

**Paper sections:** III-A Corpus and Ingestion, III-B Indexing and Retrieval,
and the retrieval rows of the limitations section.

---

## Member 2 — Generation & Verification

**Owns: what the system says, and whether it is allowed to say it.**

| Area | Files |
|---|---|
| Language model client | `src/policyverify/llm.py` |
| Claim generation | `src/policyverify/generate.py` |
| The three checks | `src/policyverify/verify/nli.py`, `numeric.py`, `citation.py` |
| Combining the checks | `src/policyverify/verify/verifier.py` |
| Abstention | `src/policyverify/abstain.py` |
| The pipeline | `src/policyverify/pipeline.py` |
| Scripts | `scripts/ask.py`, `spikes/spike_nli.py`, `spikes/spike_vram.py` |
| Tests | `tests/test_generate.py`, `tests/test_verify.py`, `tests/test_abstain.py`, `tests/test_pipeline.py` |

**Must be able to explain in the demo**

- Why the model must return JSON claims rather than prose.
- The three checks and the combination rule: support needs every check to
  agree; the numeric guard can veto but never grant.
- Fabricated vs misused vs not-retrieved citations.
- Why removed claims are shown instead of deleted, and why abstaining is a
  result rather than an error.
- The Qwen3 thinking-mode trap (empty `response` with JSON mode).

**Still to build**

1. **Relevance check** — the most important open problem. Verification asks
   "is this claim supported by its citation?" but never "does it answer the
   question?". Add a check between generation and abstention that drops claims
   that do not address the question (see `PROJECT_STATUS.md` §8, the
   hostel-mess-cost example).
2. **Batch the NLI calls** — verification scores one (passage window, claim)
   pair at a time. Passing them to the model as one batch is the obvious speed-up.
3. **Scope check** — the NLI model can treat a postgraduate-only rule as
   applying to all students. Add a lightweight check beside the numeric guard.
4. **Apply the LLM timeout** — `llm.timeout_seconds` is in the config but is
   not passed to the Ollama client.

**Paper sections:** III-C Claim Generation, III-D Claim Verification,
III-E Abstention, and the verification rows of the limitations section.

---

## Member 3 — Evaluation & Interface

**Owns: proving it works, and showing it.**

| Area | Files |
|---|---|
| Adversarial question set | `eval/adversarial.jsonl`, `eval/citation_fingerprints.json` |
| Question-set schema and validation | `src/policyverify/evalset.py`, `scripts/validate_evalset.py` |
| Evaluation harness | `src/policyverify/evaluate.py`, `scripts/evaluate.py` |
| Interface | `app.py` |
| Demo script | `scripts/demo.py` |
| Tests | `tests/test_evalset.py`, `tests/test_evaluate.py` |
| Paper | `paper/` |

**Must be able to explain in the demo**

- The five trap categories and why 15 of 64 questions are unanswerable
  (otherwise over-refusal cannot be measured).
- Tuning half vs held-out half, and why numbers are reported on held-out only.
- What each metric means, and what "citation precision" does and does not claim.
- Drives the interface during the demo.

**Still to build**

1. **Full held-out run** — `python scripts/evaluate.py --split heldout` and
   put the numbers in the paper. On a CPU-only laptop this takes over an hour;
   run it on the machine with the GPU if possible.
2. **Threshold tuning** — sweep `min_claim_score` and `min_retrieval_score`
   over the *tuning* half, pick values, then report on held-out. They are
   currently reasoned defaults, not measured ones.
3. **Error analysis** — read every failure in the run log and group them by
   cause. This is the most valuable table in the paper.
4. **Baseline comparison** — run the same questions with verification off
   (keep every generated claim) so the paper can show what verification adds.
5. **Human-labelled claim precision** — label a sample of kept claims by hand
   as correct/incorrect. The current precision figure is a citation proxy.

**Paper sections:** IV Adversarial Evaluation Set, V Experimental Setup,
VI Results, and compiles the final document.

---

## Shared — nobody changes these alone

| File | Why it is shared |
|---|---|
| `src/policyverify/schema.py` | The data contracts between the three stages. A change here changes everyone's code. |
| `src/policyverify/config.py`, `config.yaml` | Every tunable for every stage. |
| `tests/test_schema.py`, `tests/conftest.py` | Tests and fixtures all three rely on. |
| `README.md`, `PROJECT_STATUS.md`, `CONCEPTS.md` | Each person updates the parts describing their own stage. |

**Paper sections written together:** Abstract, I Introduction, II Related Work,
VII Limitations, VIII Conclusion.

---

## Working rules

- One branch per task (`m1/hybrid-retrieval`, `m2/relevance-check`, …); merge
  through a pull request that another member reads.
- `pytest` and `python -m ruff check .` must pass before merging. The tests
  need no GPU, model or network, so there is no excuse.
- A change to `schema.py` or `config.py` is announced to the other two first.
- Corpus documents, `data/raw/`, `data/processed/`, `data/index/` and `.venv/`
  are never committed.
- After Member 1 changes ingestion or chunking, Member 3 re-runs
  `python scripts/validate_evalset.py` — the drift check reports any cited
  section whose text changed under the labels.

## Demo-day running order (about 10 minutes)

| Min | Who | What |
|---|---|---|
| 0–1 | Member 3 | The problem: a confident wrong answer about policy, with a citation attached, is worse than no answer. |
| 1–3 | Member 1 | Corpus, sections, citation IDs, retrieval. Show the "Retrieved passages" panel. |
| 3–6 | Member 2 | Ask the attendance question live. Open one claim's evidence. Explain the three checks. Then the 90% false-premise question. |
| 6–8 | Member 3 | Ask the wifi-password question: the system declines. Open the Evaluation tab and walk through the metrics. |
| 8–10 | All | Limitations (relevance is not checked; verification is slow) and what each person builds next. |

Start the app **before** the audience arrives — the first question after a
cold start pays the model-loading time.
