# Project Status

Detailed current state of the Policy Verification System. Written as a handoff
document: it states what exists, what was measured, and what is not built yet.

**Last updated:** after Phase 5 (abstention), 2026-09-13.

For the project overview see [`README.md`](README.md). For the ideas behind
it see [`CONCEPTS.md`](CONCEPTS.md).

---

## 1. Summary

A local question-answering system over SRM Institute of Science and Technology
(Kattankulathur) policies. It breaks each answer into individual factual
claims, verifies every claim against the policy text it cites, reports a
verdict and confidence per claim, and will abstain when evidence is
insufficient.

| | |
|---|---|
| **Phases complete** | 0, 1, 2, 3, 4, 5 |
| **Next** | Phase 6 — adversarial question set |
| **Tests** | 168 passing, no GPU / model / network required |
| **Lint** | ruff clean |
| **Corpus** | 6 documents, 192 sections, 258 indexed chunks |
| **Working end-to-end** | question → retrieve → generate → verify → decide → `Answer` |
| **Not yet working** | adversarial set, UI, evaluation harness |

---

## 2. Environment

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.12 in `.venv/` (3.13+ not supported — PyTorch lags) |
| GPU | RTX 4060 Laptop, 8 GB VRAM |
| RAM | 16 GB |
| Generation model | `qwen3:4b` via Ollama (local, GPU) |
| Embedding model | `BAAI/bge-base-en-v1.5`, 768-dim (CPU) |
| NLI model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (CPU) |
| Vector store | Chroma, persisted to `data/index/` |
| Repo | `github.com/AbhishekVerma295/policy-verification-system` |

Measured VRAM with all three models loaded simultaneously: **3.9 GB of 8 GB**.
The embedding and NLI models are pinned to CPU on purpose so the whole GPU
stays available for Qwen.

`torch` is installed as the **CPU-only** build (~200 MB rather than ~2.5 GB).
Qwen runs through Ollama, which has its own runtime, so torch never needs CUDA.

---

## 3. Corpus

One institution, six documents, one per policy type.

| Policy type | Source document | Format | Sections | Chunks |
|---|---|---|---|---|
| `attendance` | Academic Regulations 2021 (UG / Integrated PG) | PDF | 52 | 67 |
| `examination` | Examination Policy | HTML | 26 | 71 |
| `academic_integrity` | Plagiarism Policy (2017) | PDF | 9 | 11 |
| `scholarship` | Scholarship Policy | HTML | 41 | 49 |
| `residence` | SRMIST Hostels Rules and Regulations (Jan 2025) | PDF | 50 | 40 |
| `code_of_conduct` | Code of Conduct for Students | HTML | 14 | 20 |
| | | | **192** | **258** |

Source URLs, access dates, checksums and the reasoning behind each pick are in
[`data/manifest.yaml`](data/manifest.yaml). The documents themselves are **not
committed** — only the manifest. `python scripts/ingest.py` re-downloads them.

**Scope note:** this began as a 3–4 institution plan. Narrowing to one
institution was a deliberate decision. The cost was losing "cross-university
confusion" as the signature adversarial trap; the replacement is
cross-*regulation* confusion, since SRM publishes several dated,
programme-specific regulation documents with genuinely different rules.

---

## 4. What is built

### 4.1 Data contracts — `schema.py`

The foundation; everything imports from here. Pure Pydantic models, no I/O.

| Model | Purpose |
|---|---|
| `CitationID` | `{university}/{policy}/{section}`, frozen, strictly validated |
| `Section`, `Document` | What ingestion produces |
| `Chunk`, `RetrievedChunk` | What indexing and retrieval produce |
| `Claim`, `DraftAnswer` | The structural contract the LLM must satisfy |
| `CheckResults`, `ClaimVerdict` | Verification output |
| `Answer`, `Timings` | The final response object, populated by `pipeline.py` |
| `PolicyType`, `SourceFormat`, `VerdictStatus` | `StrEnum`s, so they serialise as plain strings |

`Document.citations()` is the authoritative way to cite sections. It assigns a
unique citation per section, because `Section.section_key()` alone cannot
guarantee uniqueness — it only ever sees one section at a time.

### 4.2 Configuration — `config.py` + `config.yaml`

One validated file for every tunable: paths, embedding model and device,
chunking sizes, retrieval top-k and diversity cap, LLM model/temperature/
thinking, NLI model and threshold, verification flags, abstention thresholds.

Every setting has a Pydantic default, so the system runs with no `config.yaml`
at all — the file only overrides what it mentions.
`Config.fingerprint()` summarises the settings that affect results, for
recording alongside runs.

### 4.3 Ingestion — `ingest/` (run by `scripts/ingest.py`)

1. **`fetch.py`** — downloads each document listed in the manifest, saves raw
   bytes to `data/raw/`, records SHA-256 checksums in `data/raw/checksums.json`
   so a later re-fetch detects if the university silently changed a document.
   Checksums live in a separate generated file rather than being written back
   into `manifest.yaml`, because round-tripping that hand-commented file
   through a YAML library would strip every comment.
2. **`extract.py`** — raw bytes to plain text.
   - HTML: BeautifulSoup; strips `<script>`/`<style>`/`<nav>`/`<footer>` plus
     SRM-specific chrome (sidebar policy menu, mega-menu, breadcrumb) that is
     **not** in semantic tags and was found by inspecting the real DOM.
   - PDF: PyMuPDF; strips lines repeating 3+ times verbatim, which removes
     running page headers and footers.
3. **`normalize.py`** — splits text into `Section` objects by detecting three
   heading styles: numbered (`4.2 Minimum Attendance`), ALL-CAPS
   (`REGISTRATION AND ENROLLMENT:`), and numbered headings buried mid-line by
   PDF extraction, which are lifted onto their own line first.

### 4.4 Indexing — `indexing/` (run by `scripts/build_index.py`)

- **`chunk.py`** — Documents to Chunks on section boundaries. A chunk never
  spans two sections. Oversized sections split into overlapping windows that
  all keep the same citation and differ only in `chunk_id`. Contents-page
  noise (entries whose whole body is a page number) is dropped.
- **`store.py`** — Chroma wrapper. Writes `data/index/index_manifest.json`
  recording which embedding model built the index, and **refuses to search**
  if the configured model no longer matches.

  What gets **embedded** includes university, policy type and section heading;
  what gets **stored as chunk text** does not. Search benefits from the
  context; verification must check claims against the university's words
  alone, not a string we assembled.

### 4.5 Retrieval and generation — Phase 3

- **`retrieve.py`** — dense search with optional university/policy filters,
  plus `diversify()`, which caps how many chunks any single section may
  contribute (default 2).
- **`llm.py`** — the single place a language model is called.
  `generate(prompt) -> str`, `LLMBackend` protocol, `OllamaBackend`
  implementation. Adding a hosted backend later is one branch in `get_llm()`.
- **`generate.py`** — builds the prompt, requires JSON claims validated
  against `DraftAnswer`, retries once on unparseable output then raises.
  `drop_unknown_citations()` separates citations matching no retrieved
  passage and **reports** them as fabricated rather than discarding them.

  The prompt states explicitly that passages are reference material and any
  instructions inside them must be ignored — a prompt-injection defence,
  since the system reads documents it did not write.

### 4.6 The verifier — `verify/` (Phase 4)

Three independent checks per claim, combined by `verifier.py`.

- **`nli.py`** — `NLIChecker.check(premise, hypothesis)` returns
  entailment / neutral / contradiction with probabilities. Long passages are
  split into overlapping windows via `split_windows()` and each is scored;
  support anywhere in the passage counts as support.
- **`numeric.py`** — `check_numbers(claim, premises)` extracts numeric values
  (handling `75%`, `75 %`, `1,200`, `3.0`) and reports any number in the claim
  absent from the evidence. Ignores small bare values (0, 1, 2) that are list
  markers rather than asserted facts.
- **`citation.py`** — `check_citation()` distinguishes **fabricated** (exists
  nowhere in the corpus) from **misused** (real section, wrong attribution)
  from **not retrieved** (real but never shown to the model). A store failure
  is explicitly *not* reported as fabrication — that would blame the model for
  our own outage.
- **`verifier.py`** — combines them into one `ClaimVerdict`.

**Combination rules:**

- Support requires *every* check to agree; any single objection withholds it.
- The numeric guard can **veto** support but never **grant** it.
- A claim with no citation, or only fabricated citations, cannot be supported.
- A fabricated citation alongside a valid one does not block the valid one;
  it is noted in the explanation.

### 4.7 Abstention and the pipeline — Phase 5

- **`abstain.py`** — two separate decisions. `should_keep()` decides which
  claims survive (SUPPORTED *and* above `min_claim_score`; NEUTRAL means the
  evidence did not settle it, which is not the same as true). `decide()`
  partitions the verdicts and decides whether enough survived to answer at
  all, checking retrieval strength first — if the corpus does not cover the
  question, any claims built on it were not grounded in anything worth
  standing behind.

  `AbstentionDecision.refuted` is exposed separately from `.unsupported`,
  because "the policy says the opposite" is a more serious finding than "the
  policy does not say" and is the most useful thing to show a reader.

  **Nothing generated is ever discarded.** Every claim ends up in `kept` or
  `removed`, including when the system abstains. A test asserts this
  invariant directly, since it is easy to break when adding a branch.

- **`pipeline.py`** — `answer_question()` runs retrieve → generate → verify →
  decide, times each stage, and returns a populated `Answer`. The store, LLM
  and NLI checker are all injectable, so callers can load models once and
  reuse them, and tests can substitute fakes and run the whole path in
  milliseconds.

  An abstention returns a normal `Answer` with `abstained=True` and a reason —
  it is a result, not an error, so callers do not have to catch anything.

### 4.8 Scripts

| Script | Does |
|---|---|
| `scripts/ingest.py` | manifest → downloaded → extracted → `Document` JSON |
| `scripts/build_index.py` | processed documents → chunks → embedded index, then a smoke search |
| `scripts/ask.py` | question → retrieval → claims → verdicts, printed per claim |

### 4.9 Tests — 168 total

| File | Tests | Covers |
|---|---|---|
| `test_schema.py` | 40 | Citation format, uniqueness, validation, serialisation |
| `test_verify.py` | 30 | Numeric guard, citation resolution, verdict combination |
| `test_ingest.py` | 21 | HTML/PDF extraction, heading detection, section splitting |
| `test_generate.py` | 20 | Prompt building, JSON parsing, retries, fabrication reporting |
| `test_indexing.py` | 19 | Chunk boundaries, noise filtering, index mismatch guard |
| `test_abstain.py` | 18 | Keep/remove logic, abstention triggers, the no-claim-lost invariant |
| `test_pipeline.py` | 13 | End-to-end wiring, timings, transparent correction |
| `test_retrieve.py` | 7 | Diversity cap behaviour |

Runs in ~0.3s with no GPU, model or network. `conftest.py` provides `FakeLLM`;
`test_verify.py` and `test_pipeline.py` provide stub NLI checkers and a stub
store. Model quality is measured by the
evaluation harness (Phase 8), not asserted here — these test plumbing and
decision logic.

---

## 5. Verified behaviour

### End-to-end, real corpus

`python scripts/ask.py "What is the minimum attendance requirement to sit the final examination?"`

```
retrieved 6 passages
  [0.632] srm/examination/detention_cancellation_of_candidature_fo
  [0.628] srm/examination/detention_cancellation_of_candidature_fo
  [0.619] srm/attendance/7.3-minimum_attendance
  ...

CLAIM 1  [SUPPORTED 0.99]  "A student must maintain a minimum attendance
         record of at least 75% in individual courses..."
         -> srm/attendance/7.3-minimum_attendance

CLAIM 2  [SUPPORTED 0.98]  "Without the minimum attendance of 75%, students
         become ineligible to appear for the end semester examination."
         -> srm/attendance/7.3-minimum_attendance

CLAIM 3  [SUPPORTED 0.55]  "Students with less than 75% attendance ... awarded
         'I' Grade..."
         -> srm/attendance/7.4-attendance_shortage_and_examination
```

### Adversarial checks with the real models

| Claim | Verdict | Correct |
|---|---|---|
| "must maintain at least **75%** attendance" | SUPPORTED 1.00 | ✓ |
| "must maintain at least **80%** attendance" | **REFUTED 1.00** | ✓ |
| "students who miss classes must pay a fine of 500 rupees" | NEUTRAL 0.00 | ✓ |
| claim citing `srm/attendance/99-does-not-exist` | NEUTRAL, **fabricated** | ✓ |

Asking *"What is the campus wifi password?"* produced **zero claims** — the
model correctly declined rather than inventing an answer.

### Latency (measured, cold model load excluded)

| Stage | Time |
|---|---|
| Retrieval | ~1s warm (~12–15s first call, embedding model load) |
| Generation | ~8–9s |
| Verification | ~15s for 3 claims |

Verification is the slowest stage. It scales with claims × cited passages ×
windows. Batching the NLI calls is the obvious optimisation and has not been
done.

---

## 6. Findings worth keeping

Things that cost time to discover and would cost time to rediscover.

**Qwen3 needs thinking mode disabled.** Qwen3 is a hybrid reasoning model, and
Ollama returns its thinking tokens in a *separate* `thinking` field. With
`format="json"` the entire reply lands there and `response` comes back empty —
the model appears broken while working correctly. Controlled by
`llm.disable_thinking`. Measured: 3/3 valid JSON with it off, 0/3 with it on.

**Long citation IDs transcribe reliably.** A spike checked whether a 4B model
could copy `srm/academic_integrity/9-verbatim_plagiarism_copy_and_paste_intel`
character-for-character, since a garbled ID is indistinguishable from a
fabricated one. It could — 8/8. No numbered-reference indirection was needed.
A numbered scheme was ~44% faster but would have made fabricated-citation
detection nearly impossible, so it was rejected.

**NLI holds up on real passages.** The Phase 0 spike used 20 hand-written
pairs. Real chunks are long, multi-rule, and full of PDF noise, so this was
re-tested against live retrieved passages: **7/7**, with decisive confidences
(1.00 / 0.95 / 0.00).

**NLI model comparison** (20 hand-written pairs, Phase 0):

| Model | Overall | Numeric |
|---|---|---|
| `DeBERTa-v3-base-mnli-fever-anli` | 90% | **100%** |
| `cross-encoder/nli-deberta-v3-base` | 90% | 100% (0/2 on scope) |
| `mDeBERTa-v3-base-xnli-multilingual` | 85% | 60% |

Known weakness in the winner: **scope confusion** — mistaking a
postgraduate-only rule for one applying to all students. Not yet addressed.

**Citations collided in real documents.** SRM's scholarship page has several
independent numbered lists ("2" appears six times meaning six different
things); the Code of Conduct has six sections all headed "Consequence".
Different passages were claiming the same citation ID, which would make
verification unanswerable. Fixed by combining number + heading slug with an
occurrence suffix.

**PDF extraction buried 52 headings mid-line.** A stray table cell landed in
front of headings with body text following on the same line
(`R 7.3 Minimum Attendance:  A student must...`). The Academic Regulations
document had collapsed from 52 sections to 7, with the flagship 75% rule
inside a 39,000-character blob citable only as `yy_dd_c_l_ss_a`. Fixed by a
normalising pass; that rule now cites as `srm/attendance/7.3-minimum_attendance`.

**One oversized section monopolised retrieval.** Three of the top three
results were chunks of a single examination section, pushing the actual
attendance rule to rank 4. Fixed by the per-citation diversity cap.

**The noise filter was too aggressive.** It was dropping short but genuine
rules. The two errors are not symmetric — surviving noise merely scores badly,
whereas dropped content is permanently unretrievable — so the threshold now
errs toward keeping.

---

## 7. Not built yet

| Phase | Scope |
|---|---|
| **6** | Adversarial question set — 60–80 items across five trap categories, split into tuning and held-out halves |
| **7** | Streamlit UI — question box, per-claim colour-coded verdicts, expandable evidence, abstention banner, removed-claims panel |
| **8** | Evaluation harness — claim P/R, citation accuracy, hallucination rate, abstention quality, latency; JSONL run logs |
| **9** | Hardening, error analysis, demo script |

`Answer` and `Timings` are now populated by `pipeline.answer_question()`.
Nothing yet writes them to disk — JSONL run logging is Phase 8.

---

## 8. Known limitations

**Retrieval is dense-only.** No BM25 or hybrid search, so exact terms and
identifiers are matched only through the embedding. Hybrid retrieval is a
known improvement, not yet implemented.

**Verification is slow.** ~15s for 3 claims on CPU. NLI calls are made one at
a time; batching them is the obvious fix.

**Scope confusion is unhandled.** The NLI model can mistake a
postgraduate-only rule for one applying to all students. Identified in Phase 0,
not yet mitigated. A lightweight scope check alongside the numeric guard is
the likely approach.

**Some heading noise remains.** A few address fragments and table-of-contents
lines are still detected as sections, producing small low-value chunks. They
point at real text, so they are noise rather than wrong citations.

**Roman-numeral headings are not detected.** Content under
"II. ADMISSION TO EXAMINATIONS" merges into the preceding section rather than
being lost — a coarser citation grain, not missing content. Adding a
Roman-numeral pattern risks a worse problem: false positives on the word "I".

**Section structure is flat, not nested.** "4.2.1" is captured and citable but
is not nested under a parent "4.2". Citation accuracy does not depend on that
nesting, so this is a simplification rather than a correctness issue.

**The plagiarism policy is dated 2017.** No newer version was findable on
SRM's official domains as of 2026-08-14. Worth re-checking.

**Abstention thresholds are untuned.** `min_claim_score` (0.5),
`min_supported_claims` (1) and `min_retrieval_score` (0.25) are reasoned
defaults, not measured ones. They need tuning against the tuning half of the
adversarial set once Phase 6 exists — and false abstention needs measuring, or
tuning has nothing honest to optimise against.

---

## 9. Repository conventions

- **Commits** are authored solely by the project owner; no co-author trailers.
- **Never commit or push without explicit approval**, every time.
- **Corpus documents are never committed** — only `data/manifest.yaml`.
- `data/raw/`, `data/processed/`, `data/index/` and `.venv/` are gitignored.
- Tests must run without GPU, model download or network.

### Recent commits

```
286efbc  Phase 4: claim verifier
ce06d51  feat(generate): add retrieval and structured claim generation
b92de7a  Phase 2: chunking and vector index
53867ef  fix(ingest): recover PDF headings buried mid-line
5b26865  fix(schema): make section citations unique per document
f0b4fb7  feat(ingest): add policy document ingestion pipeline
6c67098  Policy Verification System: Phase 0 complete
```

---

## 10. Rebuilding from scratch

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull qwen3:4b

pytest                          # 168 tests
python scripts/ingest.py        # 6 documents → 192 sections
python scripts/build_index.py   # → 258 chunks, then a smoke search
python scripts/ask.py "What is the minimum attendance requirement?"
```
