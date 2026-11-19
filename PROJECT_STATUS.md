# Project Status

Detailed current state of the Policy Verification System. Written as a handoff
document: it states what exists, what was measured, and what is not built yet.

**Last updated:** 2026-11-16, after the audit and the first six improvement
tasks. **Section 0 below is the current state** and holds every measured
result. Sections 1 onward were rewritten on 2026-11-16 to describe how the
system works in the code today; they quote no results of their own, so where
a number appears in both places, section 0 is right.

For the project overview see [`README.md`](README.md). For the ideas behind
it see [`CONCEPTS.md`](CONCEPTS.md). For the final results in paper form see
[`RESEARCH_PAPER_DATA.md`](RESEARCH_PAPER_DATA.md).

---

## 0. Current state (2026-11-16)

### 0.1 What happened

A full audit on 14 Nov found the system worked end to end but was not yet
reliable: verification gave no accuracy gain over switching it off, 5 of 8
unanswerable held-out questions were answered anyway, true quotes of the
policy were being rejected, and claim-level precision and recall could not be
reported at all. A roadmap was agreed and most of it is now done.

### 0.2 Completed

| Task | What changed | Main files |
|---|---|---|
| **Scoring** | Labelled claim set (85 claims, 8 kinds) and a verifier-only benchmark; 95% intervals on headline numbers; "verification on vs off" table printed after every run; scorer bug with `Rs.5000` fixed; result summaries committed | `claimset.py`, `eval/claims.jsonl`, `scripts/evaluate_verifier.py`, `evaluate.py` |
| **Sentence-level checking** | Claims are checked against single sentences and 2-3 sentence windows instead of 900-character blocks; exact-quote shortcut; one order-independent rule for merging results; "conflicting sources" flag; the supporting sentence is stored; NLI runs in float32 (about 3x faster) | `verify/nli.py`, `verify/verifier.py`, `schema.py` |
| **Hybrid search** | BM25 keyword search merged with embedding search by Reciprocal Rank Fusion | `indexing/bm25.py`, `retrieve.py`, `indexing/store.py` |
| **Robust generation** | Replies constrained to a JSON schema; complete claims rescued from cut-off replies; whole passages shown to the model; reply limit raised to 1500 tokens; passages containing instructions aimed at the model are screened out | `llm.py`, `generate.py` |
| **Correction display** | A refuted claim shows the policy sentence it conflicts with, quoted word for word | `app.py`, `scripts/ask.py`, `verifier.py` |
| **Relevance gate** | After verification, the system checks that the kept claims answer the question (similarity plus one yes/no model call). True-but-off-topic claims are shown as such; if none answer the question, the system declines | `verify/relevance.py`, `pipeline.py`, `abstain.py` |
| **Ingestion repair** | HTML tables kept whole, one self-describing row per line; Roman-numeral headings detected; site footer removed; `scripts/ingest.py --offline`; index rebuilt | `ingest/extract.py`, `ingest/normalize.py`, `indexing/chunk.py` |
| **Structure-aware checking** | Table rows and list items are separate evidence units; a row from another table no longer counts as "sources disagree"; the first chunk of a section is checked with its heading; clock times compared as times | `verify/nli.py`, `verify/verifier.py`, `verify/numeric.py` |
| **Citation repair** | A clause number cited from inside a passage (`srm/attendance/3.7`) is mapped to that passage when exactly one retrieved passage contains it | `generate.py` |
| **Calibration and thresholds** | Calibration error and risk-coverage measured on tuning claims; thresholds kept (no clear evidence to move them); the retrieval-score gate removed because it never fired | `calibration.py`, `scripts/calibrate.py`, `abstain.py` |
| **Documentation** | README and sections 1-10 of this file rewritten from the current code | `README.md`, this file |

Label changes made with the ingestion repair, each checked against the policy
text: `fp-010` now cites `srm/examination/XVII-result_passing_board_meeting`
(the section holding the grace-mark rule); `num-016`, `xpol-001` and
`xpol-005` now accept any scholarship table stating the 100% / 75% waiver.

### 0.3 Measured results

Tuning split, 33 questions. These are development numbers, not final ones:
several settings were tuned on these same questions.

| | Audit (14 Nov) | Now |
|---|---|---|
| Accuracy | 57.6% | 78.8% |
| Hallucination rate | 18.2% | 3.0% |
| Unanswerable questions declined | 2 of 7 | 6 of 7 |
| Answerable questions wrongly declined | 4 of 26 (15.4%) | 2 of 26 (7.7%) |
| Mean seconds per question | 14.2 | about 6 |
| Accuracy, verification on vs off | equal | 78.8% vs 69.7% |
| Hallucination, verification on vs off | 18.2% vs 21.2% | 3.0% vs 18.2% |

All "Now" figures are measured (runs of 16 Nov, e.g.
`eval/results/run-20261116-084951`). Generation varies between runs by about
one question.

Verifier alone, 85 labelled claims (measured): precision 92.2%, recall 100%
(was 91.5% / 91.5%); false-accept rate 10.5%; verbatim quotes 15 of 15 (was
12 of 15); about 0.9 s per claim (was 2.9 s).

Calibration of the NLI score on the 44 tuning claims (measured, fitted and
scored on the same claims, so optimistic): expected calibration error 0.032,
0.028 after temperature scaling. Too small a gain to apply.

On 49 fresh questions (measured, three runs on 16 Nov): accuracy 85.7-89.8%,
hallucination 2.0%, unanswerable declined 8 of 9, wrongly declined 2-4 of 40.

| | |
|---|---|
| **Tests** | 472 passing, no GPU / model / network required |
| **Lint** | ruff clean |
| **Corpus** | 6 documents, 212 sections, 192 indexed sections, 262 chunks |
| **Largest section** | 10,395 characters (was about 42,000) |

### 0.3a Final held-out results (16 Nov 2026, each run once, code at commit `3782df6`)

All measured. Nothing was changed in the system after these questions were seen.

| | New 40 (teammate's questions and answer key) | Original held-out 31 | Original 31 at the audit (14 Nov) |
|---|---|---|---|
| Accuracy (scorer) | 77.5% (95% interval 62-88%) | 74.2% (57-86%) | 64.5% |
| Hallucination rate | 0.0% (0-9%) | 6.5% (2-21%) | 19.4% |
| Unanswerable declined | 6 of 6 | 7 of 8 | 3 of 8 |
| Answerable wrongly declined | 7 of 34 (20.6%) | 4 of 23 (17.4%) | - |
| Accuracy, verification on vs off | 77.5% vs 75.0% | 74.2% vs 67.7% | equal |
| Hallucination, verification on vs off | 0.0% vs 20.0% | 6.5% vs 25.8% | - |

Runs: `eval/results/run-20261116-104137` (new 40, `eval/final_heldout.jsonl`),
`run-20261116-104643` (original 31), `verifier-20261116-105128` (held-out
claims: precision 88.5%, recall 100%, false-accept 16.7% = 3 of 18, all
"right number, wrong row").

Hand reading of the new 40 against the teammate's key: 26 of 40 fully
correct (65%). Five answers the scorer passed give only half of what the
key asks (`tm-15`, `tm-22`, `tm-23`, `tm-24`, `tm-25`). Of the 7 declined,
5 had a correct answer that was blocked (`tm-11`, `tm-12`, `tm-30` by the
relevance gate; `tm-33`, `tm-35` by the verifier on a table row) and 2 had
a wrong answer that verification rightly stopped (`tm-34`, `tm-36`).
`tm-04` gave the right fine but cited the wrong section, and the verifier
accepted it. No false statement was shown as verified.

Label correction (16 Nov, after the run): `un-010` and `un-014` in the
original 31 were wrongly labelled unanswerable (the policies state the
scholarship deadline and a dress code). Re-scoring the same saved answers
with corrected labels: accuracy 74.2% (unchanged), hallucination 3.2% (was
6.5%), unanswerable declined 6 of 6, wrongly declined 5 of 25. File:
`eval/results/run-20261116-104643.relabelled.summary.json`.

Not done: the full "what each part adds" table. Only verification on/off
was measured.

### 0.4 Known weaknesses

- **Two false-premise questions are still declined** (`fp-001`, `fp-011`): the
  model only denies the premise, or the relevance gate blocks the answer.
- **Wrong section cited** on four tuning questions (`num-006`, `xpol-003`,
  `xpol-007`, `fp-005`); one unanswerable question is answered (`un-007`).
- **Right number, wrong table row** is still accepted by the verifier in 4 of
  6 benchmark claims.
- **Two tables that truly disagree** about the same category are no longer
  flagged as a conflict (table rows are excluded from that rule).
- **Close-but-not-an-answer claims** still pass the relevance gate (room-change
  rules for "how do I apply for a room change").
- **The correction sentence is not always a natural correction**; it is the
  sentence the NLI model scored as most contradictory.
- **Thresholds rest on little data**: 44 tuning claims, only two in the middle
  of the score range. They were checked and kept, not proven optimal.
- The held-out split has been viewed once and must not be used for design.

### 0.5 Remaining roadmap

| # | Task | Status |
|---|---|---|
| 1 | Fix the wrongly declined answers | Done |
| 2 | Calibration and thresholds | Done |
| 3 | Final evaluation: add the teammate's new held-out questions, run the held-out sets once, build the "what each part adds" table | Done 16 Nov: 40 teammate questions with the teammate's own answer key (`eval/final_heldout.jsonl`), results in 0.3a. The "what each part adds" table is not done. A further 50-question set (`eval_docs/SRM_Policy_QA_Independent_Test_Set_v2.md`) is ready but not yet run |
| 4 | "Right number, wrong table row" in the verifier | Not started; deliberately held until after the final run |
| 5 | Optional: compare with MiniCheck; scope and date labels per document; a few tests that use the real models; pin dependency versions | Not started |

Dropped: the separate "tie numbers to the supporting sentence" task. Most of
its benefit depended on the table repair, and item 1 covers what is left.

### 0.6 Commands

```bash
pytest
python scripts/calibrate.py                   # calibration + risk-coverage, tuning claims only
python scripts/ingest.py --offline            # re-process data/raw without downloading
python scripts/build_index.py
python scripts/validate_evalset.py            # labels still valid
python scripts/evaluate_verifier.py           # verifier alone, about 90 s, no Ollama
python scripts/evaluate.py --split tuning     # end to end, about 4 min, needs Ollama
streamlit run app.py
```

---

## 1. Summary

A local question-answering system over SRM Institute of Science and Technology
(Kattankulathur) policies. It writes each answer as separate factual claims,
checks every claim against the policy text it cites, reports a verdict and a
confidence per claim, removes what it cannot stand behind, and declines to
answer when too little survives.

Sections 1 onward were rewritten on 2026-11-16 to match the code as of commit
`e34ecbc`; the calibration work and the final evaluation that followed are
noted where they change something. Anything measured is quoted only in section 0 above, or in
`eval/results/*.summary.json`; this part of the file describes how things
work, not how well.

| | |
|---|---|
| **Pipeline** | question, retrieve, generate, verify, relevance gate, decide, `Answer` |
| **Implemented** | everything in section 4 |
| **Planned, not built** | see section 0.5 (roadmap) and section 7.3 |
| **Calibration** | checked, thresholds kept; retrieval-score gate removed (`scripts/calibrate.py`) |
| **Corpus** | 6 documents, 212 parsed sections, 192 indexed sections, 262 chunks (counted on 2026-11-16 from `data/processed/`) |
| **Labelled question set** | 64 questions, `eval/adversarial.jsonl` (33 tuning, 31 held-out) |
| **Labelled claim set** | 85 claims, `eval/claims.jsonl` |
| **Interface** | `streamlit run app.py` |
| **Tests** | no GPU, model or network needed; the count is deliberately not stated here because it keeps changing (run `pytest`) |
| **Lint** | `python -m ruff check .` |

---

## 2. Environment

| | |
|---|---|
| OS | Windows 11 |
| Python | 3.12 in `.venv/` (`pyvenv.cfg` says 3.12.10); the ruff target is 3.12 |
| GPU | RTX 4060 Laptop, 8 GB VRAM |
| RAM | 16 GB |
| Generation model | `qwen3:4b` via Ollama (local, GPU) |
| Embedding model | `BAAI/bge-base-en-v1.5`, 768-dim (CPU) |
| NLI model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (CPU, float32) |
| Vector store | Chroma, persisted to `data/index/` |
| Keyword index | BM25, built in memory from the chunks (nothing is saved on disk) |
| Repo | `github.com/AbhishekVerma295/policy-verification-system` |

Measured VRAM with all three models loaded simultaneously, on 2026-11-11:
**3.9 GB of 8 GB**. This was before the relevance gate and the sentence-level
verifier, which add no new model but do add one extra call to the same Qwen
model; it has not been re-measured.
The embedding and NLI models are pinned to CPU on purpose so the whole GPU
stays available for Qwen.

`torch` is installed as the **CPU-only** build (~200 MB rather than ~2.5 GB).
Qwen runs through Ollama, which has its own runtime, so torch never needs CUDA.

Two older notes disagreed about newer Python versions (the README said not
3.13+, this file said 3.13 was verified). Neither could be re-checked from the
code. Use 3.12, which is what the project is developed and linted on.

---

## 3. Corpus

One institution, six documents, one per policy type. Counts below were taken
on 2026-11-16 by reading `data/processed/srm/*.json` and running the chunker
on it (no model needed).

| Policy type | Source document | Format | Sections parsed | Chunks |
|---|---|---|---|---|
| `attendance` | Academic Regulations 2021 (UG / Integrated PG) | PDF | 52 | 66 |
| `examination` | Examination Policy | HTML | 59 | 88 |
| `academic_integrity` | Plagiarism Policy (2017) | PDF | 9 | 11 |
| `scholarship` | Scholarship Policy | HTML | 29 | 39 |
| `residence` | SRMIST Hostels Rules and Regulations (Jan 2025) | PDF | 50 | 38 |
| `code_of_conduct` | Code of Conduct for Students | HTML | 13 | 20 |
| | | | **212** | **262** |

Of the 212 parsed sections, **192 are indexed**: the rest are contents-page
entries, bare titles and unfinished fragments that the chunker drops (see
4.4). A section longer than 1800 characters becomes several chunks that share
one citation. The largest section is 10,395 characters (an examination
section); before the ingestion repair it was about 42,000.

Why the numbers moved since the 11 Nov version of this file: the examination
policy went from 26 to 59 sections because Roman-numeral headings are now
detected, and the scholarship page went from 41 to 29 because its tables are
now kept whole instead of being shattered into 45-character "sections".

Source URLs, access dates, checksums and the reasoning behind each pick are in
[`data/manifest.yaml`](data/manifest.yaml). The documents themselves are **not
committed** - only the manifest. `python scripts/ingest.py` re-downloads them.

**Scope note:** this began as a 3-4 institution plan. Narrowing to one
institution was a deliberate decision, and it cost the project its intended
signature adversarial trap - "cross-university confusion", where the same
question has a different right answer at a different school.

The replacement originally proposed was *cross-regulation* confusion (SRM
publishes several dated, programme-specific regulation documents). **That is
not available here either**: only one regulation document is in the corpus,
the 2021 UG/Integrated PG regulations. What was actually built is
**`cross_policy`** confusion - the same figure meaning different things in two
different policies, e.g. `75%` is the minimum attendance rule *and* a
tuition-waiver tier. See 4.9. Adding a second dated regulation PDF would make
the original cross-regulation trap possible.

---

## 4. What is built

Everything in this section exists in the code today.

### 4.1 Data contracts - `schema.py`

The foundation; everything imports from here. Pure Pydantic models, no I/O.

| Model | Purpose |
|---|---|
| `CitationID` | `{university}/{policy}/{section}`, frozen, strictly validated |
| `Section`, `Document` | What ingestion produces |
| `Chunk`, `RetrievedChunk` | What indexing and retrieval produce |
| `Claim`, `DraftAnswer` | The structural contract the LLM must satisfy |
| `CheckResults`, `ClaimVerdict` | Verification output (see below) |
| `Answer`, `Timings` | The final response object, populated by `pipeline.py` |
| `PolicyType`, `SourceFormat`, `VerdictStatus` | `StrEnum`s, so they serialise as plain strings |

`CheckResults` records which check said what. Besides the NLI label and score,
the numeric result and the citation result, it now carries: the sentence that
decided the verdict and the chunk it came from (`evidence_text`,
`evidence_chunk_id`); a `conflict` flag with the contradicting sentence;
`exact_quote`; and the relevance-gate fields (`relevant`,
`relevance_similarity`, `relevance_source`). Every new field defaults to
empty, so older saved run logs still load.
`Answer` has `relevance_judge_failed` (see 4.7) and `Timings` has
`relevance_ms`, which is part of `verify_ms`, not added on top of it.

`Document.citations()` is the authoritative way to cite sections. It assigns a
unique citation per section, because `Section.section_key()` alone cannot
guarantee uniqueness - it only ever sees one section at a time.

### 4.2 Configuration - `config.py` + `config.yaml`

One validated file for every tunable. Groups: `paths`, `embedding`,
`chunking`, `retrieval`, `llm`, `nli`, `verification`, `abstention`,
`relevance`. Every setting has a Pydantic default, so the system runs with no
`config.yaml` at all - the file only overrides what it mentions (today it
lists only a subset; the rest come from the defaults in `config.py`).

Settings that matter for the newer parts:

- `retrieval`: `hybrid_search` (on), `rrf_k` (60), `hybrid_candidate_pool`
  (50), `bm25_k1` / `bm25_b`, `max_chunks_per_citation` (2),
  `oversample_factor` (4), `top_k` (6).
- `llm`: `max_tokens` (1500), `max_passage_chars` (16000), `max_parse_retries`
  (1), `disable_thinking`, `keep_alive`, `timeout_seconds`.
- `nli`: `entailment_threshold` (0.5), `conflict_threshold` (0.9),
  `max_window_sentences` (3), `batch_size` (16).
- `relevance`: `enabled`, `similarity_accept` (0.85), `similarity_reject`
  (0.55), `similarity_fallback` (0.60), `judge_max_claims` (6).
- `abstention`: `min_claim_score` (0.5), `min_supported_claims` (1). Both were
  checked on a risk-coverage curve and kept. `min_retrieval_score` is still
  read from `config.yaml` but no longer used: the gate was removed after it
  was measured never to fire.

The values above are the defaults in `config.py` on 2026-11-16 and may change
as calibration lands. `Config.fingerprint()` summarises the settings that
affect results, for recording alongside runs. The environment variable
`POLICYVERIFY_LLM_MODEL` overrides `llm.model`.

### 4.3 Ingestion - `ingest/` (run by `scripts/ingest.py`)

1. **`fetch.py`** - downloads each document listed in the manifest, saves raw
   bytes to `data/raw/`, records SHA-256 checksums in `data/raw/checksums.json`
   so a later re-fetch detects if the university silently changed a document.
   Checksums live in a separate generated file rather than being written back
   into `manifest.yaml`, because round-tripping that hand-commented file
   through a YAML library would strip every comment.
2. **`extract.py`** - raw bytes to plain text.
   - HTML: BeautifulSoup; strips `<script>`/`<style>`/`<nav>`/`<footer>` plus
     SRM-specific chrome (sidebar policy menu, mega-menu, breadcrumb) that is
     **not** in semantic tags and was found by inspecting the real DOM.
   - **Tables (new):** every HTML table is written out one row per line, each
     line naming its table and its columns ("SAT SCORE | Category: 1401 -1500
     | Benefit: 75% ..."), cells joined by `" | "`. Before, the scholarship
     tables were flattened cell by cell and fell apart.
   - **Footer removal (new):** SRM's end-of-page "Where could your journey at
     SRMIST take you?" block is dropped, but only after checking it holds
     nothing except that prompt and links.
   - PDF: PyMuPDF; strips lines repeating 3+ times verbatim, which removes
     running page headers and footers.
3. **`normalize.py`** - splits text into `Section` objects by detecting four
   heading styles: numbered (`4.2 Minimum Attendance`), ALL-CAPS
   (`REGISTRATION AND ENROLLMENT:`), numbered headings buried mid-line by PDF
   extraction (lifted onto their own line first), and **Roman numerals (new)**
   (`II. ADMISSION TO EXAMINATIONS`, detected strictly so that the word "I"
   starting a sentence never matches; cited as e.g.
   `II-admission_to_examinations`). A table row is never a heading, and a
   numeric band such as "90 to 94.99 Percentile" is not treated as a numbered
   heading.

`python scripts/ingest.py --offline` rebuilds `data/processed/` from the
bytes already in `data/raw/` (after checking them against the recorded
checksums), so a change in output is caused by a code change and not by the
university editing a page. Rebuild the index afterwards.

### 4.4 Indexing - `indexing/` (run by `scripts/build_index.py`)

- **`chunk.py`** - Documents to Chunks on section boundaries. A chunk never
  spans two sections. Oversized sections split into overlapping windows that
  all keep the same citation and differ only in `chunk_id`. Furniture is
  dropped: contents-page entries (fewer than three real words), and, for text
  under 60 characters, bare all-capitals titles and fragments that stop on a
  comma or colon. The filter leans towards keeping, because dropped content
  can never be retrieved again.
- **`store.py`** - Chroma wrapper. Writes `data/index/index_manifest.json`
  recording which embedding model built the index, and **refuses to search**
  if the configured model no longer matches. Also builds the BM25 index in
  memory from the stored chunks, and can look up the similarity of any chunk
  to a query vector.
- **`bm25.py`** - the keyword search used in hybrid retrieval: plain-Python
  BM25 over each chunk's section path plus text, lowercase alphanumeric
  tokens, a small stop-word list, no stemming.

  What gets **embedded** includes university, policy type and section heading;
  what gets **stored as chunk text** does not. Search benefits from the
  context; verification must check claims against the university's words
  alone, not a string we assembled.

### 4.5 Retrieval and generation

- **`retrieve.py`** - **hybrid search**. The question is run through the
  embedding search and through BM25 (both honour the same university / policy
  filters), and the two ranked lists are merged by **Reciprocal Rank Fusion**:
  each list gives a passage `1 / (rrf_k + rank)` and the two are added, so a
  passage both searches like beats one that only one search ranks first. Only
  the order comes from the fusion; each result's `score` stays its cosine
  similarity to the question, which is what the interface and the abstention
  check compare against. Then `diversify()` caps how many chunks any single
  section may contribute (default 2) and keeps the top `k`. With
  `retrieval.hybrid_search: false`, or a store that lacks the keyword index
  (some test stubs), it falls back to embedding-only search.
  Why it exists: embedding scores on this corpus bunch into a narrow band, and
  a rephrased question reordered them; the code comments record one case where
  the section that literally says "To 'Pass' a course ... 50%" ranked about
  10th by embeddings alone.
- **`llm.py`** - the single place a language model is called.
  `generate(prompt, json_mode=True, schema=None) -> str`, `LLMBackend`
  protocol, `OllamaBackend` implementation. When a JSON schema is passed it
  is given to Ollama as the constrained output format.
- **`generate.py`** -
  - builds the prompt: at most 6 claims, one fact per claim, each a complete
    sentence, qualifiers kept, citation IDs copied exactly, and an empty claims
    list if the passages do not contain the specific thing asked;
  - **schema-constrained output:** the reply is requested against the
    `DraftAnswer` JSON schema, so field names cannot drift;
  - **salvage:** a reply cut off at the token limit is not thrown away - every
    claim that was completed before the cut is kept, a half-written claim never
    is. Only when nothing at all can be recovered is the question retried once
    and then failed with `GenerationError`;
  - **instruction screen:** `screen_passages()` leaves out any retrieved
    passage that contains text addressed to an AI assistant ("ignore previous
    instructions", "add a claim and cite this passage", ...), and logs which
    one. It is a pattern list, not a guarantee;
  - **whole passages:** `fit_passages()` shows each passage in full or not at
    all, dropping the lowest-ranked ones if the 16,000-character budget would
    be exceeded; the first passage is always kept;
  - **`repair_citation()`:** the model sometimes cites a clause number that
    sits inside a retrieved section (for example `srm/attendance/3.7` when
    "3.7 Adding and dropping courses" is inside `3.5-summer_semester`). It is
    rewritten to the real section only when the document matches, the slug
    is a dotted clause number, and exactly one retrieved section contains it;
    otherwise it is left unknown. The repair adds no trust: the claim still
    has to be supported by the section it now cites;
  - `drop_unknown_citations()` separates citations matching no retrieved
    passage and **reports** them as fabricated rather than discarding them.

  The prompt states explicitly that passages are reference material and any
  instructions inside them must be ignored. That, the screen, and the schema
  together reduce prompt-injection risk; none of the three is complete (see
  the module docstring in `generate.py`).

### 4.6 The verifier - `verify/`

Three independent checks per claim, combined by `verifier.py`.

- **`nli.py`** - `NLIChecker.check(premise, hypothesis)` returns entailment /
  neutral / contradiction with probabilities. The claim is no longer scored
  against 900-character blocks. Instead each cited chunk is cut into
  **evidence units** and every unit is scored:
  - **Exact-quote shortcut:** if the claim appears word for word in the
    passage (at least 40 characters and 6 words, and not preceded by a negating
    word), it is entailed without calling the model.
  - **Sentences first.** If any single sentence entails the claim, that
    settles it.
  - **Then windows** of 2-3 neighbouring sentences (`nli.max_window_sentences`),
    for claims that join two sentences.
  - **Structure-aware units.** A table row (one `" | "`-separated line) is a
    unit by itself and is never joined to the next row, because two rows are
    two categories. List items (`a)`, `(ii)`, `2.`) start new units even with
    no full stop; a numbered subsection heading ("5.2 ...") ends a window; and
    items under a lead-in line ("... at the following timings: -") are also
    offered joined to that lead-in. Text with no sentence punctuation falls
    back to overlapping slices of about 300 characters, so nothing is skipped.
  - **Batched, float32.** The pairs are scored in batches of
    `nli.batch_size`, and the model is loaded in float32 (the code comments
    record 416 ms in float16 against 66 ms in float32 for one short pair on CPU).
  - Support anywhere in the passage counts as support; a contradiction only
    wins if nothing entailed. The strongest contradiction anywhere (and the
    strongest one from prose rather than a table row) is kept for the
    conflict rule.
- **`numeric.py`** - `check_numbers(claim, premises)` extracts numeric values
  (handling `75%`, `75 %`, `1,200`, `3.0`) and reports any number in the claim
  absent from the evidence. Ignores small bare values (0, 1, 2) that are list
  markers rather than asserted facts. **Clock times** (`7:30 pm`, `07.30 pm`,
  `19:30`) are read as one value and compared as times, so "7:30 pm" is not
  mistaken for the numbers 7 and 30; a missing am/pm is not held against a
  match, a different one is. It does not understand numbers written as words
  ("two weeks" against "14 days").
- **`citation.py`** - `check_citation()` distinguishes **fabricated** (exists
  nowhere in the corpus) from **misused** (real section, wrong attribution)
  from **not retrieved** (real but never shown to the model). A store failure
  is explicitly *not* reported as fabrication - that would blame the model for
  our own outage.
- **`verifier.py`** - combines them into one `ClaimVerdict`.

**Combination rules:**

- One order-independent rule over all cited chunks: best support anywhere at
  or above `entailment_threshold` means SUPPORTED (numbers permitting);
  otherwise the strongest contradiction at or above it means REFUTED;
  otherwise NEUTRAL. Listing the citations in a different order cannot change
  the verdict.
- Support requires *every* check to agree; any single objection withholds it.
- The numeric guard can **veto** support (the claim becomes NEUTRAL with half
  the score) but never **grant** it.
- **Conflict rule:** if one cited section supports the claim and a *different*
  cited section contradicts it at or above `conflict_threshold` (0.9), the claim
  is withheld as NEUTRAL with `conflict = True`, and the explanation quotes
  both sentences. A contradiction from inside the same section never counts
  (a rule followed by its exception is normal policy writing), and neither
  does one from a table row (a row about another category is not a source
  disagreeing).
- **Section heading as premise:** the first chunk of a section is checked with
  its own heading in front, because some sections only say who they apply to
  in the heading ("For all I Year Students:"). Later chunks and table rows do
  not get one.
- A claim with no citation, or only fabricated citations, cannot be supported.
  A fabricated citation alongside a valid one does not block the valid one;
  it is noted in the explanation.
- A refuted claim shows the policy sentence it conflicts with, quoted word for
  word (`policy_says()`), together with a note that this is the sentence the
  checker found, not a rewrite. That sentence is the one the NLI model scored
  as most contradictory, so it is not always a natural correction.

### 4.7 The relevance gate - `verify/relevance.py`

Verification proves a claim is *true of its passage*. It never asked whether
the claim *answers the question*. The gate closes that gap and runs inside the
verify stage, after the claims are verified and before the keep / decline
decision.

It looks only at claims that would be kept (SUPPORTED at or above
`min_claim_score`):

1. **Similarity** (free): cosine similarity of each claim to the question,
   with the same embedding model and query prefix retrieval uses. If the best
   claim is at or above `similarity_accept`, the answer is on topic; below
   `similarity_reject`, off topic.
2. **One yes/no judge call** to the same Qwen model, with a JSON schema
   forcing `{"answered": true/false}`, when the best claim lands in between.
   The prompt carries worked examples, including one where a true statement
   corrects a mistaken question, which counts as an answer.
3. **Fallbacks, never silent:** if the judge fails, similarity alone decides
   against `similarity_fallback`, and `Answer.relevance_judge_failed` is set;
   if no similarity can be computed either, claims are left as verification
   found them and marked `unchecked`.

A claim the gate marks off topic is *not kept* but is listed among the removed
claims with the note "Supported by the policy, but it does not answer the
question asked", and the interface draws it in its own colour instead of the
green "supported" one. If nothing relevant is left, the system declines with
the reason that the policies do not appear to contain what was asked. The gate
can be switched off with `relevance.enabled: false`.

The three thresholds were chosen on the tuning questions and a set of fresh
questions only, never the held-out half, so they are somewhat optimistic for
those same questions.

### 4.8 Abstention and the pipeline

- **`abstain.py`** - two separate decisions. `should_keep()` decides which
  claims survive (SUPPORTED *and* at or above `min_claim_score` *and* not marked
  off topic by the relevance gate; NEUTRAL means the evidence did not settle it,
  which is not the same as true). `decide()` first checks that something was
  retrieved at all (no passages means decline),
  then partitions the verdicts and declines if fewer than
  `min_supported_claims` survive. The decline reason says which of three things
  happened: claims were true but off topic, claims were contradicted, or
  nothing could be confirmed.

  `AbstentionDecision` exposes `.refuted`, `.unsupported` and `.off_topic`
  separately: "the policy says the opposite" and "true but not what you asked"
  are different findings from "the policy does not say".

  **Nothing generated is ever discarded.** Every claim ends up in `kept` or
  `removed`, including when the system abstains. A test asserts this
  invariant directly, since it is easy to break when adding a branch.

- **`pipeline.py`** - `answer_question()` runs retrieve, generate, verify (the
  relevance gate is part of this stage), decide; times each stage; and returns
  a populated `Answer`. The store, LLM, NLI checker and relevance gate are
  all injectable, so callers can load models once and reuse them, and tests
  can substitute fakes and run the whole path in milliseconds. An optional
  `on_stage` callback lets an interface show progress.

  An abstention returns a normal `Answer` with `abstained=True` and a reason -
  it is a result, not an error, so callers do not have to catch anything. A
  fabricated citation is added to the reason as a note.

### 4.9 The two labelled sets

**The adversarial question set** (whole answers) - Phase 6.

- **`evalset.py`** - `EvalItem` schema, JSONL loader, and
  `validate_against_corpus()`. The schema rejects incoherent labels: an
  unanswerable item cannot expect citations, and an answerable one must name
  at least one, or citation accuracy cannot be scored for it.
- **`eval/adversarial.jsonl`** - 64 questions. Comments and blank lines are
  skipped by the loader, so the file stays annotated and machine-readable.
- **`scripts/validate_evalset.py`** - four checks. Schema (all items),
  existence (every cited section still resolves), content (the cited section
  still holds the labelled fact, where a label makes that checkable), and
  drift (the cited text is unchanged since the labels were written, against
  `eval/citation_fingerprints.json`, which currently records 33 sections).
  It prints how many items each content check could cover and says so when
  coverage is partial; the counts changed with the ingestion repair and are
  not repeated here. A forbidden value appearing in the cited section is a
  *warning*, not a failure, because a figure in a different role is a
  legitimate trap (`num-006`). `--update-fingerprints` records the current
  hashes as the accepted baseline; only run it after checking the reported
  items by hand.

| Category | Items | The trap |
|---|---|---|
| `numeric` | 16 | Thresholds and deadlines; one digit changes the outcome |
| `unanswerable` | 14 | Not covered by the corpus - should abstain |
| `false_premise` | 12 | The question asserts something untrue |
| `negation_exception` | 12 | Rules with carve-outs that must not be dropped |
| `cross_policy` | 10 | The same figure meaning different things in two policies |

33 tuning / 31 held-out, stratified so both halves cover every category.
Labels fixed in advance: `answerable`, `expected_citations`,
`required_values`, `forbidden_values`. The last of these is what makes most
items score automatically - for a numeric trap the wrong answer is known, so
`80%` appearing in a kept claim is a hallucination no human needs to confirm.
A few labels were changed after the ingestion repair, each checked against the
policy text; section 0.2 lists them.

**The claim set** (the verifier on its own) - added 2026-11-15.

- **`claimset.py`** and **`eval/claims.jsonl`** - 85 claims, each fixed in
  advance and paired with the section it cites and a label (does that section
  support it?). Retrieval and generation are taken out of the picture: the
  verifier is handed the claim and every chunk of its cited section.
- Eight **kinds**, so a single precision figure cannot hide where the verifier
  is weak. Supported: `paraphrase` (32), `verbatim` (15). Not supported:
  `number` (8, a figure changed to one not in the section), `role` (6, a figure
  that *is* in the section but means something else, i.e. the wrong table row),
  `negation` (7), `entity` (5), `unsupported` (6, plausible but not stated),
  `wrong_section` (6, a true fact cited to the wrong section). The kind
  determines the label, and the loader rejects a line where they disagree.
- The report gives precision and recall (95% intervals), false-accept and
  false-reject rates, a confusion count, how many conflicting claims were
  named REFUTED rather than merely unverified, a breakdown by kind, and a list
  of the wrong verdicts.
- Tuning and held-out splits exist in this set too. The held-out half has
  been looked at once and must not be used for design (section 0.4).

### 4.10 The evaluation harness

`src/policyverify/evaluate.py` scores an `Answer` against an `EvalItem` (pure,
no I/O) into one outcome: `correct`, `correct_abstention`, `wrong_value`,
`wrong_citation`, `incomplete`, `false_abstention`, `answered_unanswerable`,
or `error`. Only *kept* claims are scored - a forbidden value in a removed
claim is the verifier working.

`scripts/evaluate.py` runs the set through `pipeline.answer_question()`,
appending each result (with the full `Answer`) to
`eval/results/<run-id>.jsonl` as it finishes. `--resume` continues an
interrupted run; `--report` re-prints a saved one; `--report --baseline`
re-scores the same generations with verification switched off.

What the reports contain:

- accuracy, hallucination rate, citation accuracy, correct and false
  abstention, claims kept / removed, citation precision and recall (proxies by
  citation, not human judgements of truth), latency per stage;
- **95% Wilson intervals** on the headline rates, because the sets are small
  and a two-question difference is not an improvement;
- a "wrong when answering" rate: of the questions the system chose to answer,
  how many were wrong in any way, not only by a listed forbidden figure;
- the **"verification on vs off" table**, printed after every run: the same
  generations re-scored as if every claim had been kept. It is free to compute
  because removed claims are stored in the `Answer`, and it is the evidence for
  whether the verifier is earning its cost;
- a `.summary.json` per run, saved next to the log. **Summaries are committed
  to git; the per-question `.jsonl` logs are local working data.**

`scripts/evaluate_verifier.py` is the claim-set equivalent: it needs the index
and the NLI model but **not Ollama**, takes `--split` and `--kind`, and writes
`eval/results/verifier-<run-id>.summary.json`.

Numbers from these runs are in section 0.3 and in
`eval/results/*.summary.json`. They are development numbers on the tuning
split unless a file says otherwise.

### 4.11 Scripts

| Script | Does |
|---|---|
| `scripts/ingest.py` | manifest, download, extract, `Document` JSON (`--offline` to re-process the files already downloaded) |
| `scripts/build_index.py` | processed documents, chunks, embedded index, then a smoke search |
| `scripts/ask.py` | question, retrieval, claims, verdicts, decision (`--policy`, `--university`, `-k`, `--json`) |
| `scripts/demo.py` | a few showcase questions with the models loaded once (`--pause`) |
| `scripts/validate_evalset.py` | schema, existence, content and drift checks on the adversarial set |
| `scripts/evaluate.py` | the adversarial set end to end; needs Ollama |
| `scripts/evaluate_verifier.py` | the claim set against the verifier alone; no Ollama |

### 4.12 Tests

Test files in `tests/`: `test_schema`, `test_ingest`, `test_indexing`,
`test_bm25`, `test_retrieve`, `test_generate`, `test_verify`,
`test_evidence_structure`, `test_relevance`, `test_abstain`, `test_pipeline`,
`test_evalset`, `test_evaluate`, `test_claimset`, plus `conftest.py`.

They run with no GPU, model or network. `conftest.py` provides `FakeLLM`; the
verify and pipeline tests provide stub NLI checkers and a stub store. Model
quality is measured by the two evaluations (4.10), not asserted here: the unit
tests check plumbing and decision logic. The number of tests is not stated in
this file; run `pytest` for the current count. A few tests that use the real
models are on the roadmap (section 0.5) and do not exist yet.

---

## 5. Behaviour seen with the real models

**Historical: these checks were run on 2026-11-11, before the sentence-level
verifier, hybrid retrieval and the relevance gate existed. The verdicts
below are what was seen then; they have not been re-run on the current code.**
Do not read them as current results; section 0.3 and the benchmark runs are.

| Claim | Verdict then | Correct |
|---|---|---|
| "must maintain at least **75%** attendance" | SUPPORTED 1.00 | yes |
| "must maintain at least **80%** attendance" | **REFUTED 1.00** | yes |
| "students who miss classes must pay a fine of 500 rupees" | NEUTRAL 0.00 | yes |
| claim citing `srm/attendance/99-does-not-exist` | NEUTRAL, **fabricated** | yes |

Asking *"What is the campus wifi password?"* produced **zero claims** then -
the model declined rather than inventing an answer.

Latency: the 2026-11-11 figures (about 8-9 s generation, about 15 s to verify
3 claims) no longer apply, because verification was rewritten and runs in
batches in float32. Current per-question and per-claim timings are in section
0.3, and every run summary records latency per stage.

---

## 6. Findings worth keeping

Things that cost time to discover and would cost time to rediscover.

**Qwen3 needs thinking mode disabled.** Qwen3 is a hybrid reasoning model, and
Ollama returns its thinking tokens in a *separate* `thinking` field. With
`format="json"` the entire reply lands there and `response` comes back empty -
the model appears broken while working correctly. Controlled by
`llm.disable_thinking`. Measured: 3/3 valid JSON with it off, 0/3 with it on.

**Long citation IDs transcribe reliably.** A spike checked whether a 4B model
could copy `srm/academic_integrity/9-verbatim_plagiarism_copy_and_paste_intel`
character-for-character, since a garbled ID is indistinguishable from a
fabricated one. It could - 8/8. No numbered-reference indirection was needed.
A numbered scheme was ~44% faster but would have made fabricated-citation
detection nearly impossible, so it was rejected. (What the model does get
wrong is a *clause number inside a section*; see `repair_citation` in 4.5.)

**NLI model comparison** (20 hand-written pairs, Phase 0):

| Model | Overall | Numeric |
|---|---|---|
| `DeBERTa-v3-base-mnli-fever-anli` | 90% | **100%** |
| `cross-encoder/nli-deberta-v3-base` | 90% | 100% (0/2 on scope) |
| `mDeBERTa-v3-base-xnli-multilingual` | 85% | 60% |

Known weakness in the winner: **scope confusion** - mistaking a
postgraduate-only rule for one applying to all students. Reading the section
heading as part of the premise (4.6) fixes the case where the heading carries
the scope; the general problem is not addressed.

**Long premises make the NLI model reject true claims.** The first verifier
scored claims against 900-character blocks. The code comments record that it
rejected TRUE claims, including word-for-word quotes of the section they
cite, several as REFUTED at 0.95 confidence or more. NLI models are trained on
single-sentence premises. Scoring per sentence and per 2-3 sentence window
(the SummaC approach) fixed most of it; the exact-quote shortcut handles the
rest of the verbatim cases. Section 0.3 gives the measured effect on the
verbatim claims.

**Flattening a document destroys the structure the verifier needs.** After the
ingestion repair stored tables one row per line, joining neighbouring rows into
a window made a model "prove" claims that mixed two categories, and joining
across numbered subsections once "supported" a false claim about B.Tech
semesters. Hence the structure-aware units in 4.6: rows stand alone, windows
stop at a subsection heading, list items and timetable items are read with
their lead-in. Each of these was added after a specific wrong verdict recorded
in the comments of `verify/nli.py` and `verify/verifier.py`.

**The section heading can be the only place a scope is stated.** "For all I
Year Students:" is a heading; its list items never say who they apply to.
Offering the checker the heading as part of the first chunk turned a true claim
from 0.00 to 0.98 in one recorded case, and it must be applied to the first
chunk only: with the heading put in front of a later chunk, a false summer
semester claim was supported.

**Clock times were being split into two numbers.** "7:30 pm" read as 7 and 30
vetoed a correct dinner time because 30 is not in "07.30 pm". Times are now
compared as times (4.6).

**Grounding and relevance are different properties.** Early on, "How much
does the hostel mess cost per month?" - a figure the corpus never states -
returned seven claims about the mess, each correctly verified and cited, none
about cost. The system answered a different question and stood behind it.
That is what the relevance gate (4.7) was built for.

**Sampling the adversarial set found two dataset bugs and one system gap.**
Running five items live gave 3/5. The two failures were informative rather
than noisy: one exposed the relevance gap above, and the other showed four
`false_premise` items were mislabelled `answerable: false` when the topic *is*
covered and the system should correct the premise rather than abstain -
inconsistent with how `fp-001` was labelled. One note was also factually
wrong: it claimed no grace-mark policy exists, when the examination policy
states a maximum of 5 grace marks. All four items were relabelled.

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

**Undetected Roman-numeral headings hid a fifth of the corpus.** Everything from
"V. DETENTION/CANCELLATION ..." to the end of Section A of the examination
policy had folded into one section of about 42,000 characters, which the
two-chunks-per-section cap meant the retriever could barely see into.
The earlier worry that detecting them would match the word "I" was handled
by a strict pattern (4.3).

**Tables flattened cell by cell cut the scholarship page into noise.** The cell
"95 Percentile and above" was read as heading number 95, and the page broke
into dozens of 45-character sections. Tables are now written one row per line,
and a numeric band is never a heading.

**One oversized section monopolised retrieval.** Three of the top three
results were chunks of a single examination section, pushing the actual
attendance rule to rank 4. Fixed by the per-citation diversity cap.

**The noise filter was too aggressive.** It was dropping short but genuine
rules. The two errors are not symmetric - surviving noise merely scores badly,
whereas dropped content is permanently unretrievable - so the threshold
errs toward keeping.

**A model with no reply cap can run for ever.** One adversarial question made
`qwen2.5:3b` in JSON mode generate for 25 minutes without stopping. `llm.max_tokens`
caps it. The cap was first 800, which cut real replies off mid-claim; it is now
1500 and a cut-off reply is salvaged rather than discarded (4.5).

---

## 7. Interface, extras, and what is not built yet

### 7.1 Interface - `app.py`

`streamlit run app.py`. A question box and example buttons (the examples fold
away once there is an answer). Progress is shown stage by stage while a
question runs. The answer is one colour-coded card per claim, with the
cited section IDs and a plain-language explanation. Supported, refuted,
unverified and "true but off topic" claims each have their own colour. A
refuted claim shows the policy sentence it conflicts with. An "Evidence and
checks" expander shows the NLI result, whether the numbers match, whether the
citation exists, the one sentence that decided the verdict, and the full
cited policy text. There is also an abstention banner, a removed-claims panel,
per-stage timings, the retrieved passages with their scores, and the raw
`Answer` as JSON. Evaluation numbers are printed by `scripts/evaluate.py`
(the interface no longer has an Evaluation tab).

The interface was simplified on 2026-11-11: **it no longer has a policy
filter or a top-k slider** (an older version of this file said it did). It
searches every policy with `retrieval.top_k` passages. The command-line
`scripts/ask.py` still accepts `--policy`, `--university` and `-k`.

Models are loaded once per server process (`st.cache_resource`) and the
language model is warmed up at start, so the first question does not pay the
load. The headless test of the app (`streamlit.testing.AppTest`) is described
in an older version of this file; it was not re-checked for this rewrite.

### 7.2 Other additions

- `scripts/demo.py` - a few showcase questions, models loaded once.
- `POLICYVERIFY_LLM_MODEL` environment variable overrides `llm.model`, so a
  machine with no GPU can run `qwen2.5:3b` without editing `config.yaml`.
- `llm.keep_alive` (30 min) keeps the model loaded between questions.
- `llm.timeout_seconds` is applied to the Ollama client, and `llm.max_tokens`
  caps reply length (see section 6).
- `STUDY_GUIDE.md` - review prep;
  `RESEARCH_PAPER_DATA.md` - results, datasets and methodology for the paper.

### 7.3 Planned, not built

The single source of truth for this is the roadmap in section 0.5. Summary, for
convenience; nothing in this table exists yet:

| Item | Status |
|---|---|
| Check a claim against several table rows of one section together (scholarship rejections) | proposed |
| Calibration: temperature-scale the NLI score, pick pass marks from a risk-coverage curve | done; thresholds kept (section 0) |
| Final evaluation on the held-out sets and new teammate-written questions | done (section 0.3a) |
| The "what each part adds" table | not done |
| Comparison with MiniCheck | optional, not started |
| Scope and date labels per document | optional, not started |
| A few tests that run the real models | optional, not started |
| Pinned dependency versions | optional, not started |
| Human labels of claim correctness | not started; citation precision is still only a proxy |

---

## 8. Known limitations

The measured weaknesses (wrongly declined answers, right number from the wrong
table row, close-but-not-an-answer claims, uncalibrated confidence) are listed
with their current figures in section 0.4. The rest, from reading the code:

**The relevance gate is itself a model judgement.** Between the two similarity
thresholds it asks the same 4B model that wrote the claims, and its thresholds
were chosen on the tuning and fresh questions, so its results there are
optimistic. If the judge is unreachable the system falls back to similarity
alone and flags it.

**Scope confusion is only partly handled.** The NLI model can mistake a
postgraduate-only rule for one applying to all students. Using the section
heading as part of the premise covers the case where the heading states the
scope; there is no general scope check.

**The numeric guard is deliberately simple.** It compares digits, percentages
and clock times. It does not understand numbers written as words, unit
conversion or arithmetic, and it cannot tell that a figure sits in the wrong
table row (section 0.4).

**Prompt-injection protection is partial.** The instruction screen is a
pattern list and the model can still be talked round; verification does not
catch a planted passage that is itself a passage in the corpus.

**Abstention thresholds rest on little data.** `min_claim_score` and
`min_supported_claims` were checked on a risk-coverage curve and kept, but on
only 44 tuning claims. The retrieval-score check was removed because it never
fired.

**Heading noise may remain.** The chunker drops contents entries, bare titles
and unfinished fragments, but sections were not re-audited line by line after
the ingestion repair, so some low-value chunks may remain. They point at real
text, so they are noise rather than wrong citations.

**Section structure is flat, not nested.** "4.2.1" is captured and citable but
is not nested under a parent "4.2". Citation accuracy does not depend on that
nesting, so this is a simplification rather than a correctness issue.

**The plagiarism policy is dated 2017.** No newer version was findable on
SRM's official domains as of 2026-08-14. Worth re-checking.

**One institution and one small generation model.** Everything runs on
`qwen3:4b`, which writes the claims and also judges relevance. The corpus is
six documents from one university.

**The evaluation sets are small and were written inside the project.** 64
questions and 85 claims. Intervals are printed so small differences are not
over-read, and several settings were tuned on the tuning halves, so those
numbers are development numbers. The held-out half of the question set has been
viewed once and must not be used for design.

---

## 9. Repository conventions

- **Commits** are authored solely by the project owner; no co-author trailers.
- **Never commit or push without explicit approval**, every time.
- **Corpus documents are never committed** - only `data/manifest.yaml`.
- `data/raw/`, `data/processed/`, `data/index/` and `.venv/` are gitignored.
- In `eval/results/`, the `*.summary.json` files are committed and the
  per-question `*.jsonl` run logs are not.
- Tests must run without GPU, model download or network.

### Commit history (newest first, from `git log`, as of 2026-11-16)

```
e34ecbc  2026-11-16  Verifier: structure-aware evidence, heading premise, clock times; repair clause-number citations
0448551  2026-11-15  Ingestion repair: whole tables, Roman-numeral headings, footer removal; label remaps
87ee93e  2026-11-15  Hybrid retrieval, sentence-level NLI evidence, and relevance gate
6e46dff  2026-11-15  show the policy sentence beside refuted claims
53edac9  2026-11-15  schema-constrained generation, salvage, whole passages, instruction screen
4e6cfa7  2026-11-15  claim benchmark code, intervals, scorer fix, tests
5906cb5  2026-11-15  UI fixes and complete-sentence claim prompt
4770922  2026-11-15  claim-level verifier benchmark, confidence intervals, baseline comparison
12dbddf  2026-11-11  Merge pull request #2 (evaluation)
424197b  2026-11-11  Simplify the interface, show progress, and harden the LLM call
eba4cf0  2026-11-11  Merge pull request #1 (evaluation)
5606a70  2026-11-11  evaluation
461d7a6  2026-11-03  Harden adversarial-set validation with content and drift checks
c3c1741  2026-09-27  Phase 6: adversarial question set
8d6d656  2026-09-25  Phase 5: abstention and transparent correction
```

Older history (Phases 0-4) is in `git log`. Calibration (`calibration.py`,
`scripts/calibrate.py`), the final held-out evaluation and the label
corrections are committed and summarised in section 0.

---

## 10. Rebuilding from scratch

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
ollama pull qwen3:4b

pytest                              # no GPU, model or network needed
python scripts/ingest.py            # download and process the 6 documents
python scripts/ingest.py --offline  # (later) re-process data/raw without downloading
python scripts/build_index.py       # chunk, embed, then a smoke search
python scripts/validate_evalset.py  # labelled citations still resolve and have not drifted
python scripts/ask.py "What is the minimum attendance requirement?"
streamlit run app.py                # the interface (needs Ollama running)

python scripts/evaluate_verifier.py          # verifier alone on the 85 claims; no Ollama
python scripts/evaluate.py --split tuning    # end to end on the tuning questions; needs Ollama
python scripts/evaluate.py --split heldout   # the half to report on; use for final numbers only
```

Always rebuild the index (`build_index.py`) after changing ingestion or the
embedding model; the store refuses to search a mismatched index.
