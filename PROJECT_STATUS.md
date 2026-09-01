# Policy Verification System — Project Status

Context document for sharing with another AI assistant. States what exists,
not why or how to build it next.

Last updated: after Phase 3 (retrieval + structured claim generation).

## What this project is

A local Q&A system that answers questions about SRM Institute of Science and
Technology (Kattankulathur) policies, breaks each answer into individual
factual claims, verifies each claim against the retrieved policy text, shows
the citation for each claim, and abstains when it cannot find support.

## Progress

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations, data contracts, spikes, corpus | done |
| 1 | Ingestion — documents to `Document` objects | done |
| 2 | Chunking and the vector index | done |
| 3 | Retrieval and structured claim generation | done |
| 4 | The verifier (NLI, numeric guards, citation checks) | next |
| 5 | Abstention and transparent correction | not started |
| 6 | Adversarial question set | not started |
| 7 | Streamlit UI | not started |
| 8 | Evaluation harness | not started |
| 9 | Hardening, error analysis, demo | not started |

Current state: you can ask a question from the command line and get back
separate factual claims, each citing real policy sections. Nothing is
verified yet — that is Phase 4.

## Tech stack

- Python 3.12, virtual environment at `.venv/`
- Pydantic 2 for data validation
- Qwen3 4B via Ollama, run locally (8B is a later upgrade)
- `BAAI/bge-base-en-v1.5` embedding model, runs on CPU
- Chroma vector database, persisted to `data/index/`
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` as the NLI fact-checking
  model (chosen, configured, not yet wired in — Phase 4)
- Streamlit for the UI (not yet built — Phase 7)
- pytest for testing, ruff for linting
- Git repo at `github.com/AbhishekVerma295/policy-verification-system`

## Hardware

- Windows 11, RTX 4060 Laptop GPU (8 GB VRAM), 16 GB RAM
- Qwen3 4B + embedding model + NLI model together peak at 3.9 GB VRAM (measured)

## Corpus

One institution: SRM Institute of Science and Technology, Kattankulathur.
6 documents, one per policy type:

| Policy type | Source | Format |
|---|---|---|
| attendance | Academic Regulations 2021 (UG/Integrated PG) | PDF |
| examination | Examination Policy | HTML |
| academic_integrity | Plagiarism Policy (2017) | PDF |
| scholarship | Scholarship Policy | HTML |
| residence | SRMIST Hostels Rules and Regulations (Jan 2025) | PDF |
| code_of_conduct | Code of Conduct for Students | HTML |

Source URLs and checksums are in `data/manifest.yaml`. The documents
themselves are not committed, only the manifest — they are re-downloaded by
running the ingestion script.

Current numbers: **192 sections** extracted, **258 chunks** indexed.

## Project structure

```
policy-verification-system/
├── README.md                     full project documentation
├── PROJECT_STATUS.md             this file
├── requirements.txt
├── config.yaml                   all settings
├── pyproject.toml                pytest + ruff config
├── data/
│   ├── manifest.yaml             corpus source list (committed)
│   ├── raw/                      downloaded files (gitignored)
│   ├── processed/                Document JSON + text (gitignored)
│   └── index/                    Chroma database + index_manifest.json
├── src/policyverify/
│   ├── schema.py                 data models and the citation contract
│   ├── config.py                 settings loader
│   ├── llm.py                    the only place a model is called
│   ├── retrieve.py               search + per-section diversity cap
│   ├── generate.py               prompt building, JSON claim parsing
│   ├── ingest/
│   │   ├── fetch.py              downloads documents from manifest.yaml
│   │   ├── extract.py            PDF/HTML → plain text
│   │   └── normalize.py          plain text → Document with sections
│   ├── indexing/
│   │   ├── chunk.py              Document → Chunks
│   │   └── store.py              Chroma wrapper + index manifest guard
│   ├── verify/                   empty, Phase 4
│   ├── abstain.py                not yet created, Phase 5
│   └── pipeline.py               not yet created, Phase 5
├── app/                          empty, Phase 7 (Streamlit)
├── scripts/
│   ├── ingest.py                 runs the Phase 1 pipeline
│   ├── build_index.py            runs the Phase 2 pipeline
│   └── ask.py                    ask a question from the CLI
├── spikes/
│   ├── spike_corpus.py           tests extraction quality of a URL
│   ├── spike_nli.py              compares NLI model candidates
│   └── spike_vram.py             measures GPU memory usage
├── eval/                         empty, Phase 6/8
└── tests/                        107 tests, no GPU or network needed
    ├── conftest.py               shared fixtures, including FakeLLM
    ├── test_schema.py            40 tests
    ├── test_ingest.py            21 tests
    ├── test_indexing.py          19 tests
    ├── test_generate.py          20 tests
    └── test_retrieve.py          7 tests
```

## What is built and working

### Data contracts (`src/policyverify/schema.py`)

- `CitationID` — format `{university}/{policy}/{section}`, where section
  combines the document's own number with a heading slug, e.g.
  `srm/attendance/7.3-minimum_attendance`. Strictly validated and frozen.
- `Document.citations()` — assigns a unique citation to every section,
  adding an occurrence suffix (`consequence-2`) where a document repeats a
  heading or restarts its numbering.
- `Document`, `Section` — what ingestion produces
- `Chunk`, `RetrievedChunk` — what indexing and retrieval produce
- `Claim`, `DraftAnswer` — the structural contract the LLM must output
- `ClaimVerdict`, `CheckResults` — verification output (Phase 4, unused)
- `Answer`, `Timings` — the final response object (Phase 5, unused)

### Configuration (`config.yaml` + `src/policyverify/config.py`)

One validated YAML file for embedding model, chunking sizes, retrieval
top-k and diversity cap, LLM model/backend/temperature/thinking, NLI model,
and abstention thresholds. `Config.fingerprint()` summarises the settings
that affect results, for recording alongside runs.

### Ingestion (`src/policyverify/ingest/`, run by `scripts/ingest.py`)

1. **fetch.py** — downloads each document from the manifest, saves raw bytes
   to `data/raw/`, records SHA-256 checksums in `data/raw/checksums.json` so
   a later re-fetch detects if the university changed the document
2. **extract.py** — raw bytes to plain text
   - HTML: BeautifulSoup; strips `<script>`/`<style>`/`<nav>`/`<footer>`
     plus SRM-specific chrome (sidebar policy menu, mega-menu, breadcrumb)
     that is not in semantic tags
   - PDF: PyMuPDF; strips lines repeating 3+ times verbatim (running
     page headers and footers)
3. **normalize.py** — splits text into `Section` objects. Detects three
   heading styles: numbered (`4.2 Minimum Attendance`), ALL-CAPS
   (`REGISTRATION AND ENROLLMENT:`), and numbered headings buried mid-line
   by PDF extraction (`R 7.3 Minimum Attendance:  A student must...`), which
   are lifted onto their own line first.

### Indexing (`src/policyverify/indexing/`, run by `scripts/build_index.py`)

- **chunk.py** — splits Documents into Chunks on section boundaries. A chunk
  never spans two sections. Oversized sections are split into overlapping
  windows that all keep the same citation ID and differ only in `chunk_id`.
  Contents-page noise (entries whose whole body is a page number) is dropped.
- **store.py** — Chroma wrapper. Writes `data/index/index_manifest.json`
  recording which embedding model built the index, and refuses to search if
  the configured model no longer matches, because that mismatch otherwise
  fails silently and returns near-random passages.
  What gets embedded includes the university, policy type and section
  heading; what gets stored as chunk text does not, so verification later
  checks claims against the university's words alone.

### Retrieval and generation (Phase 3)

- **retrieve.py** — dense search with optional university and policy-type
  filters, plus a cap on how many chunks any single section may contribute
  (`max_chunks_per_citation`, default 2). Over-fetches then caps.
- **llm.py** — the single place a model is called. `generate(prompt) -> str`,
  with an Ollama backend. Adding a hosted backend later means one branch here.
- **generate.py** — builds the prompt, requires JSON claims validated against
  `DraftAnswer`, retries once on unparseable output then raises. Separates
  citations that match no retrieved passage and reports them as fabricated
  rather than discarding them.
- **scripts/ask.py** — CLI entry point.

## Verified behaviour

Real run (`python scripts/ask.py "What is the minimum attendance requirement
to sit the final examination?"`):

```
CLAIM 1  "A student must maintain a minimum attendance record of at least 75%
          in individual courses..."          -> srm/attendance/7.3-minimum_attendance
CLAIM 2  "Without the minimum attendance of 75%, students become ineligible
          to appear for the end semester examination."
                                             -> srm/attendance/7.3-minimum_attendance
CLAIM 3  "Students with less than 75% attendance ... awarded 'I' Grade."
                                             -> srm/attendance/7.4-attendance_shortage_and_examination
```

Retrieval ~1s warm (~14s on first call, while the embedding model loads),
generation ~8s.

## Key findings from spikes and real runs

- **Qwen3 needs thinking disabled.** Qwen3 is a hybrid reasoning model and
  Ollama returns its thinking tokens in a separate `thinking` field. With
  `format="json"` the entire reply lands there and `response` comes back
  empty — the model appears broken while working correctly. Controlled by
  `llm.disable_thinking` in config.yaml. Measured: 3/3 valid JSON with it
  off, 0/3 with it on.
- **Long citation IDs transcribe reliably.** Qwen3 4B copied IDs like
  `srm/academic_integrity/9-verbatim_plagiarism_copy_and_paste_intel`
  exactly, 8/8. No numbered-reference indirection layer is needed.
- **NLI model choice:** `DeBERTa-v3-base-mnli-fever-anli` won on 20
  hand-written claim/passage pairs — 90% overall, 100% on numeric mismatches
  ("75%" vs "80%"). Known weakness: confuses who a rule applies to
  (postgraduate-only vs. all students). Not yet tested on real retrieved
  passages — that is the main open risk for Phase 4.
- **VRAM:** all three models together peak at 3.9 GB of 8 GB.

## What is not built yet

- Phase 4: the verifier — `verify/nli.py`, `verify/numeric.py`,
  `verify/citation.py`
- Phase 5: `abstain.py`, `pipeline.py`, transparent correction (dropping
  unsupported claims while showing what was removed)
- Phase 6: adversarial test question set
- Phase 7: Streamlit UI
- Phase 8: evaluation harness (claim P/R, citation accuracy, hallucination
  rate, abstention quality, latency)
- Phase 9: hardening, error analysis, demo script

## Known limitations in current code

- Retrieval is dense-only; no BM25 or hybrid search yet, so exact terms and
  numbers are matched only through the embedding
- A few address fragments and table-of-contents lines are still misdetected
  as section headings, producing small low-value sections. They point at real
  text, so they are noise rather than wrong citations.
- Roman-numeral headings ("II. ADMISSION TO EXAMINATIONS") are not detected;
  that content merges into the preceding heading rather than being lost
- Section structure is flat, not nested — "4.2.1" is captured and citable but
  is not nested under a parent "4.2"
- One large section in the attendance document still carries a junk heading
  derived from a PDF table row
- `data/manifest.yaml` lists a 2017-dated plagiarism policy because no newer
  version was findable on SRM's official domains as of 2026-08-14

## Git state

- Branch `main`, remote
  `https://github.com/AbhishekVerma295/policy-verification-system.git`
- All commits authored solely by the project owner, no co-author trailers
- Recent commits:
  - `ce06d51` feat(generate): add retrieval and structured claim generation
  - `b92de7a` Phase 2: chunking and vector index
  - `53867ef` fix(ingest): recover PDF headings buried mid-line
  - `5b26865` fix(schema): make section citations unique per document
  - `f0b4fb7` feat(ingest): add policy document ingestion pipeline
  - `6c67098` Policy Verification System: Phase 0 complete
