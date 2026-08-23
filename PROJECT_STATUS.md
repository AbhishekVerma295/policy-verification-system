# Policy Verification System — Project Status

Context document for sharing with another AI assistant. States what exists,
not why or how to build it next.

## What this project is

A local Q&A system that answers questions about SRM Institute of Science and
Technology (Kattankulathur) policies, breaks each answer into individual
factual claims, verifies each claim against the retrieved policy text, shows
the citation for each claim, and abstains when it cannot find support.

## Tech stack

- Python 3.12, virtual environment at `.venv/`
- Pydantic 2 for data validation
- Qwen3 (4B, upgrading to 8B later) via Ollama, run locally
- `BAAI/bge-base-en-v1.5` embedding model, runs on CPU
- Chroma as the vector database (not yet integrated — Phase 2)
- `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` as the NLI fact-checking model
- Streamlit for the UI (not yet built — Phase 7)
- pytest for testing, ruff for linting
- Git repo, hosted on GitHub at `github.com/AbhishekVerma295/policy-verification-system`

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

Source URLs and checksums are recorded in `data/manifest.yaml`. The actual
documents are not committed to git, only the manifest — they are re-downloaded
by running the ingestion script.

## Project structure

```
policy-verification-system/
├── README.md                    full project documentation
├── PROJECT_STATUS.md             this file
├── requirements.txt
├── config.yaml                   all settings
├── pyproject.toml                pytest + ruff config
├── data/
│   ├── manifest.yaml              corpus source list (committed)
│   ├── raw/                       downloaded files (gitignored)
│   ├── processed/                 extracted Document JSON + text (gitignored)
│   └── index/                     vector database (gitignored, not yet used)
├── src/policyverify/
│   ├── schema.py                  data models: Document, Chunk, Claim,
│   │                               ClaimVerdict, Answer, CitationID
│   ├── config.py                  settings loader
│   ├── ingest/
│   │   ├── fetch.py                downloads documents from manifest.yaml
│   │   ├── extract.py              PDF/HTML → plain text
│   │   └── normalize.py            plain text → Document with sections
│   ├── indexing/                  empty, Phase 2
│   ├── verify/                    empty, Phase 4
├── app/                           empty, Phase 7 (Streamlit)
├── scripts/
│   └── ingest.py                  runs the full Phase 1 pipeline
├── spikes/
│   ├── spike_corpus.py             tests extraction quality of a URL
│   ├── spike_nli.py                compares NLI model candidates
│   └── spike_vram.py               measures GPU memory usage
├── eval/                          empty, Phase 6/8
└── tests/
    ├── conftest.py                 shared fixtures, including a FakeLLM
    ├── test_schema.py              36 tests
    └── test_ingest.py              18 tests
```

## What is built and working

### Data contracts (`src/policyverify/schema.py`)

- `CitationID` — format `{university}/{policy}/{section}`, e.g.
  `srm/attendance/4.2`. Strictly validated, frozen (immutable).
- `Document`, `Section` — what ingestion produces
- `Chunk`, `RetrievedChunk` — what indexing/retrieval will produce (Phase 2/3)
- `Claim`, `DraftAnswer` — the structural contract the LLM must output
- `ClaimVerdict`, `CheckResults` — verification output (Phase 4)
- `Answer`, `Timings` — the final response object

### Configuration (`config.yaml` + `src/policyverify/config.py`)

Single YAML file controlling: embedding model, chunking sizes, retrieval
top-k, LLM model/backend/temperature, NLI model, abstention thresholds.
Every setting has a Pydantic-validated default.

### Ingestion pipeline (`src/policyverify/ingest/`)

Three stages, run end-to-end by `scripts/ingest.py`:

1. **fetch.py** — downloads each document from `data/manifest.yaml`, saves raw
   bytes to `data/raw/{university}/{policy_type}.{ext}`, records SHA-256
   checksums in `data/raw/checksums.json` (detects if a source changes on
   re-fetch)
2. **extract.py** — converts raw bytes to plain text
   - HTML: BeautifulSoup, strips `<script>`/`<style>`/`<nav>`/`<footer>`, plus
     SRM-specific chrome (sidebar policy menu, mega-menu, breadcrumb) that
     isn't in semantic tags
   - PDF: PyMuPDF, strips lines that repeat 3+ times verbatim (running
     headers/footers)
3. **normalize.py** — splits plain text into `Section` objects by detecting
   heading lines (two styles: numbered like "4.2 Minimum Attendance", or
   ALL-CAPS like "REGISTRATION AND ENROLLMENT:"), producing a validated
   `Document`

Verified by actually running against all 6 live SRM documents on 2026-08-19:
**147 sections produced across the 6 documents.** Output saved to
`data/processed/{university}/{policy_type}.json` (structured) and `.txt`
(plain text, for manual review).

### Tests

54 tests total (`test_schema.py`: 36, `test_ingest.py`: 18), all passing, no
GPU or network required. Uses a `FakeLLM` fixture for anything that would
otherwise need a real model call.

### Spikes (one-time validation scripts, not part of the pipeline)

- `spike_corpus.py` — checks whether a URL extracts to clean text before
  adding it to the manifest. Run against all 6 chosen sources: all scored GOOD.
- `spike_nli.py` — ran 20 hand-written claim/passage pairs through 3 candidate
  NLI models. Winner: `DeBERTa-v3-base-mnli-fever-anli`, 90% overall accuracy,
  100% on numeric-mismatch cases. Weak point: confuses claims about who a rule
  applies to (e.g. postgraduate-only vs. all students).
- `spike_vram.py` — confirmed Qwen3 4B + embedding model + NLI model together
  use 3.9 GB of 8 GB available VRAM.

## What is not built yet

- Phase 2: chunking documents into indexed pieces, Chroma vector store
- Phase 3: retrieval, calling Qwen, structured claim generation
- Phase 4: the verifier (NLI checks, numeric guards, citation checks)
- Phase 5: abstention logic, transparent correction
- Phase 6: adversarial test question set
- Phase 7: Streamlit UI
- Phase 8: evaluation harness (5 metrics: claim P/R, citation accuracy,
  hallucination rate, abstention quality, latency)
- Phase 9: hardening, error analysis, demo script

## Known limitations in current code

- A small number of address fragments and table-of-contents lines are
  occasionally misdetected as section headings (produces a few extra small,
  low-value sections — not wrong citations, just noise)
- Section headings written as Roman numerals ("II. ADMISSION TO
  EXAMINATIONS") are not detected as their own heading; that content merges
  into whichever heading came before it
- Section structure is flat, not a nested tree — a subsection like "4.2.1"
  is correctly captured and citable, but not nested under a parent "4.2"
  node
- `data/manifest.yaml` lists a 2017-dated plagiarism policy PDF because no
  newer version could be found on SRM's official domains as of 2026-08-14 —
  worth manually re-checking

## Git state

- Single commit history (previous history was squashed by the project owner)
- All commits authored solely by the project owner, no co-author trailers
- Remote: `https://github.com/AbhishekVerma295/policy-verification-system.git`
