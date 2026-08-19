# Policy Verification System
## Implementation Plan (Simplified for Beginners)

---

## What You're Building

An AI system that answers questions about university policies. 

The key difference: it **shows you the proof**. Every answer points to the exact policy text it came from. If it can't find proof, it says nothing rather than making something up.

**Timeline:** 8-12 weeks, working alone, ~10 hours per week.

---

## Big Decisions (Already Made)

These are locked in. Don't change them.

| What | Decision |
|------|----------|
| **Policy documents** | Download from 3-4 universities (about 20-25 documents total) |
| **How to store them** | Save a list of where they came from (not the files themselves) |
| **Python version** | Use Python 3.11 or 3.12 — your 3.14 will cause problems |
| **AI model** | Qwen3 4B first (faster while learning), upgrade to 8B later |
| **How to search** | Use BAAI/bge-base embedding model on your CPU |
| **Search database** | Chroma (simple, works offline) |
| **Fact checker** | NLI model (says if evidence supports the claim) |
| **Show your work** | Display removed claims so user sees what the system rejected |

---

## Folder Organization

This is what your project will look like:

```
policy-verification-system/
├── README.md                         # explain your project here
├── requirements.txt                  # list of Python packages
├── config.yaml                       # all settings in one place
│
├── data/
│   ├── manifest.yaml                 # list of policy sources (you edit this)
│   ├── raw/                          # downloaded policy files (not in git)
│   ├── processed/                    # cleaned versions (not in git)
│   └── index/                        # search database (not in git)
│
├── src/policyverify/                 # THE CORE (your actual code)
│   ├── config.py                     # load settings
│   ├── schema.py                     # ★ START HERE - data shapes
│   ├── ingest/
│   │   ├── fetch.py                  # download from manifest
│   │   ├── extract.py                # turn PDF/HTML into text
│   │   └── normalize.py              # make everything consistent
│   ├── indexing/
│   │   ├── chunk.py                  # split into searchable pieces
│   │   └── store.py                  # put in search database
│   ├── retrieve.py                   # find relevant passages
│   ├── llm.py                        # ★ the ONE place you call Qwen
│   ├── generate.py                   # ask Qwen for claims
│   ├── verify/
│   │   ├── nli.py                    # check: does passage support claim?
│   │   ├── numeric.py                # check: do numbers match?
│   │   └── citation.py               # check: does cited section exist?
│   ├── abstain.py                    # decide: should we refuse to answer?
│   └── pipeline.py                   # connect all the pieces
│
├── app/
│   └── streamlit_app.py              # the interface (imports from src/)
│
├── scripts/
│   ├── build_index.py                # create the search database
│   ├── ask.py                        # ask from terminal
│   └── evaluate.py                   # run tests and show results
│
├── eval/
│   ├── adversarial.jsonl             # your trick questions for testing
│   └── results/                      # where test results go
│
└── tests/
    ├── conftest.py                   # fake AI model for testing
    └── test_*.py                     # actual tests
```

**Key rule:** Keep all code in `src/policyverify/`. Never put logic in `app/streamlit_app.py` — Streamlit reruns the whole file every time you click, so nothing there can be tested.

---

## The Two Decisions You Make First

Before writing any code, decide these two things. Everything else depends on them.

### 1. Citation ID Format

How you'll identify which part of which policy said something.

Example: `uni_a/academic_integrity/4.2`

This means: "University A, Academic Integrity policy, section 4.2"

Write three functions to work with this:
- `parse()` — break it apart
- `validate()` — check it's real
- `resolve()` — find the actual text

### 2. Data Shapes (Pydantic models in `schema.py`)

Define what data looks like when it moves between your functions:

| Name | Fields |
|------|--------|
| `Document` | which university, policy name, title, where from, list of sections |
| `Chunk` | ID, the text, citation ID, university, policy type, section path |
| `Claim` | the statement, list of citation IDs it came from |
| `ClaimVerdict` | the claim, is it supported/refuted/unclear, confidence score, which chunks back it up, detailed checks |
| `Answer` | the question, claims we kept, claims we removed, did we refuse, why, how long it took |

Use Pydantic. It catches bugs where your AI model returns garbage shaped data.

---

## 10 Phases (Week by Week)

### Phase 0: Setup & Spikes (Week 1)

**"Spike" = quick test before building on it**

Do three tests:

1. **Corpus test:** Download one policy from 5 different universities. Extract the text. Look at it. Keep the 3-4 that look clean. (This is how you choose your universities.)

2. **VRAM test:** Run Qwen3 4B in Ollama while the embedding model and fact-checker run on CPU. Watch `nvidia-smi`. Make sure you have enough memory.

3. **NLI test:** Write 20 fake claim/passage pairs by hand. Run them through 2-3 fact-checker models. See which one works best on policy text.

Also: set up Python venv, `requirements.txt`, empty repo, git, write `schema.py`, write `manifest.yaml`.

**Done when:**
- `import policyverify` works
- schema.py exists and you've agreed with yourself on the design
- manifest.yaml lists your chosen universities

---

### Phase 1: Ingestion (Week 2)

Get the policies from the internet into a consistent format.

- `fetch.py` — download files from manifest, verify they're what you expected
- `extract.py` — convert PDF (via PyMuPDF) and HTML (via BeautifulSoup) to plain text
- `normalize.py` — convert everything to a `Document` object with a section tree

Every university formats policies different. Do the messy work here so nothing downstream has to care.

**Done when:** 
- All 20-25 documents are valid `Document` objects
- You've read 3 of them by hand to confirm the text is clean

---

### Phase 2: Chunking & Index (Week 3)

Build the searchable database.

- `chunk.py` — split documents on headings (never mid-section), attach metadata (university, policy, section number, citation ID)
- `store.py` — wrap Chroma (the database)
- `scripts/build_index.py` — orchestrate the whole thing

**Important:** Write a manifest recording which embedding model built the index. If you rebuild with a different model and forget, search silently returns nonsense with no error.

**Done when:**
- Index builds in one command
- Manual search returns obviously relevant chunks

---

### Phase 3: Retrieval & Generation (Week 4)

Make the AI produce answers.

- `retrieve.py` — dense search + optional university filter
- `llm.py` — one function: `generate(prompt) -> str` that calls Qwen via Ollama
- `generate.py` — ask Qwen to output JSON claims with citation IDs, validate it, retry once, fail loudly

Use Ollama's JSON mode so the AI is forced to output valid JSON.

**Important:** Never generate prose and try to parse claims out of it. That breaks constantly. Make the format structural.

**Done when:** `scripts/ask.py "question"` returns claims with citations.

**Expect:** The answers will be mediocre. That's fine. The verifier fixes them.

---

### Phase 4: The Verifier (Weeks 5-6) — **The Core**

This is the important part. Spend your time here.

- `nli.py` — for each claim × cited passage, output: supported / refuted / unclear + score
- `numeric.py` — extract numbers, percentages, dates from both claim and passage. Flag mismatches. ("75%" ≠ "80%")
- `citation.py` — two separate checks: 
  - Does the cited ID exist? (fabricated citation)
  - Does that passage actually support the claim? (wrong citation)
- `pipeline.py` — wire retrieve → generate → verify into an `Answer`

**Why NLI instead of asking Qwen to grade itself?** An NLI model just answers "does text A prove text B?" — that's exactly the question you need. It runs on CPU. And it's more reliable than asking the LLM to judge its own output, which tends to repeat its own mistakes.

**Done when:** Every claim has a verdict, a score, and evidence chunk IDs.

---

### Phase 5: Abstention & Correction (Week 7)

Teach it to refuse.

- `abstain.py` — one threshold over claim scores (in `config.yaml`). Move unsupported claims into `claims_removed` instead of deleting them silently. Abstain entirely if too little survives.

**Important:** Keep the removed claims. That record is the transparency feature.

**Done when:** The system refuses to answer a question your corpus doesn't cover and explains why.

---

### Phase 6: Adversarial Dataset (Week 8)

Create trick questions to test your system.

60-80 questions across five categories:
1. **Cross-university confusion** (your signature trap) — same question, different policy
2. **Numeric traps** — "75%" vs "80%" thresholds
3. **Unanswerable but plausible** — not in the corpus (test abstention)
4. **Negation/exceptions** — "except when...", conditional clauses
5. **False premise** — "Why does the policy require X?" when it doesn't

**Critical:** Split into two sets:
- Tuning set (tune your abstention threshold)
- Held-out test set (report your final numbers on this)

Never tune and test on the same questions. You'll only prove it works on what you tuned.

---

### Phase 7: Streamlit UI (Week 9)

Build the interface.

- Question box
- University filter
- Per-claim color-coded verdicts (green = supported, red = refuted, yellow = unclear)
- Expandable evidence per claim
- Abstention banner
- **Removed-claims panel** (show what was rejected)

It imports `pipeline.py` and renders. No logic. **UI and CLI must produce identical results — proof they share one core.**

**Done when:** UI and CLI give identical verdicts for the same question.

---

### Phase 8: Evaluation (Week 10)

Measure how well it works.

`scripts/evaluate.py` runs your held-out test set and produces five numbers:

| Metric | What it means |
|--------|---------------|
| **Claim P/R** | Are "supported" claims really supported? Did you catch all the true ones? |
| **Citation accuracy** | Cited section exists AND actually supports the claim |
| **Hallucination rate** | % of claims not backed by evidence |
| **Abstention quality** | How many unanswerable questions did you refuse? (don't refuse too much!) |
| **Latency** | How long it takes (split: retrieve / generate / verify) |

Log every run to JSONL so you can track changes.

**Done when:** One command produces the metrics table.

---

### Phase 9: Hardening & Deliverables (Weeks 11-12)

Polish and finish.

- Add hybrid retrieval (BM25 + dense, ~10 lines, real gain on exact numbers)
- Error analysis (pick 15 failures, categorize them, explain why)
- README (this is what people read, not your code)
- Architecture diagram (one picture explaining the flow)
- 5-minute demo script (clean answer → trap caught → abstention)
- Buffer

**The README and demo script matter more than they look.**

---

## Things to Watch Out For

| Problem | How to fix it |
|---------|--------------|
| PDF extraction is garbage | Pick universities by extraction quality in week 1; prefer HTML |
| Qwen returns invalid JSON | JSON mode + Pydantic + one retry + fail loudly. Never silently accept bad output. |
| NLI model is weak on policy text | Test in week 1, not week 6. Fallback: add LLM judge tier later. |
| Run out of VRAM | Start with 4B model; confirm embedder and NLI are on CPU |
| Chunks too big or too small | Make it a config value; inspect real chunks by hand early |
| Scope creep | Reranker, re-retrieval, multi-hop, conversation memory, Neo4j — say no until MVP works end-to-end |

---

## Testing

Use pytest.

Write a ~15-line fake AI model that returns canned answers. This lets your tests run with no GPU and no model.

Test the pure functions:
- Chunk boundaries
- Citation ID parsing
- Numeric extraction
- Threshold logic
- Schema validation

Don't unit-test answer quality. That's what the evaluation harness is for.

---

## How You Know You're Done

Each phase has a "done when" section. For the whole project:

✓ `pytest` passes without a GPU
✓ One command each: rebuild corpus, build index, run evaluation
✓ Streamlit and CLI give identical verdicts
✓ System abstains on a question the corpus doesn't cover
✓ System correctly attributes a policy to the right university when two differ

---

## If You Fall Behind

Cut in this order:

1. Hybrid retrieval (nice to have, not essential)
2. Adversarial set down to 40 items and 3 categories (cross-university, numeric, unanswerable)
3. University filter in the UI

**NEVER cut:**
- The verifier (it's the whole project)
- Abstention (useless without it)
- Claim-level evidence view (your best demo moment)
- Evaluation harness (how you know it works)

---

## Quick Reference

**Hardest parts (hardest to debug if wrong):**
1. The verifier — spend time thinking here
2. The citation scheme — decide early
3. The data shapes — get Pydantic right first

**Easiest wins:**
1. Numeric guard (catches real errors, ~30 lines)
2. Citation existence check (deterministic)
3. Progress tracker (make abstention threshold a config value)

**Most common failures:**
1. Model/index mismatch (write the index manifest!)
2. Parsing prose claims (use JSON contract)
3. Tuning and testing on same questions (split your dataset!)

---

## Environment Setup

Before week 1:

```bash
# Create venv
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install Python 3.11 or 3.12 first
# (your 3.14 will fight with PyTorch)

# Install key packages
pip install ollama pydantic chroma-db torch transformers streamlit pytest

# Confirm
python -m pip list | grep -E "pydantic|chroma|transformers"
ollama --version
```

**Before building:** Run the three Phase 0 spikes. Don't skip them.

---

## Good luck!

You've got a solid plan. The verifier is genuinely interesting work. The fact that you're tracking false abstention (over-refusal) shows you're thinking like an engineer.

Start with `schema.py`. Get that right, and everything else flows.
