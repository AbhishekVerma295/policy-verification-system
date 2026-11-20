# Policy Verification System

A question-answering system for university policies that **shows its work**.
A transparent question-answering system for university policies.

Ask it *"What is the minimum attendance requirement?"* and it does not simply
answer. It breaks its answer into separate factual claims, checks each one
against the policy text it actually retrieved, shows the exact section behind
every claim, and stays quiet when it cannot find real evidence.

The point is the checking, not the answering. An ordinary chatbot will happily
tell you a policy says 80% when it really says 75%, and you would have no way
to know. This system is built to catch exactly that. An illustration of the
kind of output it gives (the wording is shortened):

```
CLAIM  "A student must maintain at least 75% attendance"   SUPPORTED   1.00
                                                 -> srm/attendance/7.3-minimum_attendance

CLAIM  "A student must maintain at least 80% attendance"   REFUTED     1.00
       The policy says: "A student must maintain a minimum attendance record
       of at least 75% in individual courses..."  (the sentence is copied
       word for word from the policy, not rewritten)

CLAIM  "Students who miss classes must pay a fine"         UNVERIFIED  0.00
       The cited text neither supports nor contradicts this.
```

Everything runs locally. No API keys, no cloud services, no per-token costs.

**Status:** the full pipeline works end to end, with a Streamlit interface,
two labelled evaluation sets (64 questions and 85 claims) and scripts that
score the system on both. It is still being improved: the accuracy figures are
development numbers on a small set, and the pass marks are not yet calibrated.
**For what has been measured, what is weak, and what is planned, read section 0
of [`PROJECT_STATUS.md`](PROJECT_STATUS.md)**; this README does not repeat the
numbers because they are still changing.

---

## Contents

- [Why this is interesting](#why-this-is-interesting)
- [How it works](#how-it-works)
- [Setup](#setup)
- [Usage](#usage)
- [Architecture](#architecture)
- [The corpus](#the-corpus)
- [Design decisions](#design-decisions)
- [Evaluation](#evaluation)
- [Implemented and planned](#implemented-and-planned)
- [Development](#development)

Further reading in this repository:

| Document | What it covers |
|---|---|
| [`CONCEPTS.md`](CONCEPTS.md) | The ideas behind the system - RAG, embeddings, NLI, abstention - explained from scratch |
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Section 0: current state, measured results, known weaknesses, roadmap. Later sections: how every part works, and findings worth keeping |
| [`RESEARCH_PAPER_DATA.md`](RESEARCH_PAPER_DATA.md) | Final results, datasets and evaluation methodology, for the research paper (Word copy alongside) |
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
over-refusal being counted as a failure - a system that abstains on everything
scores perfectly on hallucination rate and is useless.

**True is not the same as on topic.** A claim can be perfectly supported by
its passage and still not answer the question. A separate relevance check
catches that and says so, instead of presenting a true but irrelevant
statement as the answer.

**Numbers are where the damage is.** "75% attendance" versus "80% attendance"
is the difference between sitting an exam and being barred from it, and the
two sentences are otherwise identical. A deterministic numeric guard runs
alongside the model for exactly this reason. It also understands clock times,
so "7:30 pm" is not mistaken for the numbers 7 and 30.

**The verifier itself is measured.** Besides scoring whole answers, the
project has a fixed set of 85 labelled claims that scores the verifier alone,
with eight kinds of wrong claim (a changed number, the right number from the
wrong table row, a flipped meaning, and so on), and every headline rate is
printed with a 95% confidence interval.

---

## How it works

```
question
   |
   v
[1] RETRIEVE ---- find relevant policy passages
   |              hybrid search: keyword (BM25) + embeddings, merged by
   |              reciprocal rank fusion, at most 2 chunks per section
   v
[2] GENERATE ---- draft the answer as structured claims, each with citations
   |              Qwen3 4B via Ollama, reply forced into a JSON schema;
   |              passages that talk to the AI are screened out; claims in a
   |              reply that was cut off are rescued; clause-number citations
   |              are repaired when there is only one possible section
   v
[3] VERIFY ------ check every claim independently:
   |                * does the cited text prove it?        NLI model, sentence by
   |                                                        sentence (+ exact quote)
   |                * do the numbers and times agree?      deterministic
   |                * does the cited section exist?        deterministic
   |                * do two cited sections disagree?      conflict rule
   |              then the relevance gate:
   |                * does a true claim answer the question?  similarity, and one
   |                                                          yes/no model call
   v
[4] DECIDE ------ keep supported, on-topic claims; remove the rest (and show
   |              them); decline entirely if too little survives
   v
answer + evidence + what was removed and why
```

All four steps are built and working.

---

## Setup

| Requirement | Notes |
|---|---|
| Python | **3.12** - what the project is developed and linted on. Newer versions were not re-checked |
| GPU | NVIDIA, ~6 GB+ VRAM (developed on an RTX 4060 laptop, 8 GB) |
| [Ollama](https://ollama.com) | runs Qwen locally |
| Disk | ~5 GB for models |

The embedding and fact-checking models run on **CPU by design**, so the whole
GPU stays free for Qwen. On 2026-11-11 the peak with all three loaded was
3.9 GB of 8 GB (not re-measured since).

```bash
# 1. virtual environment on Python 3.12
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. CPU-only torch first - ~200 MB instead of ~2.5 GB for the CUDA build.
#    Qwen runs through Ollama, so torch never needs the GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. everything else
pip install -r requirements.txt

# 4. the language model
ollama pull qwen3.5:9b        # the default in config.yaml (6.6 GB, best with ~8 GB GPU memory)
ollama pull qwen3:4b          # backup: smaller and faster (2.5 GB)
```

Run the app or a question with the default model:

```bash
streamlit run app.py
python scripts/ask.py "What is the minimum attendance requirement?"
```

**Backup model.** On a machine with less GPU memory, or if the default is too
slow, use `qwen3:4b` for the current terminal window (`config.yaml` stays
unchanged):

```bash
# Windows Command Prompt
set POLICYVERIFY_LLM_MODEL=qwen3:4b
streamlit run app.py

# PowerShell
$env:POLICYVERIFY_LLM_MODEL = "qwen3:4b"
streamlit run app.py
```

Close the window (or `set POLICYVERIFY_LLM_MODEL=` in cmd) to go back to the
default. Measured on 20 Nov 2026 (tuning split): `qwen3.5:9b` 72.7% accuracy,
0% hallucination, 7/7 unanswerable declined, 9/26 answerable declined;
`qwen3:4b` 75.8%, 3.0%, 6/7, 4/26, and about 1.5x faster (`STUDY_GUIDE.md` 10.24).

Verify the install:

```bash
pytest
```

The tests need no GPU, no model and no network - deliberately, so the suite
runs anywhere. All of them should pass.

---

## Usage

### 1. Build the corpus and index

```bash
python scripts/ingest.py        # download the 6 policy documents and split them into sections
python scripts/build_index.py   # chunk and embed them, then run a smoke search
```

The first run downloads the embedding model. To re-process the documents
already in `data/raw/` without downloading again (after changing the
extraction or sectioning code), use:

```bash
python scripts/ingest.py --offline
python scripts/build_index.py   # always rebuild the index afterwards
```

### 2. Ask a question

The web interface (needs Ollama running):

```bash
streamlit run app.py
```

Type any question, or click an example. A progress panel shows which stage the
question has reached. Each claim is a card with its verdict, confidence, the
section it cites and the policy text it was checked against; a claim the policy
contradicts shows the conflicting policy sentence; removed claims, "true but
off topic" claims and declined questions are shown too. Evaluation results are
printed by `scripts/evaluate.py` (see [Evaluation](#evaluation)).

From the command line:

```bash
python scripts/ask.py "What is the minimum attendance requirement?"
python scripts/ask.py "Can a parent stay overnight?" --policy residence
python scripts/ask.py "..." -k 8 --json
python scripts/demo.py --pause   # a few showcase questions, models loaded once
```

On a machine with no GPU, pick a smaller model without editing `config.yaml`:

```bash
set POLICYVERIFY_LLM_MODEL=qwen2.5:3b          # Windows cmd
$env:POLICYVERIFY_LLM_MODEL = "qwen2.5:3b"     # PowerShell
```

Output is per claim, with a verdict, a confidence, the citation, and a
plain-language explanation of *why* that verdict was reached. Claims that
failed verification are **shown, not hidden**. The block below is an
illustration of the layout, not a recorded run:

```
ANSWER  (1 verified claims)

  [1] SUPPORTED  (1.00)
      A student must maintain at least 75% attendance in individual courses.
        -> srm/attendance/7.3-minimum_attendance

--------------------------------------------------------------------------

REMOVED  (2 claims the system would not stand behind)
  Shown rather than hidden, so you can see what was rejected and why.

  [1] REFUTED  (1.00)
      A student must maintain at least 90% attendance in individual courses.
      The cited policy text contradicts this claim (confidence 1.00).
      numbers: claim states 90%, not found in evidence (evidence has: 25%, 75%)

  [2] UNVERIFIED  (0.00)
      Students who miss classes are fined 500 rupees.
      The cited policy text neither clearly supports nor contradicts this claim.
```

When too little survives, the system declines entirely and says why, for
example:

```
  NO ANSWER GIVEN
  The retrieved passages do not answer this question, so no claims were made.
```

Other reasons it can give: the claims it made could not be confirmed against
the cited text; some were contradicted by it; or the claims were true but did
not answer the question ("the policies do not appear to contain what was
asked").

### 3. Check the system

```bash
pytest                                        # unit tests, no models needed
python scripts/validate_evalset.py            # are the labelled questions still valid?
python scripts/evaluate_verifier.py           # the verifier alone on 85 labelled claims; no Ollama needed
python scripts/evaluate.py --split tuning     # whole answers, end to end; needs Ollama; takes several minutes
```

See [Evaluation](#evaluation) for what each one measures and
`PROJECT_STATUS.md` section 0 for the current results.

---

## Architecture

```
src/policyverify/        the core library - no UI, no web framework
  schema.py              * data contracts everything else depends on
  config.py              every setting, loaded from config.yaml
  ingest/                documents in, clean Document objects out
    fetch.py               download from the manifest, checksum them
    extract.py             PDF/HTML -> plain text; tables as one row per line;
                           site footer removed
    normalize.py           text -> sections, by detecting headings
                           (numbered, ALL CAPS, Roman numerals, mid-line)
  indexing/
    chunk.py               Document -> Chunks, on section boundaries
    store.py               Chroma wrapper + index/model mismatch guard
    bm25.py                the keyword search used by hybrid retrieval
  retrieve.py            hybrid search (BM25 + embeddings, reciprocal rank
                         fusion) + per-section cap
  llm.py                 the one place a language model is called
  generate.py            prompt, schema-constrained JSON claims, salvage of
                         cut-off replies, passage screening, citation repair
  verify/                * the heart of the project
    nli.py                 does the passage prove the claim? sentence-level
                           evidence units, table-row / list aware, exact quote
    numeric.py             do the numbers and clock times agree?
    citation.py            is the citation real? was it shown to the model?
    verifier.py            combines them into one verdict (conflict rule,
                           section heading as premise)
    relevance.py           does a true claim actually answer the question?
  abstain.py             which claims to keep; whether to answer at all
  pipeline.py            the whole system end to end, in one function
  evalset.py             the question set: schema, loading, validation
  claimset.py            the claim set: schema, loading, scoring the verifier
  evaluate.py            scoring a run; metrics; confidence intervals; run logs

app.py                   the Streamlit interface
scripts/                 ingest.py, build_index.py, ask.py, demo.py,
                         validate_evalset.py, evaluate.py, evaluate_verifier.py
eval/adversarial.jsonl   64 labelled trap questions
eval/claims.jsonl        85 labelled claims, for scoring the verifier alone
eval/citation_fingerprints.json   hashes of the cited sections, to detect drift
eval/results/            run summaries (committed); per-question logs (local only)
spikes/                  throwaway experiments that de-risked decisions
tests/                   no GPU or network needed
data/manifest.yaml       the corpus recipe (URLs + checksums, not the PDFs)
config.yaml              settings you want to override
```

**The core library imports no Streamlit and no web framework.** Streamlit
re-runs its whole script on every interaction, so logic living there would be
untestable. Keeping it out means the interface, the CLI, and the evaluation
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
claim?"* unanswerable - so `Document.citations()` guarantees uniqueness,
adding an occurrence suffix where needed.

---

## The corpus

**One institution - [SRM Institute of Science and Technology, Kattankulathur](https://www.srmist.edu.in/)
- 7 documents**: one per policy type (attendance, examinations, academic
integrity, scholarships, residence, and conduct), plus, since 2026-11-20,
the Scholarship Categories PDF (2025-26, amended) beside the scholarship page,
cited as `srm/scholarship_categories/...`. On 2026-11-20 that was 225 parsed
sections, of which 222 are indexed, in 307 chunks (212 / 192 / 262 on
2026-11-16). Tables in both PDFs and web pages are stored one row per line,
merged cells included.

Every source was run through `spikes/spike_corpus.py` before adoption and
scored well on extraction quality. At this size, **curating by hand beats
writing a scraper** - no crawler, no robots.txt handling, no brittle
site-specific parser, and you can inspect every extraction yourself.

Ingestion has had to cope with real-world mess: HTML tables (the scholarship
page has dozens) are written out one self-describing row per line, Roman-numeral
headings are recognised, headings buried mid-line by PDF extraction are
recovered, and the site's footer block is removed.

The repository stores a **manifest** (source URLs, access dates, checksums)
rather than the documents. That sidesteps the copyright question and makes the
corpus rebuildable from source, which is better engineering than a folder of
unverifiable binaries. See [`data/manifest.yaml`](data/manifest.yaml).

---

## Design decisions

**Why a separate NLI model instead of asking Qwen to check itself?**
A model grading its own work repeats its own mistakes - the same misreading
that produced the bad claim produces a confident bad grade. An NLI model
answers one narrow question ("does text A prove text B?") independently, runs
on CPU, and is deterministic.

**Why structured JSON claims instead of prose?**
Generating a paragraph and then pulling claims back out of it with regexes
fails constantly, because prose has no rules. Making the format structural
turns "usually works" into "works, or fails loudly". The reply is requested
against a JSON schema, and if a long reply is cut off, the claims that were
completed are kept (a half-written claim never is). You cannot verify a claim
you could not reliably identify.

**Why hybrid search?**
Embedding search alone is touchy about wording on this corpus: all passages
are university policy, so their similarity scores bunch together and a small
rephrasing reorders them. A keyword search (BM25) finds the passage that
literally contains the words; embeddings find rewordings that share no words.
The two rankings are merged by reciprocal rank fusion, which uses only the
ranks, because the two scores are on unrelated scales. The score shown to the
user remains the embedding similarity.

**Why is the claim checked against single sentences, not whole sections?**
The first version scored claims against 900-character blocks, and the model
rejected true claims, including word-for-word quotes. NLI models are trained
on single-sentence premises. Now each sentence and each short run of 2-3
neighbouring sentences is scored, an exact quote is accepted without the model,
and the structure of the text is respected: a table row is never merged with
the next row, and list items are read with the line that introduces them.

**Why does the section heading count as part of the evidence?**
Some sections only say who they apply to in the heading ("For all I Year
Students:"). The checker reads it, for the first chunk of a section only.

**Why can the numeric guard veto support but never grant it?**
Matching digits is not proof that a passage *means* what the claim says. Only
the NLI model can grant support; only the numeric check can overrule it on
numbers.

**Why does support require every check to agree?**
Wrongly marking a claim supported puts a false statement in front of a student
with a citation attached. Wrongly withholding a true one is visible and
recoverable. The asymmetry is deliberate.

**Why not pick the agreeable source when two cited sections disagree?**
If one cited section supports a claim and a different one contradicts it
strongly, the claim is withheld and both sentences are quoted, so the reader
sees the disagreement. Contradictions inside one section, or from a table row
about another category, do not count: that is normal policy writing, not a
conflict.

**Why a relevance gate?**
Verification proves a claim is true of its passage, not that it answers the
question. Asked who the current Vice Chancellor is, the system once produced
"The Vice-Chancellor will deliver the welcome address": true, cited, and no
answer. After verification, each kept claim is compared to the question by
embedding similarity, and when that is unclear one yes/no call goes to the
language model. A true but off-topic claim is removed and labelled as such. If
the model cannot be reached, similarity alone decides and the answer is flagged.

**Why temperature 0?**
So the same question gives the same answer. A system whose output changes
between runs cannot be debugged or evaluated.

**Why record which embedding model built the index?**
Rebuild with a different model and forget, and search silently returns
nonsense with no error at all. It is the most common bug in systems like this,
so the store refuses to search a mismatched index rather than guessing.

**Why show removed claims?**
So a reader can tell "the policy does not mention this" from "we checked and it
was wrong". Nothing the model generated is ever silently discarded.

### Known weaknesses

These are described honestly, with current figures, in section 0.4 of
[`PROJECT_STATUS.md`](PROJECT_STATUS.md) and section 8 of the same file. In
short:

- True answers are sometimes declined, for example scholarship answers that
  are too general for any single table row to prove.
- A claim with the right number taken from the wrong table row can still be
  accepted.
- A claim that is close to the question but not an answer can still pass the
  relevance gate.
- The pass marks were checked against a risk-coverage curve and kept, but
  that check rests on only 44 tuning claims (`scripts/calibrate.py`).
- Prompt-injection protection (screening passages, delimiters, the schema) is
  partial.
- It is one institution, one small model (`qwen3.5:9b` since 20 Nov 2026; all earlier results used `qwen3:4b`), and small evaluation
  sets, so results carry wide error bars.

---

## Evaluation

There are two labelled sets and two scripts, because they answer different
questions.

### Whole answers: the adversarial question set

```bash
python scripts/evaluate.py --split tuning       # while changing the system
python scripts/evaluate.py --split heldout      # the half to report on; use sparingly
python scripts/evaluate.py --sample 10          # a quick spread across categories
python scripts/evaluate.py --resume <log>       # continue an interrupted run
python scripts/evaluate.py --report <log>       # re-print a saved run
python scripts/evaluate.py --report <log> --baseline   # same run, verification off
```

Needs Ollama. Each item is appended to `eval/results/<run-id>.jsonl` as it
finishes, with the full `Answer`, so a surprising score can be read rather than
re-run. A summary is saved beside it as `<run-id>.summary.json`; summaries are
committed to git, the per-question logs are not.

Every run prints, after the main report, a **"verification on vs off"** table:
the same generations re-scored as if every claim had been kept. It is free to
compute (removed claims are stored) and it answers the question that justifies
the verifier existing - do the answers get better when claims are checked?
`--baseline` does the same for a saved run.

What is reported:

| Metric | Question it answers |
|---|---|
| Accuracy | Of all questions, how many ended in the right outcome? |
| Hallucination rate | How often did the system stand behind a wrong figure, or answer an unanswerable question? |
| "Wrong when answering" | Of the answers it chose to give, how many were wrong in any way? |
| Citation accuracy | Does a kept claim rest on a section the labels expect? |
| Correct / false abstention | Does it refuse when it should - and **not** when it shouldn't? |
| Citation precision / recall | Proxies by citation, not human judgements of truth |
| Latency | Split across retrieve / generate / verify |

The headline rates carry **95% Wilson confidence intervals**, because the sets
are small: a score such as "65% on 31 questions" (an illustration, not a
result) could plausibly be anywhere from the high 40s to the high 70s. The interval is printed beside the number so a two-question
difference is not read as an improvement.

**The adversarial set** is 64 hand-written questions in
[`eval/adversarial.jsonl`](eval/adversarial.jsonl), each labelled with the
behaviour a correct system would show.

| Category | Items | The trap |
|---|---|---|
| `numeric` | 16 | Thresholds and deadlines, where one digit changes the outcome |
| `unanswerable` | 14 | Plausible questions the corpus does not cover - should abstain |
| `false_premise` | 12 | The question asserts something untrue; correct it, don't elaborate |
| `negation_exception` | 12 | Rules with carve-outs that must not be dropped |
| `cross_policy` | 10 | The same figure meaning different things in two policies |

Most items score **automatically**, because what can be known in advance is
labelled in advance: `expected_citations`, `required_values`, and
`forbidden_values`. For a numeric trap the wrong answer is known - if `80%`
appears in a claim the system stood behind, that is a hallucination, and no
human has to read it to find out. The set has both unanswerable questions (so
over-refusal is measurable) and answerable ones.

33 questions are for **tuning** and 31 are **held-out**, stratified so both
halves cover every category. Settings are tuned on the first and final results
reported on the second; tuning and reporting on the same questions would only
prove the system works on the questions it was tuned for. The held-out half has
already been looked at once, so treat it as spent for design decisions.

`python scripts/validate_evalset.py` runs four checks, each catching a failure
the previous one cannot:

| Check | Catches |
|---|---|
| **Schema** | malformed or incoherent labels |
| **Existence** | a citation pointing at nothing |
| **Content** | the cited section no longer holds the labelled fact (partial coverage; the script prints exactly how much) |
| **Drift** | the cited section's *text changed* since the labels were written, using the hashes in [`eval/citation_fingerprints.json`](eval/citation_fingerprints.json) |

Existence alone is not enough: an ingestion change can re-split a document so a
citation still resolves while the fact it was labelled for has moved elsewhere.
After a deliberate ingestion change, re-check the items the script names, then
run `--update-fingerprints` to accept the new state.

### The verifier alone: the claim set

```bash
python scripts/evaluate_verifier.py                  # all 85 claims
python scripts/evaluate_verifier.py --split tuning   # or --split heldout
python scripts/evaluate_verifier.py --kind role      # one kind of claim
```

Does **not** need Ollama: the claims are fixed in
[`eval/claims.jsonl`](eval/claims.jsonl), so no language model is involved.
Each claim is paired with the section it cites and a label (does that section
support it?), and the verifier is handed exactly that, with retrieval and
generation taken out of the picture. This is what makes real claim-level
**precision** (of the claims accepted, how many were truly supported) and
**recall** (of the truly supported ones, how many were accepted) reportable.

Each claim has a **kind**, so one number cannot hide where the verifier is
weak: `paraphrase` and `verbatim` should be accepted; `number` (a figure not
in the section), `role` (a figure that is in the section but belongs to a
different table row), `negation`, `entity`, `unsupported` and `wrong_section`
should not. The report also says how many conflicting claims were named
REFUTED rather than merely unverified. A summary is saved to
`eval/results/verifier-<run-id>.summary.json`.

### Where the results are

Current figures, with the dates and settings they were measured under, are in
section 0.3 of [`PROJECT_STATUS.md`](PROJECT_STATUS.md) and in the
`eval/results/*.summary.json` files. They are development numbers on the
tuning half unless a file says otherwise.

> **Note on trap design.** Earlier drafts of this README described
> "cross-regulation confusion" - the same question having a different answer
> under a different dated regulation. That trap is not available here: only
> one regulation document (2021 UG/Integrated PG) is in the corpus. The
> implemented equivalent is `cross_policy`, which is real - `75%` is the
> minimum attendance rule in one policy and a tuition-waiver tier in another.
> Adding a second dated regulation PDF would make the original trap possible.

---

## Implemented and planned

**Implemented**

- Ingestion of six SRM policy documents, with table, heading and footer repair
- Hybrid retrieval (BM25 + embeddings, reciprocal rank fusion, per-section cap)
- Schema-constrained claim generation, rescue of cut-off replies, screening of
  passages that talk to the AI, repair of clause-number citations
- Sentence-level verification with an exact-quote shortcut, table-row and list
  awareness, section heading as premise, a conflict rule between two cited
  sections, and a numeric guard that includes clock times
- A relevance gate (similarity plus one yes/no judge call)
- Abstention with removed claims always shown
- Streamlit interface and command-line tools
- The 64-question adversarial set and the 85-claim set, with confidence
  intervals, a "verification on vs off" comparison, and drift detection on the
  cited sections

- Calibration check of the confidence scores and pass marks
  (`scripts/calibrate.py`); the retrieval-score gate was removed after it
  was measured never to fire
- Final held-out evaluation on 40 teammate-written questions (results in
  `PROJECT_STATUS.md` section 0.3a and `RESEARCH_PAPER_DATA.md`)

**Planned, not built** (the full list and status are in
`PROJECT_STATUS.md` section 0.5)

- Checking a claim against several table rows of one section together
- A final held-out evaluation, with new questions written by a teammate
- Optional: comparison with MiniCheck, scope and date labels per document,
  a few tests that use the real models, pinned dependency versions

---

## Development

```bash
pytest                    # no GPU or network
python -m ruff check .    # lint
python -m ruff check . --fix
```

Tests use a `FakeLLM` fixture and a stub NLI checker, so nothing in the suite
downloads a model or calls Ollama. Model *quality* is measured by the two
evaluation scripts, not asserted in unit tests - those check the plumbing and
the decision logic.

Conventions: commits are authored by the project owner only (no co-author
trailers), nothing is committed or pushed without explicit approval, and the
policy documents themselves are never committed.

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

MIT for the code. The policy documents themselves are **not** redistributed -
only their URLs - and remain the property of SRM Institute of Science and
Technology.
