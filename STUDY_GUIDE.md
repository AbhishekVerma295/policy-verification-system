# Study Guide — read this before the review

Everything you need to explain the project, in the order an examiner is
likely to ask. Read sections 1–4 first; they are enough to answer most
questions. Section 6 is your own part in detail.

---

## 1. The project in three sentences

It is a question-answering system for SRM university policies (attendance,
exams, plagiarism, scholarships, hostel, code of conduct). Instead of trusting
whatever the AI model writes, it **breaks the answer into small claims and
checks every claim against the policy text it cites**. If a claim cannot be
proved it is removed and shown as removed, and if nothing can be proved the
system **refuses to answer** instead of guessing.

**The one-line pitch:** *a confident wrong answer about a rule, with a
citation attached, is worse than no answer — so we verify before we answer.*

---

## 2. What happens when you type a question

Example: *"What is the minimum attendance required?"*

| Step | What happens | File |
|---|---|---|
| 1. **Retrieve** | The question is turned into a vector (a list of 768 numbers) and compared with the vectors of all 258 policy chunks. The 6 closest chunks are picked. | `retrieve.py`, `indexing/store.py` |
| 2. **Generate** | Those 6 passages + the question go to the Qwen language model. It must reply in JSON: a list of claims, each with the citation ID of the passage that supports it. | `generate.py`, `llm.py` |
| 3. **Verify** | Each claim gets three checks (next section). Result: SUPPORTED, REFUTED or UNVERIFIED, with a confidence. | `verify/` |
| 4. **Decide** | Keep only SUPPORTED claims with confidence ≥ 0.5. If none survive, decline the whole question. | `abstain.py` |

`pipeline.py` runs these four steps in order. `app.py` draws the result on
the web page.

**This is called RAG — Retrieval-Augmented Generation.** The model is NOT
trained on the documents. It reads the relevant passages fresh for every
question. That is why any question works, and why adding a new document
needs no training — only re-running `ingest.py` and `build_index.py`.

---

## 3. The three checks (the heart of the project)

| Check | Question it asks | How | File |
|---|---|---|---|
| **Citation check** | Does the cited section really exist? | Looks the citation ID up in the index. An invented ID is called a *fabricated citation*. | `verify/citation.py` |
| **Entailment check (NLI)** | Does the cited text actually prove the claim? | A second, smaller AI model (DeBERTa) reads the passage and the claim and answers: *entailment* (proves it), *contradiction* (says the opposite), or *neutral* (does not say). | `verify/nli.py` |
| **Numeric guard** | Are the numbers in the claim present in the passage? | Plain code, no AI. Extracts numbers like `75%` and compares them exactly. | `verify/numeric.py` |

**The combining rule** (`verify/verifier.py`):

- A claim is SUPPORTED only if **all** checks agree.
- The numeric guard can **block** a claim but can never **approve** one.
- A claim with no real citation can never be supported.

**Why so strict?** Wrongly approving a false claim is invisible to the user.
Wrongly removing a true claim is visible (it appears under "Removed claims").
So the system prefers the visible mistake.

---

## 4. Words you must be able to define

| Term | Meaning in one line |
|---|---|
| **RAG** | Retrieval-Augmented Generation: fetch relevant text first, then let the model answer from it. |
| **Embedding** | Turning text into a list of numbers so that similar meanings get similar numbers. Model: `bge-base-en-v1.5`. |
| **Vector store / index** | A database of those number-lists that can be searched by similarity. We use Chroma. |
| **Chunk** | A piece of a document small enough to search. Ours follow section boundaries — one chunk never mixes two sections. |
| **Citation ID** | `university/policy/section`, e.g. `srm/attendance/7.3-minimum_attendance`. Points at exactly one section. |
| **Claim** | One single fact. "75% attendance is needed and you may appeal" is two claims. |
| **NLI** | Natural Language Inference: given a passage and a statement, does the passage prove it, contradict it, or neither? |
| **Hallucination** | The model stating something the documents do not support. |
| **Abstention** | The system declining to answer. Here it is a normal result, not an error. |
| **LLM** | Large Language Model. Ours is Qwen, run locally through a program called Ollama. |
| **Adversarial set** | Our 64 test questions, written on purpose to trick the system. |

---

## 5. The models and tools

| Job | What we use | Why |
|---|---|---|
| Writes the answer | **Qwen** (`qwen2.5:3b` on this laptop, `qwen3:4b` on the GPU machine) via **Ollama** | Small enough to run locally; no internet, no API cost, data stays private |
| Search | `BAAI/bge-base-en-v1.5` embeddings + **Chroma** | Good accuracy for its size, runs on CPU |
| Fact-checker | `DeBERTa-v3-base-mnli-fever-anli` | Scored best in our early test, 100% on number cases |
| Interface | **Streamlit** | A web page written in plain Python |
| Data shapes | **Pydantic** | Rejects badly-formed data (e.g. broken JSON from the model) immediately |
| Tests | **pytest** — 252 tests | Run in seconds with no GPU, no model, no internet |

**Everything runs on one machine.** No cloud, no API key.

---

## 6. YOUR part — interface and evaluation

### `app.py` — the web page
- Loads the models **once** (`@st.cache_resource`) so each question does not reload them.
- A text box + "Verify answer" button calls `answer_question()` from `pipeline.py` — the *same function* the command-line tool and the evaluation use. So what you demo is exactly what was measured.
- Draws each claim as a coloured card: **green = SUPPORTED, red = REFUTED, amber = UNVERIFIED**.
- "Evidence and checks" under each claim shows the actual policy text and the result of the three checks.
- "Removed claims" panel shows what was rejected and why. **Nothing is hidden.**
- A warning banner appears when the system declines to answer.
- The "Evaluation" tab shows the metrics from the last evaluation run.

### `eval/adversarial.jsonl` — the 64 trick questions

| Category | Count | The trick |
|---|---|---|
| numeric | 16 | One wrong digit changes the answer (75% vs 80%) |
| unanswerable | 14 | Not in the documents — the system should refuse |
| false_premise | 12 | The question assumes something false ("why is 90% attendance required?") |
| negation_exception | 12 | Rules with exceptions that must not be dropped |
| cross_policy | 10 | Same number, different meaning: 75% is the attendance rule AND a fee-waiver tier |

- **15 of the 64 are unanswerable.** Without them we could not measure whether the system refuses too much or too little.
- Split into **33 tuning + 31 held-out**. You adjust settings on the tuning half and report results on the held-out half — like not practising on the exam paper.
- Each question has labels written in advance: is it answerable, which section should be cited, which values must appear, which values would be wrong.

### `src/policyverify/evaluate.py` + `scripts/evaluate.py` — the scoring

| Metric | Meaning |
|---|---|
| **Accuracy** | Share of questions handled correctly |
| **Hallucination rate** | Share where the system stood behind something wrong, or answered an unanswerable question |
| **Citation accuracy** | Of answered questions, share that cite the expected section |
| **Correct abstention** | Share of unanswerable questions it refused (higher is better) |
| **False abstention** | Share of answerable questions it wrongly refused (lower is better) |
| **Latency** | Time for retrieve / generate / verify |

- **Baseline comparison:** the same answers scored as if verification were switched off (every claim kept). This shows what the verifier adds.
- Each run is saved to `eval/results/` so results can be re-read without re-running.

---

## 7. Questions the examiner may ask

**Q: Did you train the model?**
No. We use RAG. The model reads the retrieved passages at question time. Training would make it answer from memory, and then it could not cite a source and we could not check it.

**Q: What if the question is not in the documents?**
The system declines. Either the search finds nothing relevant (similarity below 0.25), or the model returns no claims, or no claim survives verification.

**Q: How do you know a citation is not made up?**
Every citation ID is looked up in the index. If it does not exist, it is flagged as fabricated and the claim cannot be supported.

**Q: Why two AI models?**
One writes (Qwen), one checks (DeBERTa). The checker is a different, smaller model trained only to judge whether text A proves statement B. A model checking its own work would repeat its own mistakes.

**Q: Why a numeric guard if the NLI model already checks?**
Numbers are where a wrong answer hurts most (75% vs 80% decides whether a student can sit an exam). The guard is a cheap, independent second opinion that cannot be fooled by fluent wording.

**Q: Why show the removed claims?**
Transparency. The user can tell the difference between "the policy does not say this" and "the policy says the opposite".

**Q: Why is it slow?**
This laptop has no GPU, so the language model runs on the CPU (about 1–2 minutes per question). On a GPU the generation step takes 8–9 seconds.

**Q: What are the limitations?** (know at least three)
1. **Relevance is not checked.** A claim can be true and correctly cited but not actually answer the question.
2. **Search is embedding-only** — no keyword search, so exact terms can be missed.
3. **Verification is slow** — claims are checked one at a time.
4. **Scope confusion** — the checker can treat a rule for postgraduates as applying to everyone.
5. **Only one university and six documents.**

**Q: What is new / what is your contribution?**
Claim-level verification with three independent checks, visible removal of unverified claims, abstention as a proper result, and a 64-question adversarial test set with labels that allow automatic scoring.

---

## 8. Numbers to remember

| | |
|---|---|
| Documents | 6 |
| Sections | 192 |
| Chunks in the index | 258 |
| Passages retrieved per question | 6 (at most 2 from one section) |
| Test questions | 64 (49 answerable, 15 unanswerable) |
| Trap categories | 5 |
| Automated tests | 252 |
| Confidence needed to keep a claim | 0.5 |
| Minimum search similarity | 0.25 |

---

## 9. Who owns what

| Person | Part | Phases |
|---|---|---|
| Abhishek | Data shapes, document loading, chunking, search | 0–2 |
| Diya | Answer generation, the three checks, abstention, pipeline | 3–5 |
| You | Interface, the 64 test questions, scoring | 6–8 |

Exact files for each person are in `TEAM.md`.
