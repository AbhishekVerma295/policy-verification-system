# Concepts

The ideas this project is built on, explained from scratch. Read this if you
want to understand *why* the system is shaped the way it is, rather than just
what the code does.

No prior knowledge of language models is assumed.

---

## Contents

1. [The problem: hallucination](#1-the-problem-hallucination)
2. [RAG: retrieval-augmented generation](#2-rag-retrieval-augmented-generation)
3. [Embeddings and vector search](#3-embeddings-and-vector-search)
4. [Chunking](#4-chunking)
5. [Claim decomposition](#5-claim-decomposition)
6. [Citations, and two ways they fail](#6-citations-and-two-ways-they-fail)
7. [NLI: the fact-checker](#7-nli-the-fact-checker)
8. [Deterministic guards](#8-deterministic-guards)
9. [Abstention](#9-abstention)
10. [How the system evaluates itself](#10-how-the-system-evaluates-itself)
11. [Glossary](#11-glossary)

---

## 1. The problem: hallucination

A language model predicts plausible text. It does not look anything up, and it
has no internal sense of "I know this" versus "this sounds right". When it
does not know, it produces something that *reads* correct.

For policy questions that is dangerous in a specific way. Consider:

> **Real rule:** "A student must maintain a minimum attendance record of at
> least **75%** in individual courses."
>
> **Model output:** "Students must maintain at least **80%** attendance."

The second sentence is fluent, confident, formatted identically, and wrong in
the one way that matters. A student who believes it turns up to an exam they
are barred from. Nothing about the sentence signals the error.

This is called **hallucination**: output that is fluent and confident but not
grounded in any real source.

Three ideas in combination address it:

1. **Ground the answer** in retrieved text rather than model memory (RAG).
2. **Check each claim** against the text it cites (verification).
3. **Refuse** when the check fails (abstention).

This project is an implementation of those three ideas, in that order.

---

## 2. RAG: retrieval-augmented generation

**RAG** means: before asking the model anything, go and fetch relevant source
text, then put that text in the prompt and instruct the model to answer *only*
from it.

```
question ──► search the documents ──► put passages in the prompt ──► model answers
```

Why this helps:

- The model has the actual policy text in front of it, so it does not need to
  recall anything.
- Because you chose the passages, you know what the answer *should* be based
  on — which is what makes checking possible at all.
- Updating a policy means re-indexing a document, not retraining a model.

**What RAG does not do:** it does not guarantee the model uses the passages. It
can still ignore them, misread them, blend two rules together, or add a
detail from memory. Retrieval makes verification *possible*; it does not make
it unnecessary. That gap is the whole reason Phase 4 of this project exists.

In this codebase: `retrieve.py` does the search, `generate.py` builds the
prompt and parses the reply.

---

## 3. Embeddings and vector search

To search by *meaning* rather than keywords, text is turned into numbers.

An **embedding model** converts a piece of text into a list of numbers (here,
768 of them) called a **vector**. The useful property: texts that mean similar
things get vectors that are close together, even when they share no words.

```
"minimum attendance requirement"  ──►  [0.21, -0.05, 0.88, ...]
"how many classes must I attend"  ──►  [0.19, -0.03, 0.91, ...]   ← close by
"hostel mess timings"             ──►  [-0.44, 0.62, 0.10, ...]   ← far away
```

Searching means: embed the question, then find the stored passages whose
vectors are nearest. "Nearest" is measured by **cosine similarity** — the
angle between two vectors. We report it as a score from 0 to 1 where higher is
better.

A **vector database** (here, **Chroma**) stores vectors alongside their text
and metadata, and answers "find me the *k* nearest" efficiently.

### The trap that bites everyone

Embeddings from two different models are **not comparable**. If you build an
index with model A and then search it with model B, nothing raises an error —
the arithmetic still works. Search simply returns near-random passages, and
the application looks fine.

This project defends against it explicitly: `store.py` writes an
`index_manifest.json` recording which model built the index, and refuses to
search if the configured model no longer matches.

---

## 4. Chunking

Whole documents are too big to embed usefully, so they are split into
**chunks** — the unit that gets stored, retrieved, and cited.

Where you cut matters more than it sounds.

**The rule here: never split across a section boundary.** A chunk that spans
two unrelated rules will happily "support" a claim that mixes them together —
exactly the failure the project exists to catch. So chunks follow the real
headings in the real document, which is also what makes a citation like
`7.3-minimum_attendance` mean something exact.

Two complications, both real in this corpus:

- **Sections too big to embed.** One section ran to 39,000 characters. Those
  get split into overlapping windows. Every window keeps the *same* citation
  (they are all genuinely that section) and differs only in its chunk ID. The
  **overlap** exists so a rule sitting on a window boundary is not cut in half
  and lost to both sides.
- **Sections that are not content.** Contents-page entries whose entire body
  is a page number ("Payment Of Fine" → "12") are dropped, since they contain
  no rule to cite.

In this codebase: `indexing/chunk.py`.

---

## 5. Claim decomposition

Most RAG systems generate a paragraph. This one does not. The model is
required to return a **list of separate factual claims**, each naming the
passages that support it, as JSON:

```json
{"claims": [
  {"text": "Students must attend at least 75% of classes.",
   "citation_ids": ["srm/attendance/7.3-minimum_attendance"]}
]}
```

Two reasons.

**Verification needs a unit.** You cannot check "a paragraph". You can check
"one factual statement against one passage". Decomposition creates something
checkable.

**Honest grading needs atomicity.** Take:

> "Students need 75% attendance and may appeal to the Dean."

If the first half is right and the second invented, what single verdict is
honest? None. Split into two claims, each gets its own verdict, and the answer
can keep one and drop the other.

**Why JSON rather than parsing prose?** Generating a paragraph and then
extracting claims from it with regexes fails constantly, because prose has no
rules. Making the format structural turns "usually works" into "works, or
fails loudly" — and the reply is validated against a schema, so malformed
output is rejected rather than half-understood.

In this codebase: `schema.py` defines `Claim` and `DraftAnswer`;
`generate.py` enforces them.

---

## 6. Citations, and two ways they fail

A **citation** identifies exactly which piece of which document backs a claim.
Here they look like:

```
srm / attendance / 7.3-minimum_attendance
 │        │              │
 │        │              └── section: the document's own number + heading slug
 │        └── which policy
 └── which institution
```

A citation must be three things at once: readable by a human (so you can look
it up), checkable by a machine (so support can be proven), and **unique**.

Uniqueness is harder than it looks. Real documents restart their numbering —
SRM's scholarship page has several independent numbered lists, so "2" appears
six times meaning six different things — and repeat their headings, with six
separate sections titled "Consequence". A citation that points at six
different passages makes "does this support the claim?" unanswerable. Hence
number *and* heading, with an occurrence suffix as a fallback.

### The two failures, kept separate

Most systems have one bucket labelled "bad citation". These are different
problems with different causes:

| Failure | What happened | Can it be fixed? |
|---|---|---|
| **Fabricated** | The cited section does not exist anywhere. The model invented it. | No — nothing can support a claim pointing at nothing |
| **Misused** | The section is real, but does not support the claim. | Yes — better retrieval or prompting |

Reporting both as "bad citation" hides which problem you actually have.
Telling them apart is much of the value in `verify/citation.py`.

---

## 7. NLI: the fact-checker

**NLI** (Natural Language Inference) is a model that takes two texts and
answers one narrow question: does the first prove the second?

```
premise:     "A student must maintain a minimum attendance record of
              at least 75% in individual courses."
hypothesis:  "The attendance requirement is 75%."
answer:      ENTAILMENT
```

Three possible answers, all useful:

| Label | Meaning | Verdict here |
|---|---|---|
| `ENTAILMENT` | the passage proves the claim | SUPPORTED |
| `CONTRADICTION` | the passage disproves it | REFUTED |
| `NEUTRAL` | it neither proves nor disproves | UNSURE |

`NEUTRAL` is not a failure of the checker. It is the honest answer when a
passage does not address a claim, and it is exactly when the system should
consider staying quiet.

### Why not ask the language model to check its own work?

Because a model grading itself repeats its own mistakes. The same misreading
that produced the wrong claim produces a confident wrong grade. An NLI model
is a genuinely independent second opinion: it is a different model, trained
for a different task, answering one narrow question. It also runs on CPU and
is deterministic, so verdicts do not drift between runs.

### The truncation trap

NLI models have a fixed input limit (512 tokens here). Chunks can exceed it.
If you let the tokenizer truncate, it silently discards the tail of the
passage — possibly the exact sentence that supports the claim. You would get a
confident "unsupported" for a claim the document plainly states.

So passages are split into overlapping windows and each is scored. Support
anywhere in the passage counts as support.

In this codebase: `verify/nli.py`.

---

## 8. Deterministic guards

Not every check needs a model. Some can be done exactly, and where that is
possible it is better — a deterministic check cannot be fooled by fluent
phrasing, because it does not read the sentence at all.

The **numeric guard** asks: does every number in the claim actually appear in
the evidence?

```
claim:    "Students need 80% attendance."
evidence: "...at least 75% in individual courses..."
          → 80% is not present → VETO
```

Two rules govern how it combines with the NLI model:

- **It can veto support but never grant it.** Matching digits is not proof
  that the passage *means* what the claim says. Only NLI grants support.
- **When they disagree about a number, the numbers win.** Fluent agreement
  does not make a figure right.

It deliberately stays dumb: no word-numbers ("two weeks" vs "14 days"), no
unit conversion, no arithmetic. Those need real comprehension, which is the
NLI model's job. A guard that tries to be clever starts producing its own
false alarms, and a false alarm here suppresses a *correct* claim.

In this codebase: `verify/numeric.py`.

---

## 9. Abstention

**Abstention** is the system deciding to say "I don't know" rather than
answer. It is a feature, not a failure.

The logic: after verification, if too little survives — no supported claims,
or retrieval scores too weak to suggest the corpus even covers the question —
the system declines to answer and says why.

### Over-abstention is also a failure

This is the part most projects miss. A system that abstains on *everything*
scores perfectly on hallucination rate and is completely useless. So
abstention quality is measured as a 2×2, not a single number:

|  | System answered | System abstained |
|---|---|---|
| **Question was answerable** | correct | **false abstention** ← failure |
| **Question was unanswerable** | **failure** | correct |

Both off-diagonal cells matter. Tracking only one produces a system that is
either dangerous or useless.

### Transparent correction

Rather than silently deleting claims that failed verification, the system
keeps them in a `claims_removed` list and shows them. The user sees what the
system started to say and then decided it could not stand behind. That record
*is* the transparency feature — it is the difference between a system you can
audit and one you have to trust.

In this codebase: `schema.py` already defines `Answer.claims_removed`;
the logic itself is Phase 5, not yet built.

---

## 10. How the system evaluates itself

Claiming "it works" is not evidence. Five metrics, each answering one
question:

| Metric | Question |
|---|---|
| **Claim verification P/R** | Are "supported" claims really supported? Did we catch all the true ones? |
| **Citation accuracy** | Does the cited section exist *and* support the claim? |
| **Hallucination rate** | What share of claims are not backed by the evidence cited? |
| **Abstention quality** | Does it refuse when it should — and not when it shouldn't? |
| **Latency** | Split by stage, so you know *where* the time goes |

These are measured against an **adversarial set**: questions written
specifically to trip the system up, in five categories —

1. **Cross-regulation confusion** — same question, different right answer
   depending on year or programme
2. **Numeric traps** — thresholds, deadlines, percentages
3. **Unanswerable but plausible** — not covered by the corpus at all
4. **Negation and exceptions** — "except when…", conditional clauses
5. **False premises** — "Why does the policy require X?" when it does not

### Tuning versus reporting

The set is split in half. Thresholds are tuned on one half and results
reported on the other. Tuning and reporting on the same questions would only
prove the system works on the questions it was tuned for — a result that
sounds impressive and means nothing.

---

## 11. Glossary

| Term | Meaning |
|---|---|
| **Abstention** | Declining to answer when evidence is insufficient |
| **Chunk** | A searchable piece of a document; the unit that gets cited |
| **Citation** | An identifier for exactly which section backs a claim |
| **Claim** | One single factual statement extracted from an answer |
| **Cosine similarity** | How close two vectors are; used as the search score |
| **Embedding** | A list of numbers representing the meaning of a text |
| **Entailment** | NLI's term for "the first text proves the second" |
| **Hallucination** | Fluent, confident output not grounded in any source |
| **NLI** | Natural Language Inference — does text A prove text B? |
| **RAG** | Retrieval-Augmented Generation — fetch sources, then answer from them |
| **Spike** | A throwaway experiment run *before* building on an assumption |
| **Vector database** | Storage that finds the nearest vectors quickly |

---

## Where to go next

- [`README.md`](README.md) — what the project is, how to run it
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md) — current state, every finding, what is not built
- `src/policyverify/schema.py` — the data contracts; the best single file to
  read first, because everything else depends on it
