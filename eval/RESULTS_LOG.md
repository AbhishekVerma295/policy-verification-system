# Results log: before / after each change

One row per evaluation run, in order. Every run uses the same fixed
experiment settings, so the only thing that changes between rows is the
change named in the row.

## Fixed experiment settings

| Setting | Value |
|---|---|
| Machine | Laptop, AMD Ryzen 7 5700U, 13.8 GB RAM, **no GPU** |
| Language model | `qwen2.5:3b` (Q4_K_M) via Ollama, from `config.local.yaml` |
| Temperature / seed | 0.001 / 42 |
| Max reply tokens / context | 1500 / 8192 |
| Embedding model | `BAAI/bge-base-en-v1.5` (CPU) |
| NLI model | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (CPU) |
| Chunking | 800 chars, 120 overlap, one summary per section |
| Retrieval | hybrid (embedding + BM25, RRF k=60), top_k 6, max 2 chunks per section, entity boost cap 0.05 |
| Audience filter | off |
| Abstention | min_claim_score 0.5 |

The numbers below are **not** comparable with `main`'s reported 74.2% on the
held-out half: that run used `qwen3:4b` on a GPU machine with 1800-character
chunks. These rows are comparable with each other.

Tuning runs use `--split tuning` (33 questions). Held-out (`--split heldout`)
is run once at the end, for reporting.

## Tuning split (adversarial set, 33 questions)

| # | Change | Accuracy | Halluc. | False abst. | Correct abst. | Citation acc. | Run |
|---|---|---|---|---|---|---|---|
| 0 | Baseline: re-ingested corpus, 800-char chunks, entity boost, hybrid search | 51.5% | 0.0% | 57.7% (15/26) | 100% (7/7) | 90.9% | `run-20261120-105410` |
| 1 | + cross-encoder relevance judge, judge views, situation prompt, claim de-dup, citation-label fix, number-word fix, PDF tables | **66.7%** | 6.1% (2) | **30.8%** (8/26) | 85.7% (6/7) | 94.4% | `run-20261120-123500` |

### Run 1, reviewed by hand

The two items counted as hallucinations:

- `num-006` "How long may a parent stay in the hostel as a guest?" - the kept
  claim quotes the policy: "a short stay (one or two days only) ... intimate
  the hostel office at least 7 days before". The answer is right and states
  7 days in its true role (notice period); the scorer counts any answer
  containing the forbidden string "7 days". **Not a hallucination on manual
  review.** The scorer is left unchanged so rows stay comparable.
- `un-011` "Which plagiarism detection software does the university use?" -
  answered from "SRM ... will use the best tools / software to detect
  plagiarism" (cross-encoder p = 0.997). **A real failure**, and the one the
  judge comparison predicted: the claim mentions software and names none.
  Fixed after this run by a name check (`names_something_new`): a "which X" /
  "who is" question must be answered by a name that is neither the
  question's own words nor the institution's name. Measured on the two
  answerable name-questions in the sets (sc-013 is "who should I contact",
  not a name question; sc-019's answer names the SRM Merit Scholarship).

Remaining false abstentions (8): `xpol-003`, `xpol-007`, `xpol-009` (vague
cross-policy questions), `neg-005` (the model wrote the wrong sentence from
the right passage), `neg-009`, `fp-001`, `fp-005`, `fp-011` (false-premise
questions, where the claim that corrects the premise scores low against the
question as asked).

### What run 0 showed

- 15 false abstentions. In 11, every removed claim was **SUPPORTED**
  (entailment 1.0, numbers and citation fine) and the relevance judge
  (`qwen2.5:3b`) said it did not answer the question - for example "Are
  hostel fees refundable?" -> "Fees once paid will not be refunded
  normally". The judge reads a "no" ruling as "no answer".
- 1 (`xpol-005`): three true claims cited `"citation_id: srm/..."` - the
  label copied from the prompt - and lost their citations.
- Retrieval is not the bottleneck: the expected section is in the top 6
  for 48 of 51 answerable adversarial questions and 21 of 21 scenario
  questions.

## Relevance judge comparison (2026-11-20)

27 hand-labelled cases: the 14 judge calls logged in run 0 (all real
answers), 4 more real answers (false-premise corrections, scenarios), and 9
true-but-off-topic traps written from the corpus (e.g. "Which plagiarism
software?" -> "SRM will use the best tools / software to detect
plagiarism"). Cases: a scratch file built from data/logs/prompts/.

| Judge | Answers kept (of 18) | Traps rejected (of 9) | Total (of 27) |
|---|---|---|---|
| qwen2.5:3b, original prompt, JSON schema | 6 | 9 | 15 |
| ... same prompt, plain JSON mode | 6 | 9 | 15 |
| ... same prompt, plain-text YES/NO | 6 | 9 | 15 |
| ... plus "a 'no' is an answer too" | 1 | 9 | 10 |
| ... one statement at a time | 6 | 9 | 15 |
| ... "copy the words that answer it" | 6 | 9 | 15 |
| **cross-encoder `ms-marco-MiniLM-L-6-v2`, p >= 0.3** | **18** | **7** | **25** |
| cross-encoder after an LLM rewrite of the question | 17 | 7 | 24 |

The 3B model's failures do not depend on wording or output format: it
reads a "no" ruling ("Fees once paid will not be refunded") as "not an
answer". The cross-encoder's two misses are both questions asking for a
NAME ("which software", "who is the current VC") answered by a claim that
mentions the thing. It also failed on a scenario phrased as a count with a
student's word for the outcome ("74 classes out of 100 ... detained",
p = 0.000); giving the judge a normalised copy - the count as a percentage,
"detained" replaced by the examination policy's own definition - raised it
to 0.976 (`judge_views` in verify/relevance.py).
