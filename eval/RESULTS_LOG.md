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
