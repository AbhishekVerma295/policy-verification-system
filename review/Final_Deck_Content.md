# Final deck: slide content for approval

Same template and style as `RAI REVIEW 1 PPT.pptx`: the same headings, background, fonts (Abril Fatface titles, Merriweather text), navy accents, corner lines and page tabs. **15 slides.** Every number comes from the code, `config.yaml`, `RESEARCH_PAPER_DATA.md` or the saved result files in `eval/results/`.

**Changes from the current deck, at a glance:**
- **Literature Survey** now uses the papers the project actually uses or references (the code and docs cite them). The six papers on the current slide aren't referenced anywhere in the project, so they're dropped, as you asked.
- **Experimental Setup** gets a third slide on the evaluation sets and how answers are scored.
- **Results and Discussion** becomes three slides: end-to-end results, verifier and ablation, and errors and limitations.
- **Speaker notes:** every slide has notes, with likely questions and answers.

---

## 1. Title (unchanged)

- **Title:** POLICY VERIFICATION SYSTEM
- **Subtitle:** Claim-level hallucination and citation verification for university policy question answering

**Notes:** Our project is a question-answering assistant for SRM's official policies. It checks every sentence it says against the policy text before showing it, and it says so when it can't answer.

---

## 2. Abstract

**Left: text**
- **Problem:** AI chatbots answer policy questions fluently, but a claim can be unsupported or numerically wrong while still looking cited.
- **Approach:** a local question-answering system over six official SRM policies. The language model answers in separate claims, each citing a policy section.
- **Verification:** every claim is checked by an independent NLI model, a number check and a citation check. A relevance check confirms it answers the question.
- **Decision:** keep the claims that pass, show removed ones with the reason, and abstain when evidence is insufficient.
- **Result:** on 40 new held-out questions, 77.5% accuracy and 0% hallucination. With verification switched off, hallucination was 20%.

**Right: panel "Three real answers from the final test run"** (copied from the Run A log)
1. **ANSWERED · SUPPORTED 1.00.** "Without the minimum attendance of 75%, in any course, students become ineligible to appear for the end semester examination in that course." Cites attendance 7.3.
2. **DECLINED · TRUE BUT OFF TOPIC.** Asked whether the hostel promises a minimum Wi-Fi speed. Found: "Free Wi-Fi facility is available to every student for academic purposes". This is true, but it doesn't answer the question, so the system declined.
3. **WITHHELD · REFUTED 0.61.** "…a CUET-PG score at the 87th percentile corresponds to a 75% tuition fee waiver." The policy row it was checked against says 75% is for the 90th–94.99th percentile; the correct answer is 50%.

**Notes:** The panel shows the three outcomes a user can see: a verified answer, an honest decline, and a wrong claim stopped before it reached the user.

---

## 3. Problem Statement & Objectives

**Problem** (unchanged): Policy chatbots can state rules the policy does not support, and the user cannot tell which parts of an answer are actually backed by the text.

**Three failure types** (unchanged):
1. Wrong fact or figure: a threshold, deadline or amount differs from the policy.
2. Citation that doesn't back the claim: the cited section is real but says something else, or does not exist.
3. Answering beyond the policy: a confident answer to a question the documents never cover.

**Objectives:**
1. Retrieve evidence from official SRM policies, with exact section citations.
2. Generate answers as single-fact claims, each citing its source section.
3. Verify every claim independently: entailment, numbers, citation and relevance.
4. Keep only verified claims, show removed ones with reasons, and abstain when evidence is insufficient.
5. Measure accuracy, hallucination, citation accuracy, abstention quality and latency on held-out questions, including a final set of 40 new questions.

**Notes:** Each objective maps to one stage of the pipeline. Objective 5 covers the metrics in the project brief.

---

## 4. Literature Survey

**Table: Paper | Approach | Key finding | How we use it**

| Paper | Approach | Key finding | How we use it |
|---|---|---|---|
| [1] Laban et al. 2022 · TACL. **SummaC** | NLI applied sentence by sentence to check a summary against its source | Sentence-level NLI beats document-level NLI; SummaC-Conv reached 74.4% balanced accuracy on a 6-dataset benchmark | Our verifier checks each claim against single sentences and 2–3 sentence windows of the cited section |
| [2] Tang, Laban & Durrett 2024 · EMNLP. **MiniCheck** | Small fact-checking models trained on synthetic data to check LLM output against grounding documents | A 770M-parameter checker reaches GPT-4-level accuracy at about 400× lower cost | Supports using a small separate verifier; the planned baseline for comparison (not run yet) |
| [3] Cormack, Clarke & Büttcher 2009 · SIGIR. **Reciprocal Rank Fusion** | Merge several rankings by summing 1 / (k + rank) | Beat Condorcet fusion and individual learning-to-rank methods | Merges embedding search and keyword search (k = 60) |
| [4] Robertson & Zaragoza 2009 · FnTIR. **BM25** | Probabilistic keyword ranking with term-frequency saturation and length normalisation | The standard keyword baseline in information retrieval | The keyword half of our hybrid search (k1 = 1.5, b = 0.75) |
| [5] Guo et al. 2017 · ICML. **Calibration** | Measures how well model confidence matches accuracy; proposes temperature scaling | Modern neural networks are over-confident; temperature scaling corrects much of it | Calibration check on verifier scores: T = 0.766 fitted, not applied (scores already well separated) |
| [6] He, Gao & Chen 2023 · ICLR. **DeBERTaV3** | ELECTRA-style pre-training with gradient-disentangled embedding sharing | Improves on DeBERTa across language-understanding benchmarks | Our NLI verifier is a DeBERTa-v3-base checkpoint fine-tuned on MNLI, FEVER and ANLI |

**Notes:** These are the papers whose methods are inside the system. Gap they leave for policy QA: none of them combines claim-level checking, citation checking and abstention in one local pipeline for institutional rules.

Figures checked: SummaC 74.4% (arXiv abstract), MiniCheck 400× and 770M (local PDF).

---

## 5. Proposed Methodology

**Four steps** (cards):
1. **Retrieve:** embedding + keyword search, merged; 6 passages, at most 2 per section.
2. **Generate:** Qwen3 writes single-fact claims as JSON, each citing a section.
3. **Verify:** for each claim, a citation check, sentence-level NLI and a number check, then a relevance check.
4. **Decide:** keep supported, relevant claims; show removed ones with reasons; abstain if none survive.

**Component table: Component | Choice | Why**

| Component | Choice | Why |
|---|---|---|
| Embedding model | BAAI/bge-base-en-v1.5 (768-d, CPU) | Open retrieval model; runs on CPU so the GPU stays free for the LLM |
| Retrieval | Chroma (cosine) + BM25, merged by Reciprocal Rank Fusion | Embedding search alone missed sections worded differently from the question |
| Generator | Qwen3 4B via Ollama · temperature 0 · JSON schema | Fits an 8 GB laptop GPU; structured claims with citations |
| Verifier | DeBERTa-v3-base-mnli-fever-anli (CPU), sentence by sentence | Independent of the generator; chosen after a 20-pair comparison of 3 NLI models |
| Numeric guard | Exact number matching (%, amounts, clock times) | Figures decide outcomes; can block support, never grant it |
| Relevance gate | Question–claim similarity, then a yes/no LLM check | A claim can be true and still not answer the question |
| Thresholds | NLI 0.5 · keep 0.5 · relevance 0.85 / 0.55 | Chosen on tuning questions only; NLI 0.5 kept after a calibration check |

---

## 6. Architecture Diagram

The simple high-level diagram (`review/Architecture_Basic_v2.png`):
- **Top row, data preparation (offline):** SRM policies → text extraction → chunking → embedding → Chroma.
- **Bottom row, question answering (online):** Student → Retriever → LLM generator → **Verifier ★** → Decision → Verified answer.

**Notes:** The 30-second walkthrough from earlier, with today's numbers (212 sections, 262 chunks, hybrid search, relevance check).

---

## 7. Low-Level Design: Preparation & Retrieval

`review/LLD_Slide1_Simple_v2.png`: 7 cards with IN / OUT. The detailed version stays as a backup.

## 8. Low-Level Design: Generation & Verification

`review/LLD_Slide2_Simple_v2.png`: 8 cards, including the relevance gate. The detailed version stays as a backup.

---

## 9. Experimental Setup: Data

**Corpus: 6 official SRM policies**

| Policy | Source document | Format | Sections | Chunks |
|---|---|---|---|---|
| Attendance | Academic Regulations 2021 | PDF | 52 | 66 |
| Examination | Examination Policy | HTML | 59 | 88 |
| Academic integrity | Plagiarism Policy (2017) | PDF | 9 | 11 |
| Scholarship | Scholarship Policy | HTML | 29 | 39 |
| Residence | Hostel Rules (Jan 2025) | PDF | 50 | 38 |
| Code of conduct | Code of Conduct for Students | HTML | 13 | 20 |
| **Total** | | | **212** | **262** |

Note under the table: 192 of the 212 sections are indexed (contents-page entries and bare titles are dropped). Each table row is stored as one self-describing line.

**Dataset preparation** (4 steps): Official SRM documents → Text extraction and cleaning → Section splitting and chunking → 262 chunks indexed (Chroma + BM25).

---

## 10. Experimental Setup: Evaluation (new slide)

**Three labelled sets**

| Set | Size | Used for |
|---|---|---|
| Adversarial questions | 64 (33 tuning / 31 held-out) | tuning; Run B |
| Final held-out questions | 40 | Run A |
| Labelled claims | 85 (44 tuning / 41 held-out) | verifier benchmark, Run C |

**Question categories** (tuning / held-out 31 / new 40): numeric 9 / 7 / 17 · unanswerable 7 / 7 / 6 · false premise 6 / 6 / 6 · yes/no with exception 6 / 6 / 6 · cross-policy 5 / 5 / 5.

**How a question is scored:**
- **Correct:** answered, right section cited, the required value present and no forbidden value. A correct decline on an unanswerable question also counts.
- **Hallucination:** a kept claim contains a known-wrong value, or the system answers a question the policies don't cover.
- **Intervals:** 95% Wilson intervals.
- **Tuning discipline:** thresholds were chosen on tuning questions only. The system was frozen, then each held-out set was run once.

**Notes:** The final 40 are the cleanest test: a separate held-out set, with the answer key fixed before the first run.

---

## 11. Experimental Setup: Environment and Configuration

**Models (no fine-tuning)**

| Role | Model | Size | Runs on |
|---|---|---|---|
| Generator / judge | Qwen3 4B, 4-bit (Q4_K_M), Ollama | 4.0 B | GPU |
| Embedder | BAAI/bge-base-en-v1.5 | 110 M | CPU |
| Verifier (NLI) | DeBERTa-v3-base-mnli-fever-anli | 184 M | CPU |

**Hardware and software:**
- **Hardware:** Intel Core i7-13620H · RTX 4060 Laptop 8 GB · 16 GB RAM · Windows 11.
- **Software:** Python 3.12.10 · PyTorch 2.13 (CPU) · Transformers 5.15 · Sentence-Transformers 5.7 · Chroma 1.5.9 · Ollama 0.32.5 · PyMuPDF 1.28 · Streamlit 1.61.

**Fixed configuration:**
- **Chunking:** ≤ 1,800 characters per chunk, 150-character overlap.
- **Retrieval:** embedding + BM25, merged by RRF (k = 60); top 6, at most 2 per section.
- **Generation:** temperature 0, JSON schema, thinking off; 8,192-token context; 1 retry on bad JSON.
- **Prompt:** passages shown whole (16,000-character budget); instruction-like text screened out.
- **Verification:** NLI per sentence and 2–3 sentence windows; support ≥ 0.5; numbers must match.
- **Abstention:** keep SUPPORTED claims ≥ 0.5 that also pass the relevance gate.

**Footer:** Temperature 0; reruns vary by about 1 question in 33 · Timing: measured warm, model loading excluded · 472 automated tests, no GPU, model or network.

*GPU memory (3.9 GB) is dropped:* it was measured before the current pipeline and hasn't been re-measured.

---

## 12. Results and Discussion: End-to-End (Runs A and B)

**Table: Metric | New 40 (Run A) | Original 31 (Run B, corrected labels)**

| Metric | New 40 (Run A) | Original 31 (Run B, corrected labels) |
|---|---|---|
| Accuracy | **77.5%** (31/40) [62.5–87.7] | **74.2%** (23/31) [57–86] |
| Hallucination rate | **0%** (0/40) [0–8.8] | **3.2%** (1/31) [1–16] |
| Unanswerable declined | 6/6 | 6/6 |
| Answerable wrongly declined | 7/34 (20.6%) | 5/25 (20.0%) |
| Citation accuracy | 96.3% | 90.0% |
| Claims generated / kept / removed | 87 / 52 / 35 | 88 / 31 / 57 |

**Speed tiles, median per question (Run A):** retrieval 0.04 s · generation 2.8 s · verification + relevance 2.4 s · **total 5.9 s** (p95 16.2 s).

**Notes:**
- How these map to the brief: hallucination rate, citation accuracy, abstention quality and latency.
- Claim precision and recall are on the next slide.
- The hand reading is stricter: checking all 40 answers by hand against the answer key gives 26/40 (65%) fully correct. Say this before the panel asks.

---

## 13. Results and Discussion: Verifier and Ablation

**Left: verifier alone, 41 held-out claims (Run C)**
- **Precision** 88.5% [71–96] · **Recall** 100% [86–100] · **F1** 93.9%.
- **Confusion matrix:** true supported 23 → accepted 23, rejected 0; not supported 18 → accepted 3, rejected 15.
- **By kind:** wrong number 4/4, negation 3/3, wrong entity 2/2, unsupported 3/3 and wrong section 3/3 caught. Wrong table row ("role"): **0/3 caught**.
- **Speed:** about 1.1 s per claim on CPU.

**Right: verification on vs off** (same model output, re-scored)

| | Run A on | Run A off | Run B on | Run B off |
|---|---|---|---|---|
| Hallucination | 0% | 20.0% | 3.2% | 19.4% |
| Accuracy | 77.5% | 75.0% | 74.2% | 71.0% |
| Unanswerable declined | 100% | 0% | – | – |
| Answerable declined | 20.6% | 0% | – | – |

**Main finding:** verification removes almost all hallucination. The cost is declining about 1 in 5 answerable questions.

**Notes:** A component-by-component ablation was **not** measured. Only on vs off was. Don't claim more.

---

## 14. Results and Discussion: Errors and Limitations

**Where it fails** (from the error analysis):

| Failure | Evidence | Cause |
|---|---|---|
| Wrong table row accepted | 3/3 "role" claims accepted | NLI matches the number and ignores the row |
| Correct table look-ups rejected | tm-33, tm-35 | NLI cannot reason that 8,500 lies in 7,001–9,000 |
| Relevance gate too strict | 6 false-premise questions declined | The correct answer doesn't echo the question's wording |
| Half answers on cross-policy questions | tm-22 to tm-25 | One of the two meanings fails verification |
| Wrong section cited | xpol-002, xpol-008, tm-04 | A neighbouring section contains the same figure |

**Limitations:** small test sets (wide intervals) · one institution, six documents · the automatic scorer is lenient (65% by hand) · no external baseline yet · one small 4B model.

**Next steps** (from the project plan): compare with MiniCheck · measure each component's effect · scope and date labels per document.

---

## 15. References

**A. Research papers**
1. P. Laban et al., "SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization," TACL, 2022.
2. L. Tang, P. Laban and G. Durrett, "MiniCheck: Efficient Fact-Checking of LLMs on Grounding Documents," EMNLP, 2024.
3. G. V. Cormack, C. L. A. Clarke and S. Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and Individual Rank Learning Methods," SIGIR, 2009.
4. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in Information Retrieval, 2009.
5. C. Guo et al., "On Calibration of Modern Neural Networks," ICML, 2017.
6. P. He, J. Gao and W. Chen, "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing," ICLR, 2023.

**B. Models and tools**
7. Qwen Team, Qwen3 model documentation.
8. S. Xiao et al., "C-Pack: Packaged Resources To Advance General Chinese Embedding" (BGE models), 2023.
9. M. Laurer, DeBERTa-v3-base-mnli-fever-anli model card, Hugging Face.
10. Chroma and Ollama documentation.
11. E. B. Wilson, "Probable Inference, the Law of Succession, and Statistical Inference," JASA, 1927 (confidence intervals).

**C. Data**
12. SRM Institute of Science and Technology, official policy documents (URLs, access dates and checksums in `data/manifest.yaml`).
