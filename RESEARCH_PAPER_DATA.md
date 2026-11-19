# Research paper data pack

**Project:** Hallucination and Citation Verifier for Policy Question Answering
**Prepared:** 16 November 2026, from code at commit `3782df6`
**Purpose:** everything needed to write the experiments, dataset and results sections.

**How to read the numbers.** Every number in this file is **measured** and comes
from a saved result file named beside it, unless it is explicitly marked
otherwise. Ranges in square brackets are 95% Wilson confidence intervals. The
test sets are small, so the intervals are wide; please quote them with the
headline figures.

---

## 1. Final experimental results

### 1.1 The three final runs

Each was run **once**, after the system was frozen. Nothing in the system was
changed after the final questions were seen.

| Run | What it measures | Items | Result file |
|---|---|---|---|
| A | End to end, **new 40 questions** written and answered by a teammate who did not build the system | 40 | `eval/results/run-20261116-104137.summary.json` |
| B | End to end, **original held-out 31** questions | 31 | `eval/results/run-20261116-104643.summary.json` |
| C | Verifier alone, **held-out claims** | 41 | `eval/results/verifier-20261116-105128.summary.json` |

### 1.2 End-to-end metrics (Runs A and B)

| Metric | Run A: new 40 | Run B: original 31 |
|---|---|---|
| Overall accuracy | **77.5%** [62.5-87.7] (31/40) | **74.2%** [56.8-86.3] (23/31) |
| Hallucination rate | **0.0%** [0.0-8.8] (0/40) | **6.5%** [1.8-20.7] (2/31) |
| Wrong when answering | 7.4% [2.1-23.4] (2 of 27 answers) | 20.0% [8.1-41.6] (4 of 20 answers) |
| Correct abstention (unanswerable declined) | 100% (6/6) | 87.5% (7/8) |
| False abstention (answerable declined) | 20.6% (7/34) | 17.4% (4/23) |
| Citation accuracy (answers resting on an expected section) | 96.3% | 89.5% |
| Claim-level citation precision | 86.5% | 86.7% |
| Claim-level citation recall | 61.8% | 70.8% |
| Claims generated / kept / removed | 87 / 52 / 35 (40.2% removed) | 88 / 31 / 57 (64.8% removed) |

Combined over all 71 held-out questions (computed from the two rows above):
accuracy 54/71 = 76.1%, hallucination 2/71 = 2.8%, unanswerable declined 13/14.

### 1.2a Run B after correcting two wrong labels (use these for Run B)

Two labels in the original held-out set were found to be wrong (details in
section 2.6). The saved answers of Run B were scored again against the
corrected labels; **no model was re-run**. File:
`eval/results/run-20261116-104643.relabelled.summary.json`.

| Metric | Run B as first scored | Run B, corrected labels |
|---|---|---|
| Overall accuracy | 74.2% (23/31) | **74.2%** [57-86] (23/31) |
| Hallucination rate | 6.5% (2/31) | **3.2%** [1-16] (1/31) |
| Wrong when answering | 20.0% (4 of 20) | 15.0% [5-36] (3 of 20) |
| Correct abstention | 87.5% (7/8) | 100% (6/6) |
| False abstention | 17.4% (4/23) | 20.0% (5/25) |
| Citation accuracy | 89.5% | 90.0% |
| Accuracy, verification on vs off | 74.2% vs 67.7% | 74.2% vs 71.0% |
| Hallucination, verification on vs off | 6.5% vs 25.8% | 3.2% vs 19.4% |

What moved: `un-014` (dress code) went from "answered an unanswerable
question" to correct, and `un-010` (scholarship deadline) went from a correct
decline to a wrong decline. Accuracy is unchanged because the two cancel.
Corrected category scores: numeric 6/8, negation_exception 7/7,
unanswerable 5/5, cross_policy 3/5, false_premise 2/6. With the corrected
labels, the combined 71 held-out questions give accuracy 54/71 = 76.1% and
hallucination 1/71 = 1.4%. Sections 1.3, 1.4 and 1.5 below show Run B as
first scored.

### 1.3 Outcome counts (the end-to-end "confusion matrix")

Every question ends in exactly one outcome.

| Outcome | Meaning | Run A | Run B |
|---|---|---|---|
| `correct` | answered, right section, required value present, no forbidden value | 25 | 16 |
| `correct_abstention` | unanswerable question declined | 6 | 7 |
| `false_abstention` | answerable question declined | 7 | 4 |
| `incomplete` | answered, but a required value is missing | 1 | 0 |
| `wrong_citation` | answered, but no kept claim cites an expected section | 1 | 2 |
| `wrong_value` | a kept claim contains a forbidden (known-wrong) value | 0 | 1 |
| `answered_unanswerable` | unanswerable question answered | 0 | 1 |
| **Total** | | **40** | **31** |

Answer / abstain matrix:

| | Run A: answered | Run A: declined | Run B: answered | Run B: declined |
|---|---|---|---|---|
| Answerable question | 27 | 7 | 19 | 4 |
| Unanswerable question | 0 | 6 | 1 | 7 |

### 1.4 Accuracy by question category

| Category | Run A: new 40 | Run B: original 31 |
|---|---|---|
| Numeric (includes table look-ups and plain facts in Run A) | 10/17 (58.8%) | 6/7 (85.7%) |
| Unanswerable | 6/6 (100%) | 6/7 (85.7%) |
| False premise | 4/6 (66.7%) | 2/6 (33.3%) |
| Yes/no with exception | 6/6 (100%) | 6/6 (100%) |
| Cross-policy | 5/5 (100%) | 3/5 (60.0%) |

(In Run B one cross-policy item, `xpol-010`, is labelled unanswerable, which is
why Run B has 8 unanswerable questions but 7 in the "unanswerable" category.)

### 1.5 Ablation: verification on versus off

Same model output, re-scored as if every generated claim had been kept
(no verifier, no relevance gate, no abstention).

| Metric | Run A on | Run A off | Change | Run B on | Run B off | Change |
|---|---|---|---|---|---|---|
| Overall accuracy | 77.5% | 75.0% | +2.5 | 74.2% | 67.7% | +6.5 |
| Hallucination rate | 0.0% | 20.0% | -20.0 | 6.5% | 25.8% | -19.4 |
| Citation accuracy | 96.3% | 94.1% | +2.2 | 89.5% | 91.3% | -1.8 |
| Correct abstention | 100% | 0% | +100 | 87.5% | 37.5% | +50.0 |
| False abstention | 20.6% | 0% | +20.6 | 17.4% | 0% | +17.4 |

**Main finding:** verification removes almost all hallucination (20% to 0%
and 26% to 6.5%) at the cost of declining roughly one in five answerable
questions. The gain in overall accuracy is small because the two effects
partly cancel.

**Not measured:** a component-by-component ablation (hybrid search off,
relevance gate off, numeric guard off, and so on). Only the on/off comparison
above exists. Do not present a per-component table as measured.

### 1.6 Verifier alone: claim-level results (Run C, 41 held-out claims)

| Metric | Value |
|---|---|
| Precision | **88.5%** [71.0-96.0] |
| Recall | **100%** [85.7-100] |
| F1 | 93.9% |
| False-accept rate | 16.7% [5.8-39.2] (3 of 18) |
| False-reject rate | 0.0% [0.0-14.3] (0 of 23) |
| Contradictions named as REFUTED | 9 of 12 (75%) |
| Time per claim | about 1.1 s on CPU |

Confusion matrix (accepted = verdict SUPPORTED):

| | Verifier accepted | Verifier rejected |
|---|---|---|
| Claim truly supported (23) | 23 (TP) | 0 (FN) |
| Claim not supported (18) | 3 (FP) | 15 (TN) |

By kind of claim:

| Kind | What it is | Correct | Supported / Neutral / Refuted |
|---|---|---|---|
| paraphrase | true claim, reworded | 16/16 | 16 / 0 / 0 |
| verbatim | true claim, quoted | 7/7 | 7 / 0 / 0 |
| number | wrong number | 4/4 | 0 / 0 / 4 |
| negation | meaning reversed | 3/3 | 0 / 0 / 3 |
| entity | wrong person or body | 2/2 | 0 / 0 / 2 |
| unsupported | section is silent | 3/3 | 0 / 3 / 0 |
| wrong_section | true elsewhere, not in the cited section | 3/3 | 0 / 2 / 1 |
| role | right number attached to the wrong row or role | **0/3** | 3 / 0 / 0 |

The three false accepts are all of one kind:

- `rol-002` "A parent may stay in the hostel as a guest for up to 7 days." (7 days is the notice period, not the stay)
- `rol-004` "A CLAT rank in the top 2500 receives a 75% tuition fee waiver." (top 2500 gets 100%)
- `rol-006` "A Minor requires 7.0 additional credits."

For reference, the full 85-claim set (tuning + held-out, run 16 Nov, not saved):
precision 92.2% [82-97], recall 100% [92-100], false-accept 10.5% [4-24],
confusion TP 47 / FP 4 / FN 0 / TN 34.

### 1.7 Latency (milliseconds per question)

| Stage | Run A mean | Run A median | Run A p95 | Run B mean | Run B median | Run B p95 |
|---|---|---|---|---|---|---|
| Retrieve | 331 | 42 | 81 | 563 | 44 | 60 |
| Generate | 3439 | 2822 | 5132 | 4119 | 2491 | 5664 |
| Verify (includes relevance gate) | 3647 | 2445 | 8911 | 2979 | 1989 | 6861 |
| **Total** | **7417** | **5921** | **16219** | **7661** | **5011** | **25737** |

The retrieve mean is inflated by the first question of each run (model
loading); the median is the fair figure. Hardware is in section 4.3.

### 1.8 Hand reading of Run A against the teammate's answer key

The automatic scorer checks the cited section and listed values. It cannot
judge whether an answer is complete. Reading all 40 answers by hand against
the teammate's key gives a stricter figure:

| | Count |
|---|---|
| Fully correct by the key (20 answers + 6 correct declines) | **26 of 40 (65%)** |
| Scorer said correct, but the answer gives only half of what the key asks | 5 (`tm-15`, `tm-22`, `tm-23`, `tm-24`, `tm-25`) |
| Correct answer was generated but blocked by the relevance gate | 3 (`tm-11`, `tm-12`, `tm-30`) |
| Correct answer was generated but rejected by the verifier (table row) | 2 (`tm-33`, `tm-35`) |
| Wrong answer was generated and rightly stopped by the verifier | 2 (`tm-34`, `tm-36`) |
| Right fact, wrong section cited, verifier accepted it | 1 (`tm-04`) |
| Listed the four committee members but never said "four" | 1 (`tm-03`) |
| False statement shown to the user as verified | **0** |

Two points worth a sentence in the paper:

- The two "false abstentions" `tm-34` and `tm-36` are the system working as
  designed: the language model read the wrong table row and the verifier
  stopped it. The scorer counts them against the system.
- For `tm-08` the model invented "Rs 1,000 for damaging the lift" and the
  verifier weakly accepted it (score 0.56). Only the relevance gate stopped
  it. This is a near miss, not a clean pass.

### 1.9 Development trajectory (how the numbers moved)

Tuning split (33 questions). These are development numbers: settings were
adjusted against these questions, so they are optimistic.

| | Audit, 14 Nov | Final, 16 Nov (`run-20261116-084951`) |
|---|---|---|
| Accuracy | 57.6% | 78.8% [62-89] |
| Hallucination rate | 18.2% | 3.0% [0.5-15] |
| Unanswerable declined | 2 of 7 | 6 of 7 |
| Answerable wrongly declined | 4 of 26 | 2 of 26 |
| Accuracy, verification on vs off | equal | 78.8% vs 69.7% |
| Hallucination, verification on vs off | 18.2% vs 21.2% | 3.0% vs 18.2% |

Original held-out 31, before and after:

| | Audit, 14 Nov | Final, 16 Nov (Run B) |
|---|---|---|
| Accuracy | 64.5% | 74.2% |
| Hallucination rate | 19.4% | 6.5% |
| Unanswerable declined | 3 of 8 | 7 of 8 |

Verifier alone, full 85 claims: precision / recall 91.5% / 91.5% at the audit,
92.2% / 100% at the end; verbatim quotes accepted 12 of 15, then 15 of 15;
2.9 s per claim, then about 0.9 s.

What changed between the two columns, in order: sentence-level evidence for
NLI; hybrid BM25 + dense retrieval; schema-constrained generation; a relevance
gate; ingestion repair (tables, headings); structure-aware evidence units;
repair of clause-number citations; calibration check (thresholds kept).

### 1.10 Calibration of the verifier score (tuning claims only)

44 tuning claims. Fitted and scored on the same claims, so optimistic.

| | Before | After temperature scaling (T = 0.766) |
|---|---|---|
| Expected calibration error (5 bins) | 0.032 | 0.028 |
| Brier score | 0.015 | 0.015 |

Scores are bimodal: 19 claims fall in 0.0-0.2 (none truly supported) and 23 in
0.8-1.0 (all truly supported); only two claims sit in between. Temperature
scaling was therefore **not applied**.

Risk-coverage on the same claims:

| Entailment threshold | Accepted | Wrong among accepted | True claims kept |
|---|---|---|---|
| 0.3 / 0.4 / 0.5 (used) | 25 | 1 (4.0%) | 24/24 |
| 0.6 | 24 | 1 (4.2%) | 23/24 |
| 0.7 / 0.8 | 23 | 0 | 23/24 |
| 0.9 / 0.95 | 22 | 0 | 22/24 |
| 0.98 | 20 | 0 | 20/24 |
| 0.99 | 17 | 0 | 17/24 |

Reproduce with `python scripts/calibrate.py`.

### 1.11 Screenshots

No screenshots are stored in the repository. To take them for the paper, run
`streamlit run app.py` and ask these three questions, which show the three
behaviours worth a figure:

1. "What's the minimum attendance I need in each course to sit for its end-sem exam?" - a normal verified answer with citations (`tm-01`).
2. "Does the hostel policy promise a minimum Wi-Fi speed, like 50 Mbps, to every resident?" - a decline, with the true-but-off-topic statement shown (`tm-06`).
3. "For the 5-year integrated Law programme, a CLAT rank of 4,200 falls in which waiver row?" - a wrong claim refuted, with the contradicting policy row quoted (`tm-34`).

Real outputs from Run A, copied from the run log, usable as text figures:

```
Q (tm-19): Can I appeal every disciplinary penalty, even if it's a low-level offence?
  SUPPORTED 0.97  Students can appeal disciplinary penalties if they are not satisfied with the
                  decision of the Disciplinary Committee and the sanctions imposed by it.
  SUPPORTED 0.99  Appeals are permitted for misconduct at Level 5 and Level 6 only.
  cited: srm/code_of_conduct/7-appeal_process

Q (tm-06): Does the hostel policy promise a minimum Wi-Fi speed, like 50 Mbps, to every resident?
  DECLINED. "The policies do not appear to contain what was asked: 1 statement(s) found were
  supported by the policy but do not answer the question."
  shown as true but off topic: Free Wi-Fi facility is available to every student for academic purposes

Q (tm-34): For the 5-year integrated Law programme, a CLAT rank of 4,200 falls in which waiver row?
  REFUTED 0.96    ... a CLAT rank of 4,200 qualifies for a 50% tuition fee waiver.
  DECLINED. "Not enough of this answer could be verified - all 1 were contradicted by the policy text."
  (correct answer by the key: 75%, rank 2,501-5,000)
```

---

## 2. Dataset details

### 2.1 Corpus

One institution (SRM Institute of Science and Technology, Kattankulathur), six
official policy documents.

| Policy type | Source document | Format | Sections parsed | Chunks |
|---|---|---|---|---|
| attendance | Academic Regulations 2021 (UG / Integrated PG) | PDF | 52 | 66 |
| examination | Examination Policy | HTML | 59 | 88 |
| academic_integrity | Plagiarism Policy (2017) | PDF | 9 | 11 |
| scholarship | Scholarship Policy | HTML | 29 | 39 |
| residence | Hostels Rules and Regulations (Jan 2025) | PDF | 50 | 38 |
| code_of_conduct | Code of Conduct for Students | HTML | 13 | 20 |
| **Total** | | | **212** | **262** |

192 of the 212 sections are indexed (contents-page entries and bare titles are
dropped). Chunks are at most about 1800 characters; tables are stored one
self-describing row per line. Source URLs, access dates and checksums are in
`data/manifest.yaml`. The documents themselves are not in the repository.

### 2.2 The three labelled sets

| Set | File | Size | Split | Written by | Used for |
|---|---|---|---|---|---|
| Adversarial questions | `eval/adversarial.jsonl` | 64 | 33 tuning / 31 held-out | project author | development (tuning) and Run B (held-out) |
| Final held-out questions | `eval/final_heldout.jsonl` | 40 | all held-out | independent teammate | Run A |
| Labelled claims | `eval/claims.jsonl` | 85 | 44 tuning / 41 held-out | project author | verifier-only benchmark, Run C |

Category counts:

| Category | Adversarial tuning | Adversarial held-out | Final 40 |
|---|---|---|---|
| numeric | 9 | 7 | 17 |
| unanswerable | 7 | 7 | 6 |
| false_premise | 6 | 6 | 6 |
| negation_exception (yes/no with exception) | 6 | 6 | 6 |
| cross_policy | 5 | 5 | 5 |
| **Total** | **33** | **31** | **40** |
| of which answerable / unanswerable | 26 / 7 | 23 / 8 | 34 / 6 |

Claim set, by kind (tuning / held-out): paraphrase 16/16, verbatim 8/7,
number 4/4, negation 4/3, role 3/3, entity 3/2, unsupported 3/3,
wrong_section 3/3. Supported 24/23, not supported 20/18.

What the five question categories test:

- **numeric** - a specific figure; the trap is a nearby but wrong figure.
- **unanswerable** - the policies do not say; the system should decline.
- **false_premise** - the question assumes something untrue; the system should correct it.
- **negation_exception** - a yes/no question whose honest answer has a condition.
- **cross_policy** - the same number or word means different things in two policies.

### 2.3 Label format

Each question is one JSON line:

```json
{"id": "tm-19", "question": "Can I appeal every disciplinary penalty, even if it's a low-level offence?",
 "category": "negation_exception", "split": "heldout", "answerable": true,
 "expected_citations": ["srm/code_of_conduct/7-appeal_process"],
 "required_values": ["5", "6"], "forbidden_values": [],
 "note": "No: appeals only for misconduct at Level 5 and Level 6."}
```

- `expected_citations` - the policy section(s) that contain the answer. Any one is enough.
- `required_values` - values a correct answer must contain.
- `forbidden_values` - known-wrong values; if one appears in a claim the system stood behind, that is a hallucination.
- `note` - the expected answer in words (for a human reader; not used in scoring).

Each claim is one JSON line with `claim`, `citation_ids`, `label`
(`supported` / `not_supported`), `kind` and `split`.

### 2.4 How the final 40 were produced

1. A teammate who did not build the system wrote 40 questions from the six
   policy documents, without seeing the system's outputs.
2. The same teammate wrote an independent answer key: correct answer, source
   document and clause, exact supporting quote, and a "wrong answer to watch
   for" (`eval_docs/SRM_Independent_Teammate_Answer_Key_40.docx`).
3. The key was converted to the scoring format **before the system was run on
   any of the questions**. Only section ids and checkable values were added;
   the correct answers are the teammate's.
4. Every expected section was checked to exist in the index, and every
   required value to appear in the cited text.
5. The system was frozen (commit `3782df6`) and run once.

An earlier 20-question version existed; 3 of its questions were found to be
rewordings of tuning questions and were replaced in the 40-question version.
On the 17 questions shared between the two versions, an answer key drafted
separately by the system's author agreed with the teammate's on every one.

### 2.5 The final 40 questions, expected answers and results

Expected answers are the teammate's. Outcome is from Run A.

| ID | Category (teammate's label) | Question | Expected answer | Outcome |
|---|---|---|---|---|
| tm-01 | Numeric | What's the minimum attendance I need in each course to sit for its end-sem exam? | 75% in each course | correct |
| tm-02 | Table lookup | A student applying to the 5-year integrated Law programme has a CLAT rank of 6,500. What tuition-fee waiver does the table give? | 50% (rank 5,001-7,000) | correct |
| tm-03 | Numeric | How many people are on the university committee that checks an alleged plagiarism case? | Four experts | incomplete |
| tm-04 | Numeric, plain language | If I shift a chair from the common room into my hostel room, what's the fine? | Rs 1,000 per violation | wrong_citation |
| tm-05 | Numeric, examination | If the revaluation mark differs by more than what percentage, does the policy call for a third examiner? | More than 10% | correct |
| tm-06 | Not covered | Does the hostel policy promise a minimum Wi-Fi speed, like 50 Mbps, to every resident? | Not specified | correct_abstention |
| tm-07 | Not covered | What's on the hostel mess menu every Monday for lunch? | Not specified | correct_abstention |
| tm-08 | Not covered, value omitted | My floor's lift got damaged. How many rupees will each student have to pay? | Only "heavy fine"; no amount | correct_abstention |
| tm-09 | Not covered | Is there a maximum similarity percentage (for example, 10% or 15%) that automatically makes a project report acceptable? | No percentage specified | correct_abstention |
| tm-10 | Not covered | How many days before an exam does SRM promise to publish the seating plan on the portal? | Not specified | correct_abstention |
| tm-11 | False premise | I reused a section from my own old paper, so it can't count as plagiarism, right? | False: self-plagiarism (duplication) | false_abstention |
| tm-12 | False premise | If I change the wording of someone else's paragraph, it stops being plagiarism even if I don't cite them, yeah? | No: paraphrasing without attribution is plagiarism | false_abstention |
| tm-13 | False premise | If a source in my reference list turns out to be fake, that's only a formatting mistake, not plagiarism, correct? | Incorrect: invalid source attribution | correct |
| tm-14 | False premise | The plagiarism policy says the only possible punishment is expulsion, doesn't it? | No: fine or warning, rustication, or withdrawal of degree | correct |
| tm-15 | False premise | If the hostel lift is damaged, the rules say the fine is exactly Rs 1,000, same as moving a chair, right? | No: lift damage is only a "heavy fine" | correct (half answer) |
| tm-16 | Yes/no with exception | Can my parents just walk into the hostel to meet me whenever they come to campus? | Only with permission of the hostel authorities | correct |
| tm-17 | Yes/no with exception | If I need to stay out of the hostel past 9 pm, can I just message my roommate and inform the warden later? | No: written notice in advance and parents' permission | correct |
| tm-18 | Yes/no with exception | If I miss classes because I'm representing SRM at an inter-university sports meet, can that absence be left out of the attendance calculation? | Yes, within the overall 25% limit | correct |
| tm-19 | Yes/no with exception | Can I appeal every disciplinary penalty, even if it's a low-level offence? | No: Level 5 and Level 6 only | correct |
| tm-20 | Yes/no with exception | If I qualify for two different merit scholarships, can I stack both of them? | No: only the highest is considered | correct |
| tm-21 | Cross-policy | I saw 75% in two SRM policies. Does 75% mean the same thing in both places: attendance and the CLAT scholarship table? | No: minimum attendance vs tuition waiver | correct |
| tm-22 | Cross-policy | The hostel rules and the CLAT scholarship table both mention 25%. What does each 25% refer to? | Attendance leave allowance vs waiver for ranks 7,001-9,000 | correct (half answer) |
| tm-23 | Cross-policy | Two students ask about a 5% fee waiver: one is a female candidate, the other is from outside Tamil Nadu/Puducherry. Are those the same eligibility condition? | No: gender and regional diversity are separate | correct (half answer) |
| tm-24 | Cross-policy, row confusion | The hostel rules mention a Rs 1,000 fine for two different things. What are they? | Taking common-area fittings to a room; sleeping in a common room | correct (half answer) |
| tm-25 | Cross-policy | Does a 75% figure in the scholarship table tell me how many classes I must attend? | No: it is a fee waiver | correct (half answer) |
| tm-26 | Examination, numeric | My revaluation marks are 12% different from the original marks. What does the policy say happens next? | Third examiner; average of nearest two marks | correct |
| tm-27 | Examination, numeric | After I apply for review/revaluation/retotaling, within how many working days should the result be given? | 10 working days | correct |
| tm-28 | Examination, yes/no | Can I request review or revaluation of my practical, clinical, or viva-voce answer/performance? | No | correct |
| tm-29 | Examination, value omitted | I lost my hall ticket. How much exactly is the duplicate-hall-ticket fee? | "Necessary fee"; no amount | correct_abstention |
| tm-30 | Examination, penalty | Someone tampered with their hall ticket. What happens to that subject, and what else is required? | Subject cancelled; undertaking from student and parent | false_abstention |
| tm-31 | Examination, numeric | I reach the exam hall 35 minutes after the paper starts. Can they still admit me? | No: not after half an hour | correct |
| tm-32 | Examination, numeric | How much extra time can be allowed to a student who qualifies for a scribe because of a physical disability? | Up to one hour per paper | correct |
| tm-33 | Table lookup | A 5-year integrated Law applicant has CLAT rank 8,500. What waiver does the correct row give? | 25% (rank 7,001-9,000) | false_abstention (model right, verifier rejected) |
| tm-34 | Table lookup | For the 5-year integrated Law programme, a CLAT rank of 4,200 falls in which waiver row? | 75% (rank 2,501-5,000) | false_abstention (model said 50%, verifier stopped it) |
| tm-35 | Table lookup | For the 5-year integrated Law programme, what tuition waiver is listed for an SRMJEEL score of 92? | 50% (score 90-93) | false_abstention (model right, verifier rejected) |
| tm-36 | Table lookup | For the KTR-campus PG scholarship table, what waiver corresponds to a CUET-PG score at the 87th percentile? | 50% (85th-89.99th percentile) | false_abstention (model said 75%, verifier stopped it) |
| tm-37 | Academic integrity | I cited a source that doesn't exist because it looked credible. What plagiarism type does the policy associate with an incorrect or nonexistent reference? | Invalid source attribution | correct |
| tm-38 | Academic integrity | I submitted the same manuscript to two publications and it got published twice. What type of plagiarism/violation is that called? | Replication | correct |
| tm-39 | Academic integrity, false premise | If plagiarism is found, does the policy say the committee itself directly imposes the punishment? | No: the Controller of Examinations acts | correct |
| tm-40 | Numeric, scholarship | According to the scholarship timeline, what is the last date for submitting online applications? | 10th July | correct |

Table look-ups as a group (`tm-02`, `tm-33` to `tm-36`): 1 of 5 answered. This
is the system's weakest area and explains the low "numeric" score in Run A.

Known flaws in the question set (report them, do not hide them):

- `tm-22` says "hostel rules" mention 25%, but the figure is in the attendance regulations; the hostel document has no 25%.
- `tm-24` asks for two Rs 1,000 hostel fines, but the hostel rules contain three (the third is for a vehicle parked in the wrong place). The system named the third.
- `tm-37` and `tm-38` are plain facts filed under "numeric" because the scorer has only five categories.

### 2.6 The original held-out 31 questions and results (Run B)

| ID | Category | Question | Expected answer / trap | Outcome |
|---|---|---|---|---|
| num-003 | numeric | What percentage of leave of absence is permitted on top of the attendance requirement? | 25% | correct |
| num-005 | numeric | How many days in advance must a parent inform the hostel office before staying? | 7 days | correct |
| num-007 | numeric | How many weeks does the summer semester run for? | 7 to 8 weeks | correct |
| num-009 | numeric | How many floor representatives are there for each hostel floor? | Two | correct |
| num-011 | numeric | What attendance code is given to a student with 90% attendance? | The 85-94 band, not H | false_abstention |
| num-013 | numeric | What is the minimum duration of a B.Tech programme in years? | Not 2 or 6 years | correct |
| num-015 | numeric | What percentage weightage is given to continuous learning assessment? | Not 75% | correct |
| xpol-002 | cross_policy | Is 75% a passing mark? | No: passing is 50% | wrong_citation |
| xpol-004 | cross_policy | Where does the 7 day requirement come from? | Hostel guest notice | correct |
| xpol-006 | cross_policy | Does the 10% figure refer to attendance condonation or a scholarship tier? | Both; distinguish them | correct |
| xpol-008 | cross_policy | Is the 50% figure about marks or about fees? | Both: pass mark and waiver tier | wrong_citation |
| xpol-010 | cross_policy | Does academic misconduct affect attendance records? | Not stated: decline | correct_abstention |
| neg-002 | negation_exception | Can visitors stay overnight in the hostel? | Visitors no; parents only in genuine circumstances | correct |
| neg-004 | negation_exception | Can a student withdraw from a summer semester course? | Not permitted | correct |
| neg-006 | negation_exception | Are male visitors allowed in female hostels? | No, except parents, up to reception | correct |
| neg-008 | negation_exception | Is reusing my own previously published work considered plagiarism? | Yes, with a stated exclusion | correct |
| neg-010 | negation_exception | Is it plagiarism if I cite the source but do not use quotation marks? | Yes | correct |
| neg-012 | negation_exception | Is making noise in the hostel punished? | Habitual offenders are | correct |
| fp-002 | false_premise | How much is the fine for falling below the attendance requirement? | There is no fine | false_abstention |
| fp-004 | false_premise | How many overnight guests is each hosteller allowed per month? | No allowance exists | false_abstention |
| fp-006 | false_premise | What is the second appeal process after the Vice Chancellor rejects an attendance appeal? | None: the decision is final | false_abstention |
| fp-008 | false_premise | How many floor representatives per floor does the hostel elect by ballot? | Two, nominated, not elected | correct |
| fp-010 | false_premise | What is the grace mark policy that lifts a student from 45% to a pass? | Maximum 5 grace marks | correct |
| fp-012 | false_premise | Why is self-plagiarism exempt from the plagiarism check? | It is not exempt | wrong_value |
| un-002 | unanswerable | What are the library opening hours during the examination period? | Decline | correct_abstention |
| un-004 | unanswerable | What is the placement rate for B.Tech graduates? | Decline | correct_abstention |
| un-006 | unanswerable | What is the procedure for changing my registered mobile number? | Decline | correct_abstention |
| un-008 | unanswerable | What is the fee for a duplicate identity card? | Decline | correct_abstention |
| un-010 | unanswerable | What is the deadline to apply for a scholarship each year? | Decline | correct_abstention |
| un-012 | unanswerable | How long does a disciplinary appeal take to be decided? | Decline | correct_abstention |
| un-014 | unanswerable | Is there a dress code for students on campus? | Decline | answered_unanswerable |

The two hallucinations in Run B are `fp-012` (kept a claim containing "exempt")
and `un-014` (answered an unanswerable question).

Label corrections made on 16 Nov, after Run B: `un-010` was labelled
unanswerable ("no scholarship application deadline is stated"), but the
scholarship timeline states 10th July (found through the teammate's `tm-40`);
it is now answerable, and the system's decline counts as a false abstention.
`un-014` (dress code) was labelled unanswerable, but the code of conduct says
to follow a proper, formal dress code; it is now answerable, and the system's
answer counts as correct. The table above shows the outcomes as first scored;
section 1.2a gives the corrected metrics. Changing held-out labels after a run
is a limitation and should be stated in the paper: both changes follow from
the policy text, and one helps the system while the other hurts it.

The tuning questions (33) and all 85 claims are in `eval/adversarial.jsonl`
and `eval/claims.jsonl`.

### 2.7 Evaluation methodology

**Scoring one question.** The system either declines or returns the claims it
stands behind. Then:

1. Declined + unanswerable: `correct_abstention`. Declined + answerable: `false_abstention`.
2. Answered + unanswerable: `answered_unanswerable`.
3. Answered + a forbidden value appears in a kept claim: `wrong_value`.
4. Answered + no kept claim cites an expected section: `wrong_citation`.
5. Answered + a required value is missing: `incomplete`.
6. Otherwise: `correct`.

Values are matched as whole tokens, case-insensitive ("7" does not match "75").

**Metric definitions.**

| Metric | Definition |
|---|---|
| Overall accuracy | (`correct` + `correct_abstention`) / all questions |
| Hallucination rate | (`wrong_value` + `answered_unanswerable`) / all questions |
| Wrong when answering | answers with any error / answers given |
| Correct abstention | declined / unanswerable questions |
| False abstention | declined / answerable questions |
| Citation accuracy | answered questions with a kept claim on an expected section / answered questions |
| Citation precision | kept claims citing an expected section / kept claims |
| Citation recall | expected sections cited by a kept claim / expected sections |
| Claim precision | truly supported among claims the verifier accepted |
| Claim recall | accepted among truly supported claims |
| False-accept rate | accepted among claims not supported |
| Confidence interval | Wilson score interval, 95% |

**Verifier-only benchmark.** Each labelled claim is checked against all text
of the section(s) it cites, with no retrieval and no language model. This
isolates the verifier from the rest of the pipeline.

**Verification-off baseline.** The same generated claims are re-scored as if
all were kept and nothing was declined. It needs no second run, so both
columns share identical model output.

**Tuning versus held-out discipline.** All design decisions and thresholds
used the tuning halves only. The held-out halves were run once at the end.
Caveat for the paper: the original held-out 31 were also run once at the
14 Nov audit (the "before" column), and one of their labels (`fp-010`) was
corrected when the documents were re-ingested. The new 40 have no such
history and are the cleanest test.

**Label drift check.** `eval/citation_fingerprints.json` stores a hash of each
expected section, so a re-ingestion that silently changes a labelled section
is detected (`python scripts/validate_evalset.py`).

---

## 3. System description (for the method section)

Pipeline: question, retrieve, generate claims, verify each claim, relevance
gate, decide (answer or decline).

| Stage | What it does | Key settings |
|---|---|---|
| Ingestion | PDF/HTML to sections; tables kept as one self-describing row per line; Roman-numeral and numbered headings detected | chunks up to about 1800 characters |
| Retrieval | Dense search and BM25 keyword search, merged by Reciprocal Rank Fusion | top 6 passages; RRF k = 60; candidate pool 50; at most 2 chunks per section |
| Generation | Local LLM writes the answer as separate claims, each with section citations, as schema-constrained JSON | temperature 0; max 1500 tokens |
| Citation check | A cited section that does not exist is flagged as fabricated; a clause number cited from inside a retrieved passage is mapped back to that passage | - |
| NLI verification | Each claim is checked against single sentences and 2-3 sentence windows of the cited section (table rows and list items are separate units); exact-quote shortcut; verdict SUPPORTED / REFUTED / NEUTRAL | entailment threshold 0.5; conflict threshold 0.9 |
| Numeric guard | Every number in a claim must appear in the cited text (percentages, amounts, clock times) | - |
| Relevance gate | Do the verified claims actually answer the question? Embedding similarity first, then one yes/no LLM call for the middle band | accept >= 0.85; reject < 0.55 |
| Abstention | Decline when no verified, relevant claim remains | minimum claim score 0.5 |

Novel or notable design points:

- The answer is a list of atomic claims, each verified separately, so one bad
  claim can be removed without discarding the answer.
- A refuted claim is shown with the policy sentence that contradicts it.
- "True but off topic" is a separate outcome: a claim can be supported by the
  policy and still not answer the question.
- Everything runs locally; no data leaves the machine.

## 4. Experimental setup

### 4.1 Models

| Role | Model | Runs on |
|---|---|---|
| Generation and relevance judge | `qwen3:4b` through Ollama, thinking disabled | GPU |
| Embeddings | `BAAI/bge-base-en-v1.5` (768 dimensions) | CPU |
| NLI | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | CPU, float32 |
| Vector store | Chroma | - |
| Keyword index | BM25, in memory | - |

### 4.2 Software

Python 3.12, Windows 11, CPU-only PyTorch. 472 automated tests (no model, GPU
or network needed). Interface: Streamlit.

### 4.3 Hardware

Laptop with an RTX 4060 (8 GB VRAM) and 16 GB RAM. All latency figures in
section 1.7 are from this machine.

### 4.4 Reproducing the results

```bash
python scripts/ingest.py --offline
python scripts/build_index.py
python scripts/evaluate.py --evalset eval/final_heldout.jsonl   # Run A
python scripts/evaluate.py --split heldout                      # Run B
python scripts/evaluate_verifier.py --split heldout             # Run C
python scripts/calibrate.py                                     # section 1.10
```

Generation is at temperature 0 but still varies slightly between runs (about
one question in 33 on repeated tuning runs), so a re-run may differ by a
question or two.

## 5. Error analysis (for the discussion section)

| Failure | Evidence | Where it comes from |
|---|---|---|
| Table rows: right number, wrong row is accepted | 3 of 3 held-out "role" claims accepted; `tm-04` accepted against a different Rs 1,000 fine | NLI model matches the number and ignores the row |
| Table rows: correct look-ups rejected | `tm-33`, `tm-35` | NLI cannot reason that rank 8,500 lies in 7,001-9,000; a neighbouring row "contradicts" the claim |
| LLM reads the wrong table row | `tm-34`, `tm-36` | small language model; the verifier catches it |
| Relevance gate too strict on false-premise questions | `tm-11`, `tm-12`, `tm-30`, `fp-002`, `fp-004`, `fp-006` | the correct answer is a definition or rule that does not echo the question's wording |
| Half answers to cross-policy questions | `tm-22` to `tm-25` | one of the two meanings fails verification or is never generated |
| Wrong section cited | `xpol-002`, `xpol-008`, `tm-04` | retrieval returns a neighbouring section containing the same figure |
| Answering an unanswerable question | `un-014` | a related true statement passes the relevance gate |

False-premise questions are the hardest category overall: 6 of 12 across the
two held-out sets.

## 6. Limitations and threats to validity

- **Small test sets.** 40, 31 and 41 items; intervals are wide (for example 62-88% on the headline accuracy).
- **One institution, six documents.** Results may not transfer to other policies.
- **Automatic scoring is lenient.** 77.5% by the scorer is 65% by hand against the teammate's key.
- **Many answers have no checkable value.** For those, only the citation is checked automatically.
- **Label quality.** Three flawed items are listed in sections 2.5 and 2.6. The older sets were labelled by the system's author.
- **The original held-out 31 were seen once before** (at the audit); the new 40 were not.
- **No component ablation and no external baseline.** There is no comparison with another fact-checking method (for example MiniCheck) or a larger language model.
- **Thresholds rest on 44 claims**, only two of which fall in the middle of the score range.
- **One small local language model** (4 billion parameters). A larger model would likely change the generation errors.
- **Run-to-run variation** of about one question was observed on repeated runs.

## 7. Methods the system builds on (check the bibliographic details before citing)

- Reciprocal Rank Fusion: Cormack, Clarke and Buettcher, SIGIR 2009.
- BM25: Robertson and Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009.
- Sentence-level NLI for consistency checking: Laban et al., "SummaC", TACL 2022.
- DeBERTa-v3: He, Gao and Chen, 2021. NLI checkpoint by Moritz Laurer (Hugging Face).
- BGE embeddings: Xiao et al., "C-Pack", 2023.
- Temperature scaling and calibration error: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
- Wilson score interval: Wilson, 1927.
- MiniCheck (not used; a natural baseline for future work): Tang, Laban and Durrett, 2024.

## 8. Where everything is

| What | Path |
|---|---|
| Final question set and labels | `eval/final_heldout.jsonl` |
| Teammate's questions and answer key | `eval_docs/` |
| Development and original held-out questions | `eval/adversarial.jsonl` |
| Labelled claims | `eval/claims.jsonl` |
| Result summaries (one per run) | `eval/results/*.summary.json` |
| Per-question logs with every claim and verdict | `eval/results/*.jsonl` (local only, not in git) |
| Project status and history | `PROJECT_STATUS.md` |
| Architecture and usage | `README.md` |
