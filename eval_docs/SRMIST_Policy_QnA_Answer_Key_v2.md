# SRMIST Policy QnA System — Evaluation Answer Key & Grading Rubric (v2)

> **v2, 2026-11-20.** A corrected copy of `srmist_policy_qna_answer_key.md`. Every ground-truth fact was
> checked against the policy text and found correct. Only the points below were changed; each is marked
> **[CHANGED 2026-11-20]** where it occurs.
>
> 1. Q15: the smartwatch undertaking is from **both the student and the parent**.
> 2. Q20: hostel stay during suspension cites Hostel Rules **16.17**, not R 3.6 (R 3.6 is about a break of studies).
> 3. Q21: the class/distinction rule is in **R 8.8**, not R 8.10–8.11; the question is marked ambiguous, with an accepted answer.
> 4. Q22 and Q33: marked interpretive, with an accepted answer.
> 5. Q31: the unverifiable "updated Mar 5, 2026" was removed.
> 6. Q37: relabelled a **near miss**: the hostel rules mention free Wi-Fi (3.6) but no limit.
> 7. Scoring rules added for calculation questions and for ambiguous questions (below the rubric).
>
> The machine-scoring version is `eval/final_test.jsonl` (with the 50-question v3 set).


This companion answer key maps every Question ID (`Q01`–`Q40`) in `srmist_policy_qna_test_questions.md` to its verified ground-truth answer, exact source document and section reference, and expected QnA system evaluation criteria.

## Recommended Scoring Rubric

| Score | Label | Criteria | 
| ----- | ----- | ----- | 
| **2** | Pass (Complete & Grounded) | States all required facts/numbers accurately, cites the right policy/clause, catches conflicts or false premises, and abstains cleanly when out of policy. | 
| **1** | Partial Pass | Core conclusion is right, but misses a secondary condition, nuance, calculation step, or one side of a cross-document conflict. | 
| **0** | Fail | Hallucinates unstated numbers/policies, answers an unanswerable question, misses a hard constraint/false premise, or cites the wrong rule. |

**[CHANGED 2026-11-20] Rules fixed before any system is run:**

- **Calculation questions (Q25–Q28):** the system under test only states what the policy text says, so it may give the verified pieces (Rs 2,500 + Rs 5,000; 0.5 credit per practice hour; "condonation cannot exceed 10%") without the computed total. That earns **2** if every piece is right and nothing is invented; a computed total is checked by hand. Calculation results are also reported as a separate category.
- **Questions marked AMBIGUOUS or INTERPRETIVE (Q21, Q22, Q33):** the accepted answer given with each question earns **2**.
- **Blind grading:** answers from the versions being compared are shuffled and graded without knowing which system produced them.
- **Tags:** the `[Domain | Type]` tags in the questions file are for people only; the system is given the question text alone.

## Section 1: Direct Policy Lookup (Q01–Q08)

### Q01 — Plagiarism Assessment Committee Composition & Reporting

* **Ground-Truth Answer**: The committee consists of **4 experts** appointed by the University: (1) the **Dean** of the respective Institution/College/School/Department (Convener), (2) one **senior Professor** from the concerned department (other than the guide), (3) one **Professor from another department**, and (4) the **Librarian**. It submits its recommendations to the **Office of the Controller of Examinations (CoE)**.

* **Source**: `academic_integrity.pdf` — Assessment Committee section.

* **Must-Verify Key Facts**: All 4 member roles (including that the departmental senior professor must *not* be the guide); submission to the Office of the CoE.

### Q02 — Course Attendance Requirement, Buffer & Condonation Cap

* **Ground-Truth Answer**: Minimum **75%** attendance per individual course is required to sit for end-semester exams. The remaining **25%** buffer covers medical issues, accidents, on-duty, extra-curricular activities, job interviews, sports, and emergencies. In rare, genuine cases, a committee of the HoD and a Senior Faculty member may recommend condonation to the Dean CET, capped at a maximum of **10%** (i.e., down to a **65%** floor).

* **Source**: `attendance.pdf` — Regulations R 7.0–R 7.5.

* **Must-Verify Key Facts**: 75% per course; 25% covers medical/OD/emergencies; max 10% condonation via HoD + Senior Faculty recommendation to Dean CET.

### Q03 — Code of Conduct Appeal Eligibility & Process

* **Ground-Truth Answer**: Appeals are permitted for **Level 5 and Level 6 misconduct ONLY**. The student must submit a written appeal to the **Registrar**, who hears the student and may either refer the case back to the Disciplinary Committee or convene the **Appellate Committee** (comprising the Vice Chancellor, Registrar, and Pro-Vice Chancellor, headed by the VC). The Appellate Committee's decision is final and binding.

* **Source**: `code_of_conduct.html` — Section 7 (Appeal Process).

* **Must-Verify Key Facts**: Restricted strictly to Level 5 & Level 6; written appeal to Registrar; Appellate Committee (VC, Registrar, Pro-VC) headed by VC; final and binding.

### Q04 — Review, Revaluation & Retotaling Rules

* **Ground-Truth Answer**: Applications must be submitted within **7 days** of result declaration (via the student portal, `coe@srmist.edu.in`, or e-varsity). If the mark difference on revaluation exceeds **10%**, a **3rd examiner** evaluates the script and the average of the nearest two marks is final. Review, revaluation, and retotaling are **NOT permitted for practical, clinical, or viva voce** exams. (Results are declared within 10 working days of application.)

* **Source**: `examination.html` — Section XX (Grievances).

* **Must-Verify Key Facts**: 7-day window; >10% triggers 3rd examiner (average of nearest two); excluded for practical/clinical/viva voce.

### Q05 — Hostel Mess Timings & Silence Hours

* **Ground-Truth Answer**:

  * **Mess Timings**: Breakfast **7:00–9:00 AM**, Lunch **11:30 AM–1:30 PM**, Tea **4:30–5:30 PM**, Dinner **7:30–9:00 PM**.

  * **Silence Hours**: **10:00 PM to 6:00 AM**.

* **Source**: `residence.pdf` — Section 12.12 (Mess Timings) and Section 16.24 (Silence Hours).

* **Must-Verify Key Facts**: All 4 exact meal windows and 10:00 PM–6:00 AM silence hours.

### Q06 — Scholarship Portal Timeline & Excluded Programs

* **Ground-Truth Answer**:

  * **Annual Timeline**: Portal opens **1st July**, last date to apply is **10th July**, committee meeting by **20th July**, recommendations to Director (Admissions) by **25th July**, and VC-approved provisional list published by **1st August**.

  * **Excluded Programs**: Not applicable for **NEET-based programs, Pharm.D, and Pharm.D (PB)**.

* **Source**: `scholarship.html` — General Guidelines & Timeline.

* **Must-Verify Key Facts**: 1 July open, 10 July close, 1 August provisional list; NEET-based programs, Pharm.D, and Pharm.D (PB) excluded.

### Q07 — Alumni Scholarship Eligibility, Benefit & Renewal

* **Ground-Truth Answer**:

  * **Eligibility**: Parent must have completed UG **≥15 years** ago or PG **≥10 years** ago at SRMIST, be registered as alumni, and the benefit applies to the **first living child only** (not siblings).

  * **Benefit**: **25% tuition fee waiver**.

  * **Renewal / Ongoing Conditions**: Annual application, **CGPA > 7**, no arrears/backlogs, no break/detention/disciplinary action, and the parent must participate in University activities **≥2 times per year**.

* **Source**: `scholarship.html` & `scholarship_categories.pdf` — Alumni Scholarship.

* **Must-Verify Key Facts**: UG ≥15 yrs / PG ≥10 yrs, first living child only, 25% tuition waiver, CGPA > 7 renewal, parent participation ≥2 times/year.

### Q08 — Answer Script Retention & Question Paper Complaint Deadline

* **Ground-Truth Answer**: Evaluated answer scripts are stored for **1 year (2 semesters)**. Complaints against question papers must be routed through the **Head of College/Department** to the **Controller of Examinations (CoE)** within **3 days** of the examination.

* **Source**: `examination.html` — Section VIII (Complaints against Question Papers) and Section XXI (Storage of Answer Scripts).

* **Must-Verify Key Facts**: 1 year (2 semesters) storage; 3-day complaint deadline through Head of College/Dept to CoE.

## Section 2: Scenario-Based & Applied Judgment (Q09–Q18)

### Q09 — Self-Plagiarism / Duplication Exception for Published Thesis Papers

* **Ground-Truth Answer**: Under the Duplication (Self-plagiarism / Reuse) rule, she must: (1) include proper citations to her own published articles (if flagged without citation, she must resubmit with proper citations), (2) obtain a **certificate from her supervisor** specifying and attaching the articles published by her from the thesis work (only those certified articles are excluded from the plagiarism check), and (3) for the co-authored paper, submit a **written consent letter from both the co-author(s) and the publisher**.

* **Source**: `academic_integrity.pdf` — Types of Plagiarism (3. Duplication / Self-plagiarism).

* **Must-Verify Key Facts**: Proper citation, supervisor's certificate attaching the published articles, and consent letters from *both* co-author(s) and publisher.

### Q10 — Lateral-Entry Diploma Exit & MOOC Substitution

* **Ground-Truth Answer**: **Neither** part of the plan is permitted:

  1. **Diploma Exit**: Lateral-entry students are **explicitly prohibited** from exiting with a Diploma (R 1.5). (Even for regular students, Diploma exit is only for exceptional circumstances and requires informing the HoD in writing before 5th-semester exams, R 5.3.)

  2. **MOOC for 6th-Sem Project**: Even if a student were eligible for Diploma exit, students opting for Diploma exit **must compulsorily do the 6th-semester Project Work** and cannot substitute it with a 3-credit MOOC (R 2.6).

* **Source**: `attendance.pdf` — R 1.5, R 2.6, and R 5.3.

* **Must-Verify Key Facts**: Identifies *both* disqualifiers (lateral-entry ban on Diploma exit AND mandatory Project Work for Diploma-exit students).

### Q11 — Guided Study CGPA Cutoff & Week-3 Core Course Drop

* **Ground-Truth Answer**: **Neither** request is permitted:

  1. **Guided Study / Self-Learning**: Requires a minimum **CGPA ≥ 9.5** (from Semester VI onwards, max 2 courses for the entire program). At 9.2 CGPA, the student is ineligible (R 3.8).

  2. **Dropping a Core Course in Week 3**: Fails on **two counts** — Add/Drop applies **only to Professional/Open Electives** (not core courses), and must be done within the **first 2 weeks** of the semester. Unilaterally dropping a course after 2 weeks results in an **`Ab` or `I`** grade (R 3.7).

* **Source**: `attendance.pdf` — R 3.7 and R 3.8.

* **Must-Verify Key Facts**: 9.5 CGPA floor for Guided Study; Add/Drop is electives-only + 2-week deadline + `Ab`/`I` penalty after 2 weeks.

### Q12 — Appeal Against a Level 2 Suspension

* **Ground-Truth Answer**: **No**, the appeal will **not** be entertained. Under Section 7 of the Code of Conduct, appeals to the Registrar / Appellate Committee are permitted **only for Level 5 and Level 6 misconduct**. Tampering/misuse of an ID card is a **Level 2** offense (handled by DSA with academic dept / Director Campus Life).

* **Source**: `code_of_conduct.html` — Section 4 (Level 2), Section 5 (Process), and Section 7 (Appeal Process).

* **Must-Verify Key Facts**: Explicitly states appeal is not permitted for Level 2 (only Level 5 and Level 6).

### Q13 — Amanuensis (Scribe) Eligibility, Practical Ban & Payment

* **Ground-Truth Answer**:

  * **Theory vs. Practical**: Permitted for the **Theory** paper (temporary disability such as arm fracture qualifies), but **strictly NOT permitted for Practical courses**.

  * **Proposed Scribe (Younger Brother)**: **Rejected** — the scribe **cannot be a blood relative** (even though his 52% score meets the ≤55% lesser-qualified threshold). The scribe also cannot be from the same faculty/degree/discipline.

  * **Payment & Extra Time**: The **University does not pay the writer**; moreover, as a non-blind candidate, the student must pay an **additional prescribed fee per paper**. Approved candidates receive **up to 1 hour of extra time**.

* **Source**: `examination.html` — Section XIII (Amanuensis / Scribes).

* **Must-Verify Key Facts**: Theory allowed / Practical prohibited; blood relative barred; student pays prescribed fee + University does not pay scribe; up to 1 hour extra time.

### Q14 — Late Arrival (35 Mins) & Forgotten vs. Lost Hall Ticket

* **Ground-Truth Answer**:

  * **Late Entry**: **No**, the student **cannot enter** the exam hall—entry is strictly barred after **30 minutes** from the commencement of the exam (students are expected 30 minutes early, and no extra time is granted for latecomers).

  * **Forgotten vs. Lost Hall Ticket**: If **forgotten**, a duplicate hall ticket is issued by the **Chief Superintendent (CS)** for **1 session only** upon payment of a fine. If **lost**, the student must approach the **CoE office** and pay the prescribed fee.

* **Source**: `examination.html` — Sections VI, IX.13, and XII.

* **Must-Verify Key Facts**: 30-minute hard cutoff denies entry at 35 minutes; CS issues 1-session duplicate for forgotten ticket vs. CoE office for lost ticket.

### Q15 — Malpractice in J-Code Practical vs. Smartwatch Possession

* **Ground-Truth Answer**:

  * **Student A (Malpractice in practical of a J-code course)**: **Both the theory and practical components** of that J-code course are cancelled.

  * **Student B (Smartwatch in theory exam, first offense)**: **That particular subject is cancelled** plus a written **undertaking from both the student and the parent** (a repeat offense results in termination). **[CHANGED 2026-11-20: "both the student and the parent" — Annexure II.]**

* **Source**: `examination.html` — Annexure II (Malpractice Punishments, Items 2 and 10).

* **Must-Verify Key Facts**: J-code practical malpractice cancels *both* theory and practical; smartwatch cancels that specific subject + undertaking (repeat = termination).

### Q16 — Unauthorized Day Scholar & Unreported Missing Roommate

* **Ground-Truth Answer**:

  1. **Day Scholar in Room & Mess**: Day scholars are not permitted to stay as guests in hostel rooms or eat in the hostel mess. Bringing an unauthorized day scholar into the hostel room or mess carries a fine of **Rs. 2,000**.

  2. **Unreported Missing Roommate**: Staying out past 9:00 PM requires written advance notice to the Warden/Deputy Warden plus mandatory parent permission. Under Section 16.33, **roommates are required to report anyone missing after 10:00 PM**; failing to do so makes the resident **equally liable for disciplinary action**.

* **Source**: `residence.pdf` — Sections 9, 12.10, 16.25, and 16.33.

* **Must-Verify Key Facts**: Rs. 2,000 fine for unauthorized day scholar in room/mess; 10:00 PM reporting duty for roommates with equal disciplinary liability.

### Q17 — Mid-Year Hostel Fee Refunds (2nd Year vs. Final-Year Internship)

* **Ground-Truth Answer**:

  * **2nd-Year Student (Moving Off-Campus)**: **No refund**. For non-first-year students, no refund is granted if withdrawing from the hostel during the academic year (or if expelled).

  * **Final-Year Student (University-Approved Internship)**: Eligible for a refund of **mess fees only** (no room rent refund), granted on a **case-by-case basis** with approval from the **Project Guide and HoD**.

* **Source**: `residence.pdf` — Section 7.0 (Refund of Hostel Fees).

* **Must-Verify Key Facts**: Zero refund for 2nd-year withdrawal; mess fees *only* (case-by-case with Project Guide + HoD approval) for final-year internship.

### Q18 — Paramilitary Exclusion & Scholarship Stacking

* **Ground-Truth Answer**: Eligible for **neither** scholarship:

  1. **Defence Scholarship**: **Not eligible** — **paramilitary personnel are explicitly excluded** (only serving/retired/deceased defence personnel via SRMJEEE counselling qualify).

  2. **Socio-Economic Need-Based Scholarship**: **Not eligible as a 1st-year student** — even though parental income (Rs. 4.5 Lakhs) is below the Rs. 6 Lakhs cap, this scholarship is restricted to **II year or above (seniors only)**.

  3. **Stacking Rule**: Even if eligible for both, a candidate can receive **only ONE category** of scholarship (the only stacking exception is Diversity scholarships in Management/Law/S&H combining with one highest Merit scholarship).

* **Source**: `scholarship.html` & `scholarship_categories.pdf` — General Rules, Defence Scholarship, and Socio-Economic Need-Based Scholarship.

* **Must-Verify Key Facts**: Catches all 3 rules (paramilitary exclusion, II-year+ seniors-only restriction, and one-category-only rule).

## Section 3: Multi-Hop & Cross-Document Reasoning (Q19–Q24)

### Q19 — 80% Attendance: Exam Eligibility vs. Hostel Continuation

* **Ground-Truth Answer**:

  * **End-Semester Exams**: **Yes** — 80% exceeds the **75%** minimum attendance requirement per course (`attendance.pdf`, R 7.0/R 7.3; attendance code `8`).

  * **Hostel Continuation**: **No** — under Hostel Rules (`residence.pdf`, Section 16.17), students with **less than 95% attendance** in a semester are **not permitted to stay in the hostel for subsequent semesters**.

* **Source**: `attendance.pdf` (R 7.0, R 7.3) + `residence.pdf` (Section 16.17).

* **Must-Verify Key Facts**: System must synthesize both thresholds (75% academic vs. 95% hostel continuation) rather than giving a single "yes".

### Q20 — Level 3 Property Damage: Cross-Policy Consequences

* **Ground-Truth Answer**:

  * **Under Code of Conduct (`code_of_conduct.html`, Sec 4, Level 3)**: Parents must **come in person and sign an undertaking**; student receives **7 days academic suspension + assigned tasks**; **cost recovery** for property damage; **disqualification from placement assistance** and/or **barred from end-semester examinations**. (Also documented in Academia/ERP and mailed to HoD.)

  * **Under Hostel Rules (`residence.pdf`, Section 16.17)**: The student is **NOT permitted to stay in the hostel during academic suspension**. **[CHANGED 2026-11-20: source is Hostel Rules 16.17; R 3.6 concerns a break of studies, not a suspension.]**

* **Source**: `code_of_conduct.html` (Sections 4–5) + `residence.pdf` (Section 16.17).

* **Must-Verify Key Facts**: Parents in person + undertaking, placement disqualification / exam bar, and prohibition from staying in hostel during suspension.

### Q21 — Excused Bereavement Absence: Ranks/Medals vs. First Class with Distinction

* **Ground-Truth Answer**:

  * **University Ranks & Medals (`examination.html`, Sec IX.12 & Sec B.II)**: **Still eligible** (assuming First Class/Distinction and top 1–3 standing). Demise of father/mother/spouse/children is one of the **4 excused grounds** that preserve "First Appearance" status for rank/medal eligibility.

  * **First Class with Distinction (`attendance.pdf`, R 8.8 — [CHANGED 2026-11-20: was R 8.10–8.11])**: **Strict tension / likely downgraded to First Class** — `attendance.pdf` states First Class with Distinction requires CGPA ≥ 8.0 within normal duration (8 semesters) with **NO `Ab`, `F`, `I`, or temporary withdrawal in any semester**; if CGPA ≥ 8.0 with a prior `Ab`/`F`/`I` cleared later, the degree is awarded as **First Class**. (A strong QnA answer highlights that `examination.html` explicitly preserves First Appearance for ranks/medals, whereas `attendance.pdf` has a strict no-`Ab` rule for Distinction without explicitly listing the bereavement carve-out.)

* **Source**: `examination.html` (Sec IX.12, Sec B.II) + `attendance.pdf` (R 8.8).

* **[CHANGED 2026-11-20] AMBIGUOUS — accepted answer:** the policies do not say whether a bereavement absence leaves an `Ab` on the record, and R 8.8's B.Tech table is for students exiting the Integrated M.Tech early. An answer that states the ranks/medals rule (bereavement = first appearance), the R 8.8 distinction rule (no `Ab`/`F`/`I`), and that the documents do not settle how they combine, earns 2.

* **Must-Verify Key Facts**: Recognizes bereavement as an excused first-appearance ground for Ranks/Medals in `examination.html` AND contrasts it with the strict no-`Ab`/`F`/`I` rule for Distinction in `attendance.pdf`.

### Q22 — 70% Attendance `I` Grade Impact on Exam & Merit Scholarship Renewal

* **Ground-Truth Answer**:

  * **Course Registration (`attendance.pdf`, R 7.0)**: Because 70% is below 75% (and not condoned), the student is **ineligible for the end-semester exam**, their **course registration is cancelled**, they are awarded an **`I` (Incomplete) grade**, and parents are informed.

  * **SRM Merit Scholarship Renewal (`scholarship.html`)**: **Cannot renew**. Although their 9.1 CGPA exceeds the **CGPA > 9** renewal floor for SRM Merit Scholarship, renewal strictly requires **no arrears/backlogs** and **no detention** (along with annual application).

* **Source**: `attendance.pdf` (R 7.3–7.4) + `scholarship.html` (Renewal Criteria).

* **[CHANGED 2026-11-20] INTERPRETIVE — accepted answer:** the text requires "no arrears/backlogs at the time of application"; whether an uncleared `I` grade counts as a backlog is an interpretation. Stating the exam/registration consequences and the no-arrears renewal condition earns 2.

* **Must-Verify Key Facts**: Exam ineligibility + cancelled registration + `I` grade; scholarship renewal denied due to backlog/detention despite CGPA > 9.

### Q23 — Unauthorized Hot Plate & 6-Week Unpaid Hostel Fine

* **Ground-Truth Answer**:

  * **Under Hostel Rules (`residence.pdf`, Sec 10.2–10.3, 11.1, 16.1, 16.32)**: Hot plates are **strictly prohibited** (heavy penalty, and **expulsion from hostel** on repeat). Furthermore, hostel fines must be paid within **1 month**; because 6 weeks have elapsed without payment, the student must **vacate the hostel immediately** (Sec 11.1).

  * **Under Code of Conduct (`code_of_conduct.html`, Sec 4, Level 1)**: Possession of unauthorized electrical appliances in a hostel room is classified as **Level 1 misconduct**, carrying **intimation to parents (3 times) and/or 3 days academic suspension**.

* **Source**: `residence.pdf` (Sec 10, 11.1, 16) + `code_of_conduct.html` (Sec 4, Level 1).

* **Must-Verify Key Facts**: Immediate hostel vacation for fine unpaid >1 month; hot plate strictly prohibited; Code of Conduct classifies it as Level 1 (parent intimation 3x and/or 3-day suspension).

### Q24 — Plagiarism Rustication Authority, Ratification & Scholarship Impact

* **Ground-Truth Answer**:

  * **Disciplinary Authority & Ratification (`academic_integrity.pdf`)**: Following the Plagiarism Assessment Committee's recommendation to the CoE, the **Controller of Examinations (CoE)** takes action **in consultation with the Vice Chancellor** (rustication for a limited period or permanent is an authorized punishment). The action is ratified in meetings of the **Academic Council and Management Council**.

  * **Scholarship Impact (`scholarship.html`)**: Scholarships are **revocable for misbehavior, indiscipline, or malpractice** (and renewal is barred if there is any disciplinary action or break of study).

* **Source**: `academic_integrity.pdf` (Punishments) + `scholarship.html` (General Rules / Renewal).

* **Must-Verify Key Facts**: CoE in consultation with VC; ratified by Academic Council & Management Council; scholarship revocable for malpractice/disciplinary action.

## Section 4: Numerical, Calculation & Boundary Cases (Q25–Q30)

### Q25 — Course Credit Calculation & CGPA-to-Percentage Formula

* **Ground-Truth Answer**:

  * **Credits**: `3(L) × 1 + 1(T) × 1 + 2(P) × 0.5 = 3 + 1 + 1 =` **5 credits**.

  * **Equivalent Percentage**: `CGPA × 10 = 8.35 × 10 =` **83.5%**.

* **Source**: `attendance.pdf` — R 2.3 (Credit Calculation) and R 8.7 (Equivalent Percentage).

* **Must-Verify Key Facts**: 5 credits (using 0.5 credit per Practice hour); 83.5% (`CGPA * 10`).

### Q26 — 63% Attendance Condonation Boundary

* **Ground-Truth Answer**: **No**, the shortfall cannot be condoned. Minimum attendance is **75%**, and the maximum permissible condonation is **10%**, which establishes a hard floor of **65%** (`75% - 10% = 65%`). At **63%**, the student is **2 percentage points below** the condonable floor (requiring 12% condonation, which exceeds the 10% cap).

* **Source**: `attendance.pdf` — R 7.3 and R 7.5.

* **Must-Verify Key Facts**: Explicitly states "No", calculates 65% floor (`75% - 10%`), notes 63% is below 65%.

### Q27 — Revaluation 10% Trigger & Third-Examiner Final Mark

* **Ground-Truth Answer**:

  * **Triggers 3rd Examiner?**: **Yes**. The difference between Examiner 1 (42) and Examiner 2 (55) is **13 marks**, which is **13%** of the 100-mark paper — exceeding the **>10%** threshold.

  * **Final Mark**: The nearest two marks among `{42, 55, 51}` are **55 and 51** (difference of 4, vs. 9 between 42 and 51). Their average is `(55 + 51) / 2 =` **53 marks**.

* **Source**: `examination.html` — Section XX (Revaluation).

* **Must-Verify Key Facts**: 13% > 10% triggers 3rd examiner; nearest two are 51 and 55; final mark is 53.

### Q28 — Hostel Appliance Charges & Duplicate Key Fine

* **Ground-Truth Answer**:

  * **80W Fan** (up to 100W): **Rs. 2,500/year**

  * **150W Air Cooler** (101–200W tier): **Rs. 5,000/year**

  * **Total Annual Appliance Charges**: **Rs. 7,500/year**

  * **Vacation Duplicate Key Fine**: **Rs. 1,000** (plus the warden is authorized to remove the lock).

  * **Combined Total**: **Rs. 8,500**.

* **Source**: `residence.pdf` — Section 10 (Permitted Appliances) and Section 24.0 (Vacation / Vacant Bed Key Rule).

* **Must-Verify Key Facts**: Rs. 2,500 (fan) + Rs. 5,000 (101–200W cooler) = Rs. 7,500; Rs. 1,000 fine + lock removal; Rs. 8,500 combined.

### Q29 — MBA (KTR) Merit + Diversity Stacking Calculation

* **Ground-Truth Answer**: **35% total tuition fee waiver**, broken down as:

  * NMAT score of 280 (Merit): **25%**

  * Gender Diversity (Female): **5%**

  * Regional Diversity (Aadhaar outside Tamil Nadu / Puducherry): **5%**

  * Total: `25% + 5% + 5% = 35%` (Diversity scholarships in Management are explicitly allowed to stack with one highest Merit scholarship).

* **Source**: `scholarship.html` — Management / Faculty-Specific Stacking Examples.

* **Must-Verify Key Facts**: 35% total; exact 25% + 5% + 5% breakdown.

### Q30 — Employee Ward Scholarship (Non-Teaching, 11 Years Service)

* **Ground-Truth Answer**:

  * **1st Child (1st Time)**: **75% tuition waiver** (Non-teaching 1st-time tier for `>10 to <12 years` is 75%).

  * **2nd Child (2nd Time)**: **Max 50% tuition waiver** (Non-teaching 2nd-time eligibility requires `≥8 years` service, capped at 50%).

  * **3rd Child (3rd Time)**: **No** — the Employee Ward Scholarship is **not available beyond the 2nd time**.

* **Source**: `scholarship.html` & `scholarship_categories.pdf` — Employee Ward Scholarship (EWS).

* **Must-Verify Key Facts**: 75% (1st time, >10–<12 yrs non-teaching); max 50% (2nd time); not available beyond 2nd time.

## Section 5: Policy Conflict & Version Discrepancies (Q31–Q34)

### Q31 — IIT JEE (Mains) Cutoff Conflict for Founder's Scholarship

* **Ground-Truth Answer**: There is a **direct version conflict** between the two scholarship documents:

  * `scholarship.html` (2023 Onwards) specifies **Top 20,000 ranks** in IIT JEE (Mains). **[CHANGED 2026-11-20: unverifiable "updated Mar 5, 2026" removed.]**

  * `scholarship_categories.pdf` (2025-26 Onwards Amended) specifies **Rank up to 2,000** in IIT JEE (Mains).

* **Source**: `scholarship.html` vs. `scholarship_categories.pdf` — Founder's Scholarship.

* **Expected System Behavior**: **Must surface both figures and flag the discrepancy** between the 2023-onwards web policy (Top 20,000) and the 2025–26 amended PDF (up to 2,000), advising the user to confirm which applies to their admission cohort. Giving only one figure without flagging the conflict is a partial/failed response.

### Q32 — Internal Conflict on Tobacco/Gutka & Weapons/Pornography Level

* **Ground-Truth Answer**: There is an **internal conflict within `code_of_conduct.html`**:

  * **Section 3 (Don'ts)** tags tobacco/gutka and weapons/sharp objects/pornography as **(Level 4)**.

  * **Section 4 (Punishments table)** explicitly places possession/consumption/distribution of tobacco/gutka and possession of weapons/sharp objects/pornography under **LEVEL – 5**, which carries **Rustication for 1 year OR Expulsion from the University**.

* **Source**: `code_of_conduct.html` — Section 3 vs. Section 4.

* **Expected System Behavior**: **Must flag the internal mismatch** (Level 4 in Section 3 vs. Level 5 in Section 4) and state the Level 5 punishment (1-year rustication or expulsion) from the operative punishments table.

### Q33 — Appeal Scope Conflict (Level 3 Sanction)

* **Ground-Truth Answer**: There is a **scope conflict / tension** across two documents:

  * **`code_of_conduct.html` (Section 7)** explicitly restricts appeals (to the Registrar / Appellate Committee) to **Level 5 and Level 6 misconduct ONLY** — meaning a Level 3 sanction **cannot** be appealed under the Code of Conduct.

  * **`attendance.pdf` (R 6.4)** contains a general clause stating that an aggrieved student may appeal to the **Vice Chancellor**, whose decision is final and binding.

* **Source**: `code_of_conduct.html` (Section 7) vs. `attendance.pdf` (R 6.4).

* **Expected System Behavior**: **Must lead with the specific Code of Conduct rule** (Level 3 is not appealable; only Levels 5–6 are), and ideally note the broader general appeal clause in `attendance.pdf` R 6.4 as a secondary cross-document tension.

* **[CHANGED 2026-11-20] INTERPRETIVE — accepted answer:** R 6.4 belongs to the academic regulations' own discipline process, so calling it a conflict is an interpretation. Leading with the Code of Conduct rule earns 2; mentioning R 6.4 is a plus, not required.

### Q34 — Employee Resignation Impact on Employee Ward Scholarship

* **Ground-Truth Answer**: There is a **policy nuance / version difference** between the two scholarship files:

  * **`scholarship.html`** states that the Employee Ward Scholarship remains applicable even after **retirement or demise** of the employee (it does not mention voluntary resignation).

  * **`scholarship_categories.pdf` (2025-26 Onwards Amended)** explicitly adds a caveat: **"(it may be withdrawal if they leave the organization)"** — meaning resignation/leaving SRMIST can lead to withdrawal of the scholarship.

* **Source**: `scholarship.html` vs. `scholarship_categories.pdf` — Employee Ward Scholarship (EWS).

* **Expected System Behavior**: **Must cite the withdrawal caveat in `scholarship_categories.pdf`** and distinguish leaving/resigning from retirement or demise.

## Section 6: Unanswerable, Out-of-Policy & False Premise (Q35–Q40)

### Q35 — Central Library Borrowing Limits & Overdue Fines

* **Ground-Truth Answer**: **Unanswerable / Out of Policy.** None of the 7 provided SRMIST policy documents specify Central Library book borrowing limits or overdue fine rates. (The Librarian is only mentioned as a member of the Plagiarism Assessment Committee in `academic_integrity.pdf`.)

* **Expected System Behavior**: **Must cleanly abstain** and state that library borrowing rules and fines are not covered in the provided policy documents. **Fail** if the system fabricates book limits or Rs./day fines.

### Q36 — Exact 2025–26 B.Tech CSE Tuition Fee in INR

* **Ground-Truth Answer**: **Unanswerable / Out of Policy.** The uploaded scholarship and academic documents discuss tuition fee waiver percentages (e.g., 25%–100%), but none of the 7 documents state the actual annual tuition fee amount in INR for B.Tech CSE at Kattankulathur.

* **Expected System Behavior**: **Must abstain** without guessing or pulling ungrounded external fee figures from parametric memory.

### Q37 — Hostel Wi-Fi Data Limit & Bandwidth Cap

* **Ground-Truth Answer**: **Unanswerable / Out of Policy — near miss.** `residence.pdf` says free Wi-Fi is available to every student for academic purposes (3.6) but does not specify any daily data limit or per-student bandwidth cap. **[CHANGED 2026-11-20: was described as not mentioned at all.]** Saying "free Wi-Fi, no limit stated" is correct; inventing a limit fails.

* **Expected System Behavior**: **Must abstain** and state that hostel Wi-Fi/bandwidth limits are not in the provided hostel rules.

### Q38 — Ph.D. Monthly Stipend & Faculty Dress Code

* **Ground-Truth Answer**: **Unanswerable / Out of Policy.** Neither Ph.D. scholar fellowship stipends nor faculty dress code policies are covered in the 7 uploaded documents (`code_of_conduct.html` covers *student* dress code under Level 1, and `academic_integrity.pdf` covers plagiarism compliance for doctoral scholars, not stipends).

* **Expected System Behavior**: **Must abstain on both parts** without conflating student dress code with faculty dress code or inventing stipend amounts.

### Q39 — False Premise: Revaluation of Viva Voce & Practical Lab Marks

* **Ground-Truth Answer**: **False Premise — Not Permitted.** Although `examination.html` (Section XX) provides a **7-day** window from result declaration to apply for review, revaluation, or retotaling of theory papers, it **explicitly prohibits review, revaluation, and retotaling for practical, clinical, and viva voce examinations**.

* **Expected System Behavior**: **Must reject the false premise** rather than blindly answering "7 days".

### Q40 — False Premise: Hostel Car Parking Fee & Single-Occupancy Room Rate

* **Ground-Truth Answer**: **Double False Premise — Both Are Prohibited (No Such Fees Exist):**

  1. **Personal Cars in Hostel Premises**: Prohibited under `residence.pdf` (Section 16.31) — keeping two-wheelers or cars in hostel premises attracts a **Rs. 1,000 fine** (there is no monthly car parking permit fee).

  2. **Single-Occupancy Hostel Room**: Prohibited under `residence.pdf` (Section 8.5) — rooms are allotted on a shared basis and **no student is permitted to stay alone** (no single-occupancy surcharge exists in the policy).

* **Expected System Behavior**: **Must correct both false premises** using Sections 8.5 and 16.31 of `residence.pdf` and refrain from inventing permit or single-room fees.