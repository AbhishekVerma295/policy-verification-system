# SRMIST Policy QnA System — Evaluation Test Questions (v2)

> **v2, 2026-11-20.** A copy of `srmist_policy_qna_test_questions.md`; the questions are unchanged. The
> `[Domain | Type]` tags are for people only: the system is given the question text alone (see
> `eval/final_test.jsonl`, where these are items `fa-01` to `fa-40`). Answer key:
> `SRMIST_Policy_QnA_Answer_Key_v2.md`.


This test suite contains **40 evaluation questions** grounded in the 7 uploaded SRM Institute of Science and Technology (SRMIST) policy documents (`academic_integrity.pdf`, `attendance.pdf`, `code_of_conduct.html`, `examination.html`, `residence.pdf`, `scholarship.html`, and `scholarship_categories.pdf`).

Answers, exact document citations, and grading rubrics are provided in the companion file: `srmist_policy_qna_answer_key.md`.

## Test Suite Overview

| Section | Category | Question IDs | Count | What It Evaluates |
| :--- | :--- | :--- | :---: | :--- |
| Section 1 | Direct Policy Lookup | Q01–Q08 | 8 | Exact fact retrieval, rule precision, and single-document grounding |
| Section 2 | Scenario-Based & Applied Judgment | Q09–Q18 | 10 | Applying policy rules and exceptions to realistic student situations |
| Section 3 | Multi-Hop & Cross-Document Reasoning | Q19–Q24 | 6 | Synthesizing rules across two or more separate policy documents |
| Section 4 | Numerical, Calculation & Boundary Cases | Q25–Q30 | 6 | Credit/percentage math, threshold boundaries, fines, and waiver stacking |
| Section 5 | Policy Conflict & Version Discrepancies | Q31–Q34 | 4 | Detecting contradictions across amended versions or within a document |
| Section 6 | Unanswerable, Out-of-Policy & False Premise | Q35–Q40 | 6 | Abstention accuracy, hallucination resistance, and false-premise correction |

---

## Section 1: Direct Policy Lookup (Q01–Q08)

**Q01** `[Domain: Academic Integrity | Type: Direct Lookup]`
Who serves on the SRMIST Plagiarism Assessment Committee, and to which office does the committee submit its recommendations?

**Q02** `[Domain: Attendance & Academics | Type: Direct Lookup]`
What is the minimum attendance percentage required per individual course to be eligible for end-semester examinations, what absences does the remaining buffer cover, and what is the maximum condonation percentage allowed in genuine cases?

**Q03** `[Domain: Code of Conduct | Type: Direct Lookup]`
Under the Code of Conduct for Students, who can file an appeal against a disciplinary decision, who hears the appeal, and which misconduct levels are eligible for appeal?

**Q04** `[Domain: Examinations | Type: Direct Lookup]`
Within how many days of result declaration must a student apply for review, revaluation, or retotaling, what mark difference triggers a third examiner during revaluation, and which examination types are excluded from this process?

**Q05** `[Domain: Hostels / Residence | Type: Direct Lookup]`
What are the official hostel mess timings for all four meals, and what are the designated silence hours inside the hostel?

**Q06** `[Domain: Scholarships | Type: Direct Lookup]`
What is the annual scholarship portal application schedule (opening date, closing date, and provisional list date), and which degree programs are completely excluded from SRMIST scholarships?

**Q07** `[Domain: Scholarships | Type: Direct Lookup]`
What are the eligibility requirements, tuition waiver percentage, and ongoing renewal conditions for the Alumni Scholarship?

**Q08** `[Domain: Examinations | Type: Direct Lookup]`
How long does the University store evaluated answer scripts, and what is the deadline and channel for submitting a complaint about a question paper?

---

## Section 2: Scenario-Based & Applied Judgment (Q09–Q18)

**Q09** `[Domain: Academic Integrity | Type: Scenario]`
A doctoral researcher at SRMIST has already published two journal articles based on her thesis work, one of which is co-authored with an external collaborator. She wants to include content from these published papers in her final thesis without being penalized for self-plagiarism (duplication). What exact steps and supporting documents are required?

**Q10** `[Domain: Attendance & Academics | Type: Scenario]`
A lateral-entry B.Tech student facing personal hardship wants to exit the program with a Diploma after the 6th semester and wants to take an approved 3-credit NPTEL MOOC instead of doing the 6th-semester Project Work. Is either part of this plan permitted under the 2021 Academic Regulations?

**Q11** `[Domain: Attendance & Academics | Type: Scenario]`
A 6th-semester B.Tech student with a CGPA of 9.2 wants to register for a Guided Study (Self-Learning) course this semester. Separately, in Week 3 of the semester, the same student decides to drop a compulsory core course they are struggling with. How do the Academic Regulations treat both requests?

**Q12** `[Domain: Code of Conduct | Type: Scenario]`
A student caught misusing and tampering with their university ID card is given a 5-day academic suspension (Level 2 misconduct). Believing the penalty is harsh, the student submits a written appeal to the Registrar requesting a review by the Appellate Committee. Should the appeal be entertained?

**Q13** `[Domain: Examinations | Type: Scenario]`
A B.Tech student fractures their dominant arm one week before exams and requests a scribe (amanuensis) for both a Theory exam and a Practical lab exam. They propose using their younger brother—who is in a lower grade and scored 52%—as the writer, and ask whether the University will pay the scribe. Which parts of this request are permitted, which are rejected, and what extra time applies?

**Q14** `[Domain: Examinations | Type: Scenario]`
A student arrives at the examination hall 35 minutes after the start of an end-semester exam due to heavy traffic, and also realizes they forgot their hall ticket back in their hostel room. Can they enter the exam hall, and what is the normal procedure for a forgotten versus a lost hall ticket?

**Q15** `[Domain: Examinations | Type: Scenario]`
In two separate incidents during end-semester exams: (a) Student A is caught committing malpractice during the practical component of a Joint (J-code) course, and (b) Student B is caught carrying a smartwatch into a theory exam hall for the first time. What exact punishments apply to Student A and Student B?

**Q16** `[Domain: Hostels / Residence | Type: Scenario]`
A 2nd-year hostel resident invites a day-scholar friend into their hostel room and takes them to eat dinner in the hostel mess. Later that same week, the resident's roommate fails to return to the hostel after 10:30 PM, and the resident chooses not to inform the warden. What rules have been violated and what penalties or liabilities apply?

**Q17** `[Domain: Hostels / Residence | Type: Scenario]`
Two students request a prorated refund of their hostel room rent and mess fees mid-academic year: (a) a 2nd-year student who decides to move into an off-campus apartment, and (b) a final-year student leaving for a University-approved semester internship. What refund, if any, is each student eligible to receive?

**Q18** `[Domain: Scholarships | Type: Scenario]`
An incoming 1st-year B.Tech student whose father is a serving paramilitary officer applies for the Defence Scholarship. Because their combined parental income is Rs. 4.5 Lakhs per year, they also ask to combine it with the Socio-Economic Need-Based Scholarship. Are they eligible for either scholarship, or both?

---

## Section 3: Multi-Hop & Cross-Document Reasoning (Q19–Q24)

**Q19** `[Domains: Attendance + Residence | Type: Multi-Hop]`
A hostel resident finishes an academic semester with 80% attendance across all registered courses and passes every subject. Can this student sit for the end-semester examinations, and are they eligible to continue staying in the hostel for the subsequent semester?

**Q20** `[Domains: Code of Conduct + Residence + Attendance | Type: Multi-Hop]`
A hostel resident is found guilty of damaging campus property (Level 3 misconduct) and receives a 7-day academic suspension along with cost recovery. Beyond the suspension and repair cost, what other consequences apply regarding their parents, placement assistance, end-semester exams, and staying in the hostel during the suspension?

**Q21** `[Domains: Attendance + Examinations | Type: Multi-Hop]`
A B.Tech student misses one regular end-semester exam because of the sudden demise of their mother, clears that paper on the next available attempt, and graduates in 8 semesters with an 8.6 CGPA and no other failures or absences. How does this single excused absence affect their eligibility for (a) University Ranks/Medals, and (b) First Class with Distinction?

**Q22** `[Domains: Attendance + Scholarships | Type: Multi-Hop]`
A student holding an SRM Merit Scholarship attends only 70% of classes in one course, does not receive condonation, and therefore receives an `I` grade in that course, while still maintaining an overall CGPA of 9.1. What happens to their course registration for that exam, and can they renew their SRM Merit Scholarship for the next academic year?

**Q23** `[Domains: Residence + Code of Conduct | Type: Multi-Hop]`
A hostel student is caught using an unauthorized hot plate in their hostel room and is fined, but fails to pay the hostel fine for 6 weeks. What consequences are triggered under the Hostel Rules, and how does the Code of Conduct classify unauthorized electrical appliances in a hostel room?

**Q24** `[Domains: Academic Integrity + Scholarships | Type: Multi-Hop]`
If a scholarship-holding student is found guilty of severe plagiarism and recommended for rustication for a limited period, who takes the final disciplinary action, which governing bodies ratify it, and what happens to the student's active scholarship?

---

## Section 4: Numerical, Calculation & Boundary Cases (Q25–Q30)

**Q25** `[Domain: Attendance & Academics | Type: Calculation]`
A B.Tech course is structured with 3 hours of Lecture (L), 1 hour of Tutorial (T), and 2 hours of Practice (P) per week. How many credits is this course worth? Additionally, if a student finishes their degree with a CGPA of 8.35, what is their official equivalent percentage?

**Q26** `[Domain: Attendance & Academics | Type: Boundary Case]`
A student finishes a Theory course with 63% attendance due to a genuine medical emergency and submits a condonation request. Can the HoD and Dean CET condone the attendance shortfall so the student can write the end-semester exam? Show the exact threshold math.

**Q27** `[Domain: Examinations | Type: Calculation & Boundary]`
In a 100-mark Theory end-semester exam, Examiner 1 awards 42 marks. The student applies for revaluation, and Examiner 2 awards 55 marks. Does this trigger a third examiner? If a third examiner is appointed and awards 51 marks, what is the student's final mark?

**Q28** `[Domain: Hostels / Residence | Type: Calculation]`
A hostel student obtains prior warden approval to use both an 80W fan and a 150W air cooler in their room for the year. Later, the student leaves for vacation without submitting a duplicate room key even though their room has a vacant bed. Calculate the exact total annual appliance charges and the fine incurred.

**Q29** `[Domain: Scholarships | Type: Calculation & Stacking]`
A female applicant to the MBA program (KTR campus) whose Aadhaar address is outside Tamil Nadu and Puducherry scores 280 in NMAT. What total tuition fee waiver percentage is she eligible for, and what is the exact component-by-component breakdown?

**Q30** `[Domain: Scholarships | Type: Calculation & Boundary]`
A non-teaching staff member at SRMIST with 11 years of continuous service applies for the Employee Ward Scholarship for their first child (1st time), and later applies for their second child (2nd time). What tuition fee waiver percentage applies for each child, and could they apply a 3rd time for a third child?

---

## Section 5: Policy Conflict & Version Discrepancies (Q31–Q34)

**Q31** `[Domains: scholarship.html vs. scholarship_categories.pdf | Type: Version Conflict]`
What IIT JEE (Mains) rank cutoff is required to qualify for the Founder's Scholarship?

**Q32** `[Domain: code_of_conduct.html | Type: Internal Document Conflict]`
Under the Code of Conduct for Students, what misconduct level applies to the possession/consumption of tobacco/gutka or possession of weapons/pornography, and what is the corresponding punishment?

**Q33** `[Domains: code_of_conduct.html vs. attendance.pdf | Type: Cross-Document Conflict]`
Can a student who receives a Level 3 disciplinary sanction appeal the decision to a higher university authority?

**Q34** `[Domains: scholarship.html vs. scholarship_categories.pdf | Type: Version Conflict]`
If an SRMIST employee resigns and leaves the organization while their child is receiving the Employee Ward Scholarship, does the scholarship continue or can it be withdrawn?

---

## Section 6: Unanswerable, Out-of-Policy & False Premise (Q35–Q40)

**Q35** `[Type: Out-of-Policy / Unanswerable]`
What is the maximum number of books an undergraduate student can borrow at one time from the SRMIST Central Library, and what is the overdue fine per day?

**Q36** `[Type: Out-of-Policy / Unanswerable]`
What is the exact annual tuition fee in INR for the B.Tech Computer Science and Engineering program at the Kattankulathur campus for the 2025–26 academic year?

**Q37** `[Type: Out-of-Policy / Unanswerable]`
What is the daily Wi-Fi data limit and per-student internet bandwidth cap in SRMIST hostels?

**Q38** `[Type: Out-of-Policy / Unanswerable]`
What is the monthly fellowship stipend for full-time Ph.D. scholars at SRMIST, and what is the formal dress code policy for faculty members?

**Q39** `[Domain: Examinations | Type: False Premise]`
Within how many days of result declaration can a student apply for revaluation of their final-year Viva Voce and Practical Lab examination marks?

**Q40** `[Domain: Hostels / Residence | Type: False Premise]`
How much is the monthly parking permit fee for a hostel resident to keep their personal car inside the hostel premises, and how much extra does it cost to book a single-occupancy hostel room?
