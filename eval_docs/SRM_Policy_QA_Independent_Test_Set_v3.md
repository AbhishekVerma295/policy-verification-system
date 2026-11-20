# SRM policy QA – independent test set (v3)

> **v3, 2026-11-20.** v2 with the changes below, each marked **[CHANGED 2026-11-20]**. The questions are
> unchanged. Machine-scoring version: `eval/final_test.jsonl`, items `fb-01` to `fb-50`.
>
> 1. **Q24 relabelled answerable.** The Scholarship Categories PDF (2025-26, amended), now part of the corpus,
>    gives the Defence Scholarship "up to 25%" tuition fee. (The web page alone states no percentage.)
> 2. **Q46 answer updated.** Still "No, not automatically", but the PDF lists several schemes that also cover
>    hostel/transport fees, not only the School of Law case.
> 3. **Q16–Q18:** the R 8.8 class table is now readable in the stored text, and the quoted values were confirmed.
> 4. **Q5 and Q29, known gap:** the plagiarism policy's type 1 ("Secondary sources") and preamble are not in
>    the system's index (text before the first detected heading is dropped). Found while converting this set;
>    deliberately NOT repaired, so the result measures the gap. Fixing it is a separate decision.
> 5. The answer key was written from stored text that later turned out to have lost some table data. All
>    required values were re-checked automatically against the repaired text on 2026-11-20 (see the
>    conversion note in `eval/final_test.jsonl`).


This is a revision of `SRM_Policy_QA_Independent_Test_Set.docx`. The answer key is based only on the policy text in `data/processed/srm/*.txt`. It is an independent test set and was not derived from the system's outputs.

Abbreviations:

| Abbreviation | Source document | File |
|---|---|---|
| AR | Academic Regulations 2021 | attendance.txt |
| EP | Examination Policy | examination.txt |
| PI | Plagiarism Policy | academic_integrity.txt |
| SP | Scholarship Policy | scholarship.txt |
| HR | Hostel Occupancy Rules & Regulations, January 2025 | residence.txt |
| CC | Code of Conduct for Students | code_of_conduct.txt |

## 1. Questions only

| ID | Category | Question |
|---|---|---|
| Q1 | NUMERIC | when does normal course registration usually happen, like how long before the sem starts? |
| Q2 | NUMERIC | If I miss the normal registration date for a genuine reason, how long do I have to complete late registration? |
| Q3 | NUMERIC | Under the Code of Conduct, how many days of academic suspension can be given for Level 4 misconduct? |
| Q4 | NUMERIC | If my CGPA is 8.2, what percentage does that convert to under the regulations? |
| Q5 | NUMERIC | How many types of plagiarism does the Plagiarism Policy list? |
| Q6 | NUMERIC | how early do exam fee dues get raised b4 exams? |
| Q7 | NUMERIC | How early must a Hall Superintendent report before an examination starts? |
| Q8 | NUMERIC | How many days of academic suspension can a Level 2 misconduct lead to under the Code of Conduct? |
| Q9 | NUMERIC | is the hostel ambulance avaliable all day? how many hours? |
| Q10 | NUMERIC | What's the longest someone can get rusticated for under the Code of Conduct? |
| Q11 | TABLE LOOKUP | I scored 1,350 in SAT. What tuition fee waiver does the applicable scholarship table give? |
| Q12 | TABLE LOOKUP | I scored 91 in SRMJEEM and I'm joining the MBA at KTR. What scholarship do I get? |
| Q13 | TABLE LOOKUP | got 96 percentile in MAT, what scholarship do I get? |
| Q14 | TABLE LOOKUP | My BBA CUET score is 92 percentile. What waiver does the table show? |
| Q15 | TABLE LOOKUP | I got 93% in my Plus Two board exams. What merit scholarship does the table give? |
| Q16 | TABLE LOOKUP | Int. M.Tech student with CGPA 7.2 – what class do they get? |
| Q17 | TABLE LOOKUP | An Integrated M.Tech student has CGPA 8.4 and an F grade in one semester, later cleared. What class does the table indicate? |
| Q18 | TABLE LOOKUP | What class is given to an Integrated M.Tech student with CGPA 5.6? |
| Q19 | NOT COVERED | Exam policy says late applications get a fine. how much is it exactly, in rupees? |
| Q20 | NOT COVERED | Does the examination policy give a fixed rupee amount for the normal examination application fee? |
| Q21 | NOT COVERED | The academic regulations mention the minimum percentage or CGPA prescribed by SRMIST for admission. What is that exact percentage for every B.Tech branch? |
| Q22 | NOT COVERED | does the hostel policy say how much they charge if my parents stay overnite? |
| Q23 | NOT COVERED | How many days before check-in do I have to hand in my hostel fee receipt? |
| Q24 | NUMERIC (was NOT COVERED) | What percentage tuition fee waiver does the Defence Scholarship give? |
| Q25 | NOT COVERED | What is the exact number of days within which a student must appeal a disciplinary decision? |
| Q26 | NOT COVERED | If I'm accused of plagiarism, exactly how many days do I get to respond to the allegation? |
| Q27 | FALSE PREMISE | I thought students can register for the next semester just by submitting the course registration form. Is that enough? |
| Q28 | FALSE PREMISE | I heard that students can get a refund of their exam fee if they simply don't attend the examination. Is that right? |
| Q29 | FALSE PREMISE | I used a meta-study for my report but only cited the original papers it reviewed. Since I credited the original authors, that isn't plagiarism, right? |
| Q30 | FALSE PREMISE | my roommate moved out, so I get the room to myself now, right? |
| Q31 | FALSE PREMISE | Hostel students are allowed to stay on through the whole summer vacation, so what do I need to do to keep my room over the summer? |
| Q32 | FALSE PREMISE | Can I organise a religious gathering inside the hostel without informing anyone, as long as it is only for residents? |
| Q33 | FALSE PREMISE | The Code of Conduct only covers what I do on campus, so anything I do outside campus can't count as misconduct, right? |
| Q34 | FALSE PREMISE | Is it correct that a student can receive a scholarship even if they have an academic backlog at the time of renewal? |
| Q35 | YES/NO WITH EXCEPTION | If I didn't appear for all the previous semester's exams because I was ill, can I still register for the next semester? |
| Q36 | YES/NO WITH EXCEPTION | Can a student take a non-programmable calculator into the examination hall? |
| Q37 | YES/NO WITH EXCEPTION | Can I walk out of the exam hall early if I'm done? |
| Q38 | YES/NO WITH EXCEPTION | can I bring my own data book or tables into the exam? |
| Q39 | YES/NO WITH EXCEPTION | Does SRM use software to detect plagiarism? |
| Q40 | YES/NO WITH EXCEPTION | Under the Code of Conduct, can I bring a passed-out senior onto campus? |
| Q41 | YES/NO WITH EXCEPTION | Can I use results and ideas from a group project I worked on in my own report? |
| Q42 | YES/NO WITH EXCEPTION | My mother is an SRM graduate. Can I get the Alumni Scholarship? |
| Q43 | CROSS-POLICY | The Academic Regulations and Examination Policy both discuss late registration. Are they talking about exactly the same situation? |
| Q44 | CROSS-POLICY | The Academic Regulations and Examination Policy both mention registration. Does completing course registration automatically mean that the examination application has been paid for? |
| Q45 | CROSS-POLICY | The Academic Regulations and the Hostel Rules both say when a student cannot stay in the hostel. Can I keep staying in the hostel (a) during a break in studies, or (b) during an academic suspension? |
| Q46 | CROSS-POLICY | The Scholarship Policy and Hostel Rules both mention hostel accommodation. Does receiving a tuition fee scholarship automatically mean that hostel accommodation is free? |
| Q47 | CROSS-POLICY | The Academic Regulations and Scholarship Policy both use CGPA. Does the CGPA used to calculate equivalent percentage automatically determine scholarship renewal eligibility? |
| Q48 | CROSS-POLICY | The Plagiarism Policy and the Code of Conduct both ask students to sign something. Is it the same document? |
| Q49 | CROSS-POLICY | The Academic Regulations and Hostel Rules both mention summer vacation. Does the summer semester mean students can automatically continue staying in their hostel? |
| Q50 | CROSS-POLICY | The Hostel Rules and the Code of Conduct both deal with ragging. Under which law do the Hostel Rules say ragging is punished, and what misconduct level does the Code of Conduct put ragging at? |

## 2. Full answer key

### NUMERIC

**Q1**
- ID: Q1
- Category: NUMERIC
- Question: when does normal course registration usually happen, like how long before the sem starts?
- Correct answer: Registration is normally completed at least one month before the semester starts.
- Source document(s): Academic Regulations 2021.
- Section: R 3.1 – Course Registration for a Semester.
- Exact supporting text: "The Institute generally has an online registration system, and the registration is normally done at least one month before the starting of each semester."
- Required value: `at least one month`
- Wrong answer to watch for: One week; two weeks.

**Q2**
- ID: Q2
- Category: NUMERIC
- Question: If I miss the normal registration date for a genuine reason, how long do I have to complete late registration?
- Correct answer: Late registration requires special permission from the respective Head of Department and must be completed no later than two weeks after classes commence.
- Source document(s): Academic Regulations 2021.
- Section: R 3.3 – Late Registration / Enrollment.
- Exact supporting text: "If, for any compelling reason like illness, etc., a student is unable to register / enroll on the day of registration /enrollment, he/she can register / enroll after obtaining special permission from the respective Heads of Department and should be done not later than two weeks from the commencement of classes."
- Required value: `two weeks`
- Wrong answer to watch for: One week; one month.

**Q3** (replaced)
- ID: Q3
- Category: NUMERIC
- Question: Under the Code of Conduct, how many days of academic suspension can be given for Level 4 misconduct?
- Correct answer: Fifteen days of academic suspension, coupled with tasks/assignments (alongside the other Level 4 consequences: parents signing an undertaking, being barred from semester-end examinations and/or the placement drive).
- Source document(s): Code of Conduct for Students.
- Section: 4 – Punishments, Level 4 consequences.
- Exact supporting text: "Fifteen days academic suspension coupled with tasks / assignments."
- Required value: `Fifteen days`
- Wrong answer to watch for: Seven days (Level 3); five days (Level 2).

**Q4** (required value corrected)
- ID: Q4
- Category: NUMERIC
- Question: If my CGPA is 8.2, what percentage does that convert to under the regulations?
- Correct answer: The equivalent percentage is 82%, calculated by multiplying CGPA by 10.
- Source document(s): Academic Regulations 2021.
- Section: R 8.7 – Equivalent Percentage Calculation.
- Exact supporting text: "Equivalent Percentage of marks = (CGPA * 10)"
- Required value: `82`
- Wrong answer to watch for: 8.2%; 82.2%.

**Q5** (replaced)
- ID: Q5
- Category: NUMERIC
- Question: How many types of plagiarism does the Plagiarism Policy list?
- Correct answer: Ten. The policy lists ten numbered types, from secondary sources (1) to complete plagiarism (10).
- Source document(s): Plagiarism Policy.
- Section: Types of Plagiarism.
- Exact supporting text: "Types of Plagiarism:" / "1. Secondary sources (Inaccurate citation)" / "10. Complete plagiarism (Intellectual Theft, Stealing)"
- Required value: `10`
- Wrong answer to watch for: 6; 9.

**Q6**
- ID: Q6
- Category: NUMERIC
- Question: how early do exam fee dues get raised b4 exams?
- Correct answer: Examination fee dues are raised online at least six weeks before the examinations commence.
- Source document(s): Examination Policy.
- Section: IV – Issue, Collection of Examination Application Forms & Payment of Examination Fees.
- Exact supporting text: "The examination fee dues are raised through online at least six weeks before the commencement of examinations."
- Required value: `six weeks`
- Wrong answer to watch for: Four weeks; two weeks.

**Q7**
- ID: Q7
- Category: NUMERIC
- Question: How early must a Hall Superintendent report before an examination starts?
- Correct answer: The Hall Superintendent must report at least one hour before the examination begins.
- Source document(s): Examination Policy.
- Section: Duties of Hall Superintendent.
- Exact supporting text: "HS should report to duty well in advance at least one hour prior to the commencement of Examinations."
- Required value: `one hour`
- Wrong answer to watch for: Half an hour; two hours.

**Q8** (replaced)
- ID: Q8
- Category: NUMERIC
- Question: How many days of academic suspension can a Level 2 misconduct lead to under the Code of Conduct?
- Correct answer: Five days of academic suspension (and/or intimation to parents).
- Source document(s): Code of Conduct for Students.
- Section: 4 – Punishments, Level 2 consequences.
- Exact supporting text: "Intimation to parents" / "Five days academic suspension"
- Required value: `Five days`
- Wrong answer to watch for: Three days (Level 1); seven days (Level 3).

**Q9**
- ID: Q9
- Category: NUMERIC
- Question: is the hostel ambulance avaliable all day? how many hours?
- Correct answer: An ambulance is available on campus 24 hours for emergency transportation to and from the hospital.
- Source document(s): Hostel Occupancy Rules & Regulations.
- Section: 26.0 – Medical Assistance.
- Exact supporting text: "Ambulance is available on campus 24 hours to transport students to hospital and back if any emergency arises."
- Required value: `24 hours`
- Wrong answer to watch for: 12 hours; only during hospital working hours.

**Q10**
- ID: Q10
- Category: NUMERIC
- Question: What's the longest someone can get rusticated for under the Code of Conduct?
- Correct answer: Rustication can be imposed for a period of up to one year.
- Source document(s): Code of Conduct for Students.
- Section: 4 – Punishments, (ix) Rustication.
- Exact supporting text: "The student is expelled temporarily from the university for a period up to one year for the misconduct."
- Required value: `one year`
- Wrong answer to watch for: Six months; two years.

### TABLE LOOKUP

**Q11**
- ID: Q11
- Category: TABLE LOOKUP
- Question: I scored 1,350 in SAT. What tuition fee waiver does the applicable scholarship table give?
- Correct answer: A score of 1,350 falls in the 1301–1400 band and gives a 50% tuition fee waiver.
- Source document(s): Scholarship Policy.
- Section: 16 School of Law, 5-year Integrated Programs – SAT Score table.
- Exact supporting text: "SAT SCORE | Category: 1301-1400 | Scholarship Name: Merit Scholarship 50% | Benefit: 50% Tuition Fee Waiver."
- Required value: `50% Tuition Fee Waiver`
- Wrong answer to watch for: 25%; 75%.

**Q12** (replaced)
- ID: Q12
- Category: TABLE LOOKUP
- Question: I scored 91 in SRMJEEM and I'm joining the MBA at KTR. What scholarship do I get?
- Correct answer: A score of 91 falls in the 90 to 93 band, which gives Merit Scholarship 25%, a 25% tuition fee waiver.
- Source document(s): Scholarship Policy.
- Section: 15 Faculty of Management, a) MBA (Only for KTR campus) – SRMJEEM score table.
- Exact supporting text: "Applicable for MBA (Only for KTR campus) - Based on SRM IST entrance examinations (SRMJEEM) score | Score: 90 to 93 | Scholarship Name: Merit Scholarship 25% | Benefit: 25% Tuition Fee Waiver."
- Required value: `25% Tuition Fee Waiver`
- Wrong answer to watch for: 50% (the 94 to 97 band); no scholarship.

**Q13**
- ID: Q13
- Category: TABLE LOOKUP
- Question: got 96 percentile in MAT, what scholarship do I get?
- Correct answer: The 95–97.99 percentile band gives a 50% tuition fee waiver.
- Source document(s): Scholarship Policy.
- Section: 15 Faculty of Management – MAT Score table.
- Exact supporting text: "MAT SCORE | Percentile: 95 to 97.99 Percentile | Scholarship Name: Merit Scholarship 50% | Benefit: 50% Tuition Fee Waiver."
- Required value: `50% Tuition Fee Waiver`
- Wrong answer to watch for: 25%; 100%.

**Q14**
- ID: Q14
- Category: TABLE LOOKUP
- Question: My BBA CUET score is 92 percentile. What waiver does the table show?
- Correct answer: The 90–94.99 percentile band gives a 75% tuition fee waiver.
- Source document(s): Scholarship Policy.
- Section: BBA – Based on Common University Entrance Test (CUET) Scores.
- Exact supporting text: "Percentile: 90 to 94.99 Percentile | Scholarship Name: Merit Scholarship 75% | Benefit: 75% Tuition Fee Waiver."
- Required value: `75% Tuition Fee Waiver`
- Wrong answer to watch for: 50%; 100%.

**Q15**
- ID: Q15
- Category: TABLE LOOKUP
- Question: I got 93% in my Plus Two board exams. What merit scholarship does the table give?
- Correct answer: The 90–94.99 percentage band gives a 10% merit scholarship, equivalent to a 10% tuition fee waiver.
- Source document(s): Scholarship Policy.
- Section: Plus 2 Board Examination Performance.
- Exact supporting text: "Plus 2 Board Examination Performance | Percentage: 90 to 94.99 Percentage | Scholarship Name: Merit Scholarship 10% | Benefit: 10% Tuition Fee Waiver."
- Required value: `10% Tuition Fee Waiver`
- Wrong answer to watch for: 15%; 25%.

**Q16**
- ID: Q16
- Category: TABLE LOOKUP
- Question: Int. M.Tech student with CGPA 7.2 – what class do they get?
- Correct answer: The student receives First Class.
- Source document(s): Academic Regulations 2021.
- Section: R 8.8 – Class / Distinction.
- Exact supporting text: "≥ 6.0 & < 8.0" / "First Class"
- Required value: `First Class`
- Wrong answer to watch for: Second Class; First Class with Distinction.

**Q17**
- ID: Q17
- Category: TABLE LOOKUP
- Question: An Integrated M.Tech student has CGPA 8.4 and an F grade in one semester, later cleared. What class does the table indicate?
- Correct answer: The student receives First Class, not First Class with Distinction, because the distinction condition requires completion without an F grade in any semester.
- Source document(s): Academic Regulations 2021.
- Section: R 8.8 – Class / Distinction.
- Exact supporting text: "with ‘Ab’ or ‘F’ or ‘I’ in any Semester but obtained pass grade (‘O’ to ‘P’) subsequently" / "First Class"
- Required value: `First Class`
- Wrong answer to watch for: First Class with Distinction; Second Class.

**Q18**
- ID: Q18
- Category: TABLE LOOKUP
- Question: What class is given to an Integrated M.Tech student with CGPA 5.6?
- Correct answer: The student receives Second Class.
- Source document(s): Academic Regulations 2021.
- Section: R 8.8 – Class / Distinction.
- Exact supporting text: "≥ 5.0 & < 6.0" / "Second Class"
- Required value: `Second Class`
- Wrong answer to watch for: First Class; First Class with Distinction.

### NOT COVERED

For these questions the correct system behaviour is to abstain rather than manufacture a number. The nearest related clause is quoted.

**Q19**
- ID: Q19
- Category: NOT COVERED
- Question: Exam policy says late applications get a fine. how much is it exactly, in rupees?
- Correct answer: The Examination Policy does not specify a fixed rupee amount for the late application fine.
- Source document(s): Examination Policy.
- Section: IV – Issue, Collection of Examination Application Forms & Payment of Examination Fees.
- Exact supporting text: No supporting clause for a specific amount. Nearest related sentence: "Late applications are collected with a minimum amount of fine."
- Required value: `none`
- Wrong answer to watch for: ₹500; ₹1,000.

**Q20**
- ID: Q20
- Category: NOT COVERED
- Question: Does the examination policy give a fixed rupee amount for the normal examination application fee?
- Correct answer: No fixed rupee amount is specified in the cited examination-fee provisions.
- Source document(s): Examination Policy.
- Section: IV – Issue, Collection of Examination Application Forms & Payment of Examination Fees.
- Exact supporting text: No supporting clause for a fixed amount. Nearest related sentence: "For each course (purely internal course, seminar, theory, practical, theory combined with laboratory, projects, pedagogy etc.,) which should be reflected in the mark sheet / grade sheet, there is prescribed fees for appearing the examinations."
- Required value: `none`
- Wrong answer to watch for: A fabricated fixed examination fee; treating the late fee as the normal examination fee.

**Q21**
- ID: Q21
- Category: NOT COVERED
- Question: The academic regulations mention the minimum percentage or CGPA prescribed by SRMIST for admission. What is that exact percentage for every B.Tech branch?
- Correct answer: The regulations refer to the minimum percentage or CGPA prescribed by SRMIST but do not give one universal numerical percentage for every B.Tech branch in that clause.
- Source document(s): Academic Regulations 2021.
- Section: R 1.2 – Eligibility for Admission.
- Exact supporting text: No supporting clause for one universal value. Nearest related sentence: "The Minimum Percentage of Marks / CGPA prescribed by SRMIST"
- Required value: `none`
- Wrong answer to watch for: 60%; 75%.

**Q22**
- ID: Q22
- Category: NOT COVERED
- Question: does the hostel policy say how much they charge if my parents stay overnite?
- Correct answer: The policy permits approved overnight stays by parents in genuine circumstances and says they will be charged accordingly, but does not state a fixed amount.
- Source document(s): Hostel Occupancy Rules & Regulations.
- Section: 21 – Visitors to the Hostels.
- Exact supporting text: No supporting clause for a specific amount. Nearest related sentence: "Approved overnight stay will be charged accordingly."
- Required value: `none`
- Wrong answer to watch for: ₹500; ₹1,000.

**Q23**
- ID: Q23
- Category: NOT COVERED
- Question: How many days before check-in do I have to hand in my hostel fee receipt?
- Correct answer: The policy requires the fee receipt to be submitted while checking in but does not specify a number of days beforehand.
- Source document(s): Hostel Occupancy Rules & Regulations.
- Section: 22.0 – Check-in.
- Exact supporting text: No supporting clause for a prior-day deadline. Nearest related sentence: "A copy of the fee receipt and undertakings (affidavits) have to be submitted to the warden/deputy warden/SRO while checking in to the hostel."
- Required value: `none`
- Wrong answer to watch for: 7 days; 15 days.

**Q24** (replaced; **[CHANGED 2026-11-20] relabelled answerable**)
- ID: Q24
- Category: NUMERIC (was NOT COVERED)
- Question: What percentage tuition fee waiver does the Defence Scholarship give?
- Correct answer: Up to 25% of the tuition fee, per the Scholarship Categories PDF (2025-26 Onwards, Amended), row XII Defence Scholarship (Defence Personnel; fresh and continuation). The scholarship web page itself gives no percentage, only who can apply and the prerequisites.
- Source document(s): SRMIST Scholarship Categories 2025-26 (Amended) PDF; Scholarship Policy page.
- Section: XII – Defence Scholarship (PDF); 12 – Defence Scholarship (page).
- Exact supporting text: "Defence Scholarship | Defence Personnel | Fresh | (i) Retired from defence (ii) Still Serving | - | - | up to 25%"
- Required value: `25%`
- Wrong answer to watch for: 50%; 100%; "not stated anywhere".

**Q25**
- ID: Q25
- Category: NOT COVERED
- Question: What is the exact number of days within which a student must appeal a disciplinary decision?
- Correct answer: The Student Code of Conduct describes the appeal stages but does not specify a fixed number of days for submitting the appeal.
- Source document(s): Code of Conduct for Students.
- Section: 7 – Appeal Process.
- Exact supporting text: No supporting clause for a numerical deadline. Nearest related sentence: "If a student is not satisfied with the decision of the Disciplinary Committee and the sanctions imposed by it, then he/she can appeal for his/her case in writing to the Registrar, SRMIST."
- Required value: `none`
- Wrong answer to watch for: 7 days; 15 days.

**Q26** (replaced)
- ID: Q26
- Category: NOT COVERED
- Question: If I'm accused of plagiarism, exactly how many days do I get to respond to the allegation?
- Correct answer: The Plagiarism Policy does not give a number of days. Under procedural fairness it only says the person has the right to a reasonable period of time to respond to the allegations.
- Source document(s): Plagiarism Policy (all six files searched; no response deadline for plagiarism allegations is given anywhere).
- Section: Procedure for handling alleged Plagiarism – Procedural Fairness, (b).
- Exact supporting text: No supporting clause for a number of days. Nearest related sentence: "(b) Have a reasonable period of time within which to respond to the allegations against them;"
- Required value: `none`
- Wrong answer to watch for: 7 days; 15 days; 30 days.

### FALSE PREMISE

**Q27**
- ID: Q27
- Category: FALSE PREMISE
- Question: I thought students can register for the next semester just by submitting the course registration form. Is that enough?
- Correct answer: No. Students must complete both course registration and semester enrolment. Semester enrolment also requires payment of the prescribed annual tuition fees.
- Source document(s): Academic Regulations 2021.
- Section: R 3.1 and R 3.2 – Course Registration and Semester Enrollment Requirements.
- Exact supporting text: "For a student, to attend classes he/she has to necessarily complete both course registration and semester enrolment"
- Required value: `both course registration and semester enrolment`
- Wrong answer to watch for: Course registration alone is sufficient; payment alone is sufficient.

**Q28**
- ID: Q28
- Category: FALSE PREMISE
- Question: I heard that students can get a refund of their exam fee if they simply don't attend the examination. Is that right?
- Correct answer: No. A candidate who fails to appear for an examination is not entitled to a refund or to carry the fee forward as a deposit for a subsequent examination.
- Source document(s): Examination Policy.
- Section: IV – Issue, Collection of Examination Application Forms & Payment of Examination Fees.
- Exact supporting text: "Fails to present himself/herself for an examination; or Has appeared in an examination provisionally and is subsequently declared ineligible by the Controller of Examinations, according to the Ordinances, shall not be entitled to refund of the fee or to have it kept in deposit for a subsequent examination."
- Required value: `shall not be entitled to refund`
- Wrong answer to watch for: Full refund; fee carried forward automatically.

**Q29** (replaced)
- ID: Q29
- Category: FALSE PREMISE
- Question: I used a meta-study for my report but only cited the original papers it reviewed. Since I credited the original authors, that isn't plagiarism, right?
- Correct answer: No. The policy calls this Secondary Source Plagiarism: using a secondary source such as a meta study but citing only the primary sources inside it. It fails to credit the authors of the secondary source and gives a false sense of how much review went into the work.
- Source document(s): Plagiarism Policy.
- Section: Types of Plagiarism – 1. Secondary sources (Inaccurate citation).
- Exact supporting text: "Secondary Source Plagiarism happens when a researcher uses a secondary source like a Meta study but only cites the primary sources contained within the secondary one."
- Required value: `Secondary Source Plagiarism`
- Wrong answer to watch for: Citing the primary sources is enough; it only counts as plagiarism if nothing is cited.

**Q30**
- ID: Q30
- Category: FALSE PREMISE
- Question: my roommate moved out, so I get the room to myself now, right?
- Correct answer: No. The resident must accept a new roommate or move in with another student who is staying single, as the hostel management instructs.
- Source document(s): Hostel Occupancy Rules & Regulations.
- Section: 24.0 – Miscellaneous.
- Exact supporting text: "Any hostel resident, whose roommate has checked out due to any reason, will accept a roommate or will shift with another hostel student who is staying single, as per the instructions of the hostel management."
- Required value: `will accept a roommate or will shift`
- Wrong answer to watch for: The remaining student automatically gets single occupancy; the room must remain vacant.

**Q31** (reworded so the question states the false premise)
- ID: Q31
- Category: FALSE PREMISE
- Question: Hostel students are allowed to stay on through the whole summer vacation, so what do I need to do to keep my room over the summer?
- Correct answer: The premise is wrong. All hostellers have to vacate the hostel during the summer vacation every year. A student may stay only if the university advises it for academic reasons, as decided by the hostel administration, and the same room next year is not assured.
- Source document(s): Hostel Occupancy Rules & Regulations.
- Section: 25.0 – Summer/Winter Break.
- Exact supporting text: "All Hostlers have to vacate the hostel during the summer vacation every year." / "In case a student is advised by the university to stay in the hostel during vacation due to academic reasons he/she may be allowed to stay as decided by the hostel administration."
- Required value: `have to vacate`
- Wrong answer to watch for: Explaining a procedure for keeping the room as if staying were normal; students can stay with a warden's verbal permission.

**Q32**
- ID: Q32
- Category: FALSE PREMISE
- Question: Can I organise a religious gathering inside the hostel without informing anyone, as long as it is only for residents?
- Correct answer: No. Functions, meetings and religious gatherings require special permission; to organise a function, prior permission from the Director (Hostels) through the respective hostel Warden is required.
- Source document(s): Hostel Occupancy Rules & Regulations.
- Section: 16.11 – General (Discipline).
- Exact supporting text: "Students should not arrange any function, meeting, or religious gathering within the hostel campus without special permission of the authorities." / "they are required to get prior permission from the Director (Hostels) through their respective hostel Warden."
- Required value: `prior permission from the Director (Hostels)`
- Wrong answer to watch for: No permission is required for residents-only gatherings; permission from any floor representative is sufficient.

**Q33** (replaced)
- ID: Q33
- Category: FALSE PREMISE
- Question: The Code of Conduct only covers what I do on campus, so anything I do outside campus can't count as misconduct, right?
- Correct answer: No. The Code of Conduct says misconduct committed inside or outside the campus is a violation of University discipline. A complaint of misconduct received from outside the campus is itself listed as Level 2 misconduct.
- Source document(s): Code of Conduct for Students.
- Section: 1 – The following are considered as misconduct (item 1); 4 – Punishments, Level 2.
- Exact supporting text: "Any act of misconduct committed by a student inside or outside the campus shall be an act of violation of discipline of the University" / "2. Any complaint of misconduct received from outside the campus."
- Required value: `inside or outside the campus`
- Wrong answer to watch for: Off-campus behaviour is outside the Code; only hostel or classroom conduct counts.

**Q34**
- ID: Q34
- Category: FALSE PREMISE
- Question: Is it correct that a student can receive a scholarship even if they have an academic backlog at the time of renewal?
- Correct answer: No. The scholarship renewal conditions require that the student have no arrears or backlogs at the time of application.
- Source document(s): Scholarship Policy.
- Section: Conditions for Renewal.
- Exact supporting text: "No arrears (or) backlogs at the time of application"
- Required value: `No arrears (or) backlogs`
- Wrong answer to watch for: One backlog is permitted; backlogs are considered only at graduation.

### YES/NO WITH EXCEPTION

**Q35**
- ID: Q35
- Category: YES/NO WITH EXCEPTION
- Question: If I didn't appear for all the previous semester's exams because I was ill, can I still register for the next semester?
- Correct answer: Yes, but only if the Vice-Chancellor permits it. The student must appeal to the Vice-Chancellor through the head of the institution within two weeks from the commencement of the university examinations, with valid (medical) reasons.
- Source document(s): Examination Policy.
- Section: Registration and Enrollment of Courses.
- Exact supporting text: "If a student did not appear for all the courses during the end semester examinations (University examinations) of the previous semester, he or she shall give an appeal to the Vice-Chancellor through concerned head of the institution within two weeks from the commencement of University examinations with valid reasons (medical grounds). Only if the Vice –Chancellor permits, he or she will be allowed to register for the courses in the subsequent semester."
- Required value: `Only if the Vice –Chancellor permits`
- Wrong answer to watch for: Automatic registration; permission from the class advisor alone.

**Q36**
- ID: Q36
- Category: YES/NO WITH EXCEPTION
- Question: Can a student take a non-programmable calculator into the examination hall?
- Correct answer: Yes. Non-programmable calculators are permitted, but programmable calculators and other electronic gadgets are prohibited.
- Source document(s): Examination Policy.
- Section: 14 – Permissible Materials Inside the Examination Hall.
- Exact supporting text: "However, students can bring non –programmable calculators"
- Required value: `non –programmable calculators`
- Wrong answer to watch for: All calculators are prohibited; programmable calculators are permitted.

**Q37**
- ID: Q37
- Category: YES/NO WITH EXCEPTION
- Question: Can I walk out of the exam hall early if I'm done?
- Correct answer: Generally no. Only on rare occasions may students be permitted to leave, and then only during the last half an hour of the examination.
- Source document(s): Examination Policy.
- Section: Instructions to Candidates Appearing for Examinations.
- Exact supporting text: "Only on rare occasion, the students may be permitted to leave the hall during the last half an hour of the examination."
- Required value: `last half an hour`
- Wrong answer to watch for: Students may leave at any time after submitting their answers; students may never leave under any circumstances.

**Q38**
- ID: Q38
- Category: YES/NO WITH EXCEPTION
- Question: can I bring my own data book or tables into the exam?
- Correct answer: Yes, provided they are properly attested by the concerned HOD with seal and contain no written material or additional sheets.
- Source document(s): Examination Policy.
- Section: Instructions to Candidates Appearing for Examinations.
- Exact supporting text: "Data books/IS codes/Tables brought by the students properly attested by the concerned HOD with seal and should not contain any written material or additional sheets in them."
- Required value: `attested by the concerned HOD with seal`
- Wrong answer to watch for: Any personally prepared notes are permitted; HOD attestation is unnecessary.

**Q39** (replaced)
- ID: Q39
- Category: YES/NO WITH EXCEPTION
- Question: Does SRM use software to detect plagiarism?
- Correct answer: Yes, SRM says it will use the best tools/software to detect plagiarism, but software is not the final word: the policy says detecting plagiarism is a judgment made by a person who understands the subject and knows the definition of plagiarism.
- Source document(s): Plagiarism Policy.
- Section: How to detect Plagiarism.
- Exact supporting text: "SRM Institute of Science and Technology will use the best tools / software to detect plagiarism." / "The detection of plagiarism is a judgment to be made by a person who understands the subject and who is also aware of the definition of plagiarism."
- Required value: `a judgment to be made by a person who understands the subject`
- Wrong answer to watch for: The software result alone decides; no software is used.

**Q40** (replaced)
- ID: Q40
- Category: YES/NO WITH EXCEPTION
- Question: Under the Code of Conduct, can I bring a passed-out senior onto campus?
- Correct answer: Yes, but only with proper permission. Bringing passed-out students or outsiders onto campus without proper permission is Level 1 misconduct. (This is about the campus; inside the hostel, Hostel Rules 16.8 separately forbids entertaining ex-students.)
- Source document(s): Code of Conduct for Students.
- Section: 3 – Student Conduct Expectations, Don'ts.
- Exact supporting text: "Avoid bringing passed-out students and/or outsiders inside the campus without proper permission. (Level 1)"
- Required value: `without proper permission`
- Wrong answer to watch for: Alumni can always enter freely; outsiders are never allowed in under any circumstances.

**Q41** (replaced)
- ID: Q41
- Category: YES/NO WITH EXCEPTION
- Question: Can I use results and ideas from a group project I worked on in my own report?
- Correct answer: Yes, but only if you cite the collaborative nature of the study and the participants involved. Using collaborative work without doing so is unethical collaboration, which the policy lists as a type of plagiarism.
- Source document(s): Plagiarism Policy.
- Section: Types of Plagiarism – 8. Unethical collaboration (Inaccurate Authorship).
- Exact supporting text: "Using written work, outcomes and ideas that are the result of collaboration, without citing the collaborative nature of the study and participants involved, is unethical."
- Required value: `citing the collaborative nature of the study and participants involved`
- Wrong answer to watch for: Group work can be reused freely because you contributed to it; collaborative work can never be reused.

**Q42** (replaced)
- ID: Q42
- Category: YES/NO WITH EXCEPTION
- Question: My mother is an SRM graduate. Can I get the Alumni Scholarship?
- Correct answer: Yes, if she completed her UG at SRM at least 15 years (or PG at least 10 years) before the scholarship application and was registered as alumni with SRMIST when you applied for admission. You must be the first living child and eligible for first-year admission; siblings cannot also apply.
- Source document(s): Scholarship Policy.
- Section: 9 – Alumni Scholarship.
- Exact supporting text: "Candidates whose parent is an Alumni of SRM having completed his/her UG at least 15 years/PG at least 10 years, before the application of scholarship." / "The parent must have registered as alumni with SRMIST at the time of application for admission"
- Required value: `UG at least 15 years/PG at least 10 years`
- Wrong answer to watch for: Any child of any graduate qualifies; no minimum gap since graduation.

### CROSS-POLICY

**Q43**
- ID: Q43
- Category: CROSS-POLICY
- Question: The Academic Regulations and Examination Policy both discuss late registration. Are they talking about exactly the same situation?
- Correct answer: No. The Academic Regulations cover late course registration for compelling reasons such as illness, which needs special permission from the HOD. The Examination Policy covers a student who did not appear for all previous-semester examinations, who needs the Vice-Chancellor's permission to register for the next semester.
- Source document(s): Academic Regulations 2021; Examination Policy.
- Section: AR R 3.3 – Late Registration / Enrollment; EP – Registration and Enrollment of Courses.
- Exact supporting text:
  - AR: "Late Registration / enrollment will not be encouraged." / "special permission from the respective Heads of Department"
  - EP: "If a student did not appear for all the courses during the end semester examinations (University examinations) of the previous semester" / "Only if the Vice –Chancellor permits"
- Required value: `special permission from the respective Heads of Department` versus `Only if the Vice –Chancellor permits`
- Wrong answer to watch for: Both situations require only HOD approval; both situations require only Vice-Chancellor approval.

**Q44**
- ID: Q44
- Category: CROSS-POLICY
- Question: The Academic Regulations and Examination Policy both mention registration. Does completing course registration automatically mean that the examination application has been paid for?
- Correct answer: No. Course registration is the process of signing up for courses. Registration for examination and payment of the examination fee are separate, mandatory requirements.
- Source document(s): Academic Regulations 2021; Examination Policy.
- Section: AR R 3.1 – Course Registration; EP – Admission to Examinations.
- Exact supporting text:
  - AR: "The process of signing-up for courses is called ‘Registration’."
  - EP: "Registration for examination is mandatory. There will be a fee for each course"
- Required value: `Registration for examination is mandatory`
- Wrong answer to watch for: Course registration automatically completes examination registration; examination fees are included automatically in course registration.

**Q45** (replaced)
- ID: Q45
- Category: CROSS-POLICY
- Question: The Academic Regulations and the Hostel Rules both say when a student cannot stay in the hostel. Can I keep staying in the hostel (a) during a break in studies, or (b) during an academic suspension?
- Correct answer: No in both cases. (a) The Academic Regulations say a student on a break of studies will not be permitted to stay in the SRMIST hostel. (b) The Hostel Rules say a hosteller is not permitted to stay in the hostel during an academic suspension period.
- Source document(s): Academic Regulations 2021; Hostel Occupancy Rules & Regulations.
- Section: AR R 3.6 – During the Break of Studies; HR 16.17 – General (Discipline).
- Exact supporting text:
  - AR: "R 3.6  During the Break of Studies, a student:" / "(ii) Will not be permitted to stay in the ‘Hostel’ facility provided by SRMIST"
  - HR: "A Hosteller is not permitted to stay in the hostel during his/her academic suspension period."
- Required value: `Will not be permitted to stay in the ‘Hostel’ facility` (AR) and `not permitted to stay in the hostel during his/her academic suspension period` (HR)
- Wrong answer to watch for: The student can stay as long as hostel fees are paid; only one of the two situations bars hostel stay.

**Q46** (HR quote corrected; question unchanged)
- ID: Q46
- Category: CROSS-POLICY
- Question: The Scholarship Policy and Hostel Rules both mention hostel accommodation. Does receiving a tuition fee scholarship automatically mean that hostel accommodation is free?
- Correct answer: **[CHANGED 2026-11-20]** No. A tuition fee waiver is not automatically a hostel fee waiver. Only specific schemes also cover hostel/transport fees. Per the Scholarship Categories PDF (2025-26, amended): Founder's (100%), Chancellor Constituency – Perambalur (100%), Socio-Economic NGO/Tribal (100%), Unnat Bharat Abhiyan (100%) and need-based (25-100%, seniors). Per the page: School of Law students from outside Tamil Nadu/Puducherry. Original wording follows: The Scholarship Policy gives a hostel-fee benefit only in specific cases (for example, School of Law students from outside Tamil Nadu/Puducherry who qualify through SRMJEEL scores and opt for the hostel), while the Hostel Rules require the applicable hostel fees to be paid at admission.
- Source document(s): Scholarship Policy; Hostel Occupancy Rules & Regulations.
- Section: SP 16 School of Law – Regional Diversity; HR 6.0 – Payment of Fees.
- Exact supporting text:
  - SP: "Students from outside Tamil Nadu/Puducherry who are eligible for any of the above scholarships based on their SRMJEEL scores will also receive an equivalent percentage of scholarship on their hostel fees, should they choose to opt for the hostel facility."
  - HR: "All the applicable hostel fees must be paid at the time of getting hostel admission."
- Required value: `equivalent percentage of scholarship on their hostel fees` (SP) and `must be paid at the time of getting hostel admission` (HR)
- Wrong answer to watch for: Every tuition fee scholarship automatically includes free hostel accommodation.

**Q47**
- ID: Q47
- Category: CROSS-POLICY
- Question: The Academic Regulations and Scholarship Policy both use CGPA. Does the CGPA used to calculate equivalent percentage automatically determine scholarship renewal eligibility?
- Correct answer: No. The Academic Regulations multiply CGPA by 10 to calculate equivalent percentage, while the Scholarship Policy sets separate renewal CGPA requirements for different scholarship types.
- Source document(s): Academic Regulations 2021; Scholarship Policy.
- Section: AR R 8.7 – Equivalent Percentage Calculation; SP – Conditions for Renewal.
- Exact supporting text:
  - AR: "Equivalent Percentage of marks = (CGPA * 10)"
  - SP: "Merit Scholarship – 8 CGPA" / "Gender / Regional Diversity Scholarship – 7 CGPA"
- Required value: `CGPA * 10` versus `8 CGPA` and `7 CGPA`
- Wrong answer to watch for: A 10 CGPA is required for every scholarship; equivalent percentage is the scholarship renewal threshold.

**Q48** (replaced)
- ID: Q48
- Category: CROSS-POLICY
- Question: The Plagiarism Policy and the Code of Conduct both ask students to sign something. Is it the same document?
- Correct answer: No. The Plagiarism Policy requires every student (UG, PG, Doctoral and Post Doctoral) to submit a signed statement that they know the plagiarism policy and that their work is their own. The Code of Conduct asks each student, together with a parent, to sign an affidavit at application and admission agreeing to abide by the Code of Conduct.
- Source document(s): Plagiarism Policy; Code of Conduct for Students.
- Section: PI – Compliance Statements; CC 2 – Disciplinary Action Policy.
- Exact supporting text:
  - PI: "All students (UG, PG, Doctoral and Post Doctoral) are required to submit a signed statement that they are aware of the plagiarism policy of the University"
  - CC: "At the time of application and on admission, each student along with his/her parent is requested to sign an affidavit willing to abide by this Code of Conduct."
- Required value: `signed statement` (PI) and `sign an affidavit` (CC)
- Wrong answer to watch for: One signed form covers both policies; the parent must also sign the plagiarism statement.

**Q49**
- ID: Q49
- Category: CROSS-POLICY
- Question: The Academic Regulations and Hostel Rules both mention summer vacation. Does the summer semester mean students can automatically continue staying in their hostel?
- Correct answer: No. The Academic Regulations describe summer-semester courses, whereas the Hostel Rules require residents to vacate during summer vacation unless the university advises them to stay for academic reasons and the hostel administration allows it.
- Source document(s): Academic Regulations 2021; Hostel Occupancy Rules & Regulations.
- Section: AR R 3.5 – Summer Semester; HR 25.0 – Summer/Winter Break.
- Exact supporting text:
  - AR: "During the summer vacation, i.e., (May - June), there may be one short Summer Semester."
  - HR: "All Hostlers have to vacate the hostel during the summer vacation every year."
- Required value: `one short Summer Semester` versus `have to vacate`
- Wrong answer to watch for: Enrolling in summer semester automatically grants hostel accommodation.

**Q50** (replaced)
- ID: Q50
- Category: CROSS-POLICY
- Question: The Hostel Rules and the Code of Conduct both deal with ragging. Under which law do the Hostel Rules say ragging is punished, and what misconduct level does the Code of Conduct put ragging at?
- Correct answer: The Hostel Rules say ragging is punished as per the Tamil Nadu Prohibition of Ragging Act, 1997 (and call it a cognizable offence). The Code of Conduct puts ragging at Level 6, whose consequence is expulsion from the University.
- Source document(s): Hostel Occupancy Rules & Regulations; Code of Conduct for Students.
- Section: HR 15.0 – Ragging (15.1, 15.4); CC 3 – Don'ts and 4 – Punishments, Level 6.
- Exact supporting text:
  - HR: "Ragging is a cognizable offence." / "shall be punished as per the Tamil Nadu Prohibition of Ragging Act, 1997."
  - CC: "Avoid ragging- a criminal offence which is prohibited in any form. (Level 6)" / "Expulsion from the University"
- Required value: `Tamil Nadu Prohibition of Ragging Act, 1997` (HR) and `Level 6` (CC)
- Wrong answer to watch for: Ragging is Level 1 to 3; the Hostel Rules cite only the UGC guidelines (those are cited in the Code of Conduct, not the Hostel Rules).

## 3. Coverage and quality-control note

### Category count

| Category | Questions | Count |
|---|---|---|
| NUMERIC | Q1–Q10 | 10 |
| TABLE LOOKUP | Q11–Q18 | 8 |
| NOT COVERED | Q19–Q26 | 8 |
| FALSE PREMISE | Q27–Q34 | 8 |
| YES/NO WITH EXCEPTION | Q35–Q42 | 8 |
| CROSS-POLICY | Q43–Q50 | 8 |
| Total | | 50 |

### Document coverage

Primary-source assignment (single-document questions), plus appearances in cross-policy questions.

| Document | Primary questions | Primary count | Cross-policy | Total appearances |
|---|---|---|---|---|
| Academic Regulations | Q1, Q2, Q4, Q16, Q17, Q18, Q21, Q27 | 8 | Q43, Q44, Q45, Q47, Q49 | 13 |
| Examination Policy | Q6, Q7, Q19, Q20, Q28, Q35, Q36, Q37, Q38 | 9 | Q43, Q44 | 11 |
| Plagiarism Policy | Q5, Q26, Q29, Q39, Q41 | 5 | Q48 | 6 |
| Scholarship Policy | Q11, Q12, Q13, Q14, Q15, Q24, Q34, Q42 | 8 | Q46, Q47 | 10 |
| Hostel Rules | Q9, Q22, Q23, Q30, Q31, Q32 | 6 | Q45, Q46, Q49, Q50 | 10 |
| Code of Conduct | Q3, Q8, Q10, Q25, Q33, Q40 | 6 | Q48, Q50 | 8 |

Every document has at least 5 primary questions.

### Tables used

| Table | Document | Questions |
|---|---|---|
| SAT Score scholarship table (School of Law) | Scholarship Policy | Q11 |
| SRMJEEM score table (MBA) | Scholarship Policy | Q12 |
| MAT Score scholarship table | Scholarship Policy | Q13 |
| BBA CUET scholarship table | Scholarship Policy | Q14 |
| Plus Two performance table | Scholarship Policy | Q15 |
| Integrated M.Tech Class/Distinction table | Academic Regulations | Q16–Q18 |

### Changes from v1

- Replaced: Q3, Q5, Q8, Q12, Q24, Q26, Q29, Q33, Q39, Q40, Q41, Q42, Q45, Q48, Q50.
- Corrected: Q4 required value (`82`); Q46 Hostel Rules quote (the v1 quote "The hostel fees should be paid as per the schedule announced by the hostel administration." does not appear in the Hostel Rules and was replaced with the actual 6.2 sentence).
- Reworded for category fit: Q31 (now states the false premise).
- Reworded into casual student language (facts and answers unchanged): Q1, Q6, Q9, Q10, Q13, Q16, Q19, Q22, Q23, Q30, Q37, Q38. Q6 ("b4"), Q9 ("avaliable"), Q16 ("Int."), Q22 ("overnite") contain a short form or typo on purpose.
- The v1 question-only table had stale wording for Q35, Q38, Q43 and Q49 that did not match the answer key; v2 uses the answer-key wording throughout.

### Unverified quotes carried over from v1

- Q16, Q17, Q18: the R 8.8 Integrated M.Tech Class/Distinction table is not readable in `attendance.txt` (only fragments such as "≥ 5.0" survive extraction). The quoted cell text ("≥ 6.0 & < 8.0", "First Class", etc.) could not be confirmed word for word and should be checked against the original PDF/HTML.

### Known overlaps carried over from v1 (not changed)

- Q31 and Q49 both rely on the Hostel Rules "have to vacate" summer clause.
- Q2 and Q43 both use the AR R 3.3 HOD-permission clause; Q35 and Q43 both use the EP Vice-Chancellor-permission clause.
- Q4 and Q47 both use the AR "CGPA * 10" formula.
- Q32 (function/gathering permission) remains; its duplicate Q40 was replaced.
