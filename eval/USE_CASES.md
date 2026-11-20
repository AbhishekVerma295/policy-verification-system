# Use cases and question types

What people ask the Policy Verification System, and which evaluation
questions test each kind. Every type below has questions in
`eval/adversarial.jsonl` (category column) or `eval/scenarios.jsonl`
(`sc-*`), so a change that breaks one kind of question shows up in a number.

The corpus is six SRM policies: attendance and academic regulations,
examinations, academic integrity (plagiarism), code of conduct, hostel
(residence) rules, and scholarships.

## Who asks

| User | Typical need | Example |
|---|---|---|
| Current student | A rule that affects their marks, attendance or exams | "What is the minimum attendance?" |
| Hostel resident | What they may do in the hostel, charges, visitors | "Can I keep my own air cooler?" |
| Parent / guardian | Visiting, staying over, being informed | "Can parents stay overnight in the hostel?" |
| Applicant | Scholarships and eligibility | "I got rank 300 in SRMJEEE - which scholarship?" |
| Student facing discipline | Penalties and appeals | "How do I appeal a disciplinary decision?" |
| Staff / reviewer | Checking what a policy actually says | "What does 75% mean in the scholarship policy?" |

## Kinds of question

| # | Kind | What a correct answer does | Example | Tested by |
|---|---|---|---|---|
| 1 | **Factual lookup** | States the rule, cited | "Are hostel fees refundable?" | `neg-007`, `sc-016` |
| 2 | **Numeric threshold** | Gives the exact figure with its qualifier ("at least", "cannot exceed") | "How much condonation can be granted at most?" | `numeric` (17 items) |
| 3 | **Eligibility / condition** | Says who qualifies and what they must keep doing | "My father is a retired army officer - can I get a scholarship?" | `sc-018`, `sc-019`, `num-016` |
| 4 | **Procedure / how-to** | Gives the steps and who to approach | "I lost my hall ticket - what do I do?" | `sc-006`, `sc-008`, `sc-020` |
| 5 | **Scenario (apply a rule to a situation)** | Finds the governing rule and applies it to the facts given | "I have 72% attendance - can I write the exam?" | `scenario` (21 answerable) |
| 6 | **Exception / negation** | Keeps the carve-out, not just the headline rule | "Does the 75% include medical absences?" | `negation_exception` (13 items) |
| 7 | **False premise** | Corrects the wrong assumption instead of explaining it | "Why does the university require 90% attendance?" | `false_premise` (12 items) |
| 8 | **Cross-policy / ambiguous figure** | Picks the right policy when a number means different things in different ones | "What does 75% mean in the scholarship policy?" | `cross_policy` (10 items) |
| 9 | **Consequence / penalty** | States what happens if a rule is broken | "A friend stayed overnight without approval - what happens to me?" | `sc-011`, `fp-007` |
| 10 | **Out of scope** | Declines and says the documents do not cover it - never guesses | "Can I keep a pet cat in my hostel room?" | `unanswerable` (12) and `sc-u01`..`sc-u06` |

## What "answering properly" means, per kind

- Every statement in an answer must be supported by the section it cites
  (verified by the NLI checker, the number check and the citation check).
- A statement that is true but does not answer the question is withheld
  (the relevance gate), so an answer never pads itself with nearby facts.
- When nothing verified answers the question, the system declines and says
  so. For kind 10 that is the correct answer; for kinds 1-9 it is a
  "false abstention" and counts against accuracy.
- A figure from the wrong policy or the wrong role (the Rs 2500 cooler rate
  instead of Rs 5000; the 25% leave allowance instead of the 10% condonation
  cap) counts as a wrong value - the worst outcome after a hallucination.

## Not covered (known limits)

- Policies outside the six documents (placements, fees schedules, transport,
  Wi-Fi). These are declined by design.
- Questions that need arithmetic across rules ("I have 68%; with maximum
  condonation will I reach 75%?"). The system states both rules but does not
  do the sum.
- Questions about a specific person's record. The system only knows policy.
