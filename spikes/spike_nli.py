"""
SPIKE 2 of 3 - is an NLI model good enough to be our fact-checker?

WHY THIS MATTERS MOST
    The verifier is the heart of this project, and this spike decides whether
    the approach works at all. If no NLI model can reliably tell "the policy
    says 75%" from "the policy says 80%" on real policy language, you need to
    know that in week 1 - while it is still cheap to change plan - and not in
    week 6 with the whole pipeline already built around it.

WHAT NLI IS
    Natural Language Inference. The model is given two pieces of text and
    answers one narrow question:

        premise:    "Students must attend at least 75% of scheduled classes."
        hypothesis: "The attendance requirement is 75%."
        answer:     ENTAILMENT   (the premise proves the hypothesis)

    The three possible answers are:
        ENTAILMENT     - the passage proves the claim        -> SUPPORTED
        CONTRADICTION  - the passage disproves the claim     -> REFUTED
        NEUTRAL        - the passage says nothing either way -> NEUTRAL

    That maps exactly onto what we need, which is why we use an NLI model
    rather than asking the language model to grade its own homework. A model
    checking its own work tends to repeat its own mistakes: the same
    misreading that produced the bad claim produces a confident bad grade.

HOW TO USE IT
        python spikes/spike_nli.py

    First run downloads the models (a few hundred MB each), so give it time.
    Everything runs on CPU, leaving the GPU free for Qwen.

WHAT TO LOOK FOR
    Overall accuracy is the headline, but look at the per-category numbers.
    The NUMERIC cases matter most - those are the "75% vs 80%" errors that
    actually hurt a student relying on this system. A model that scores well
    overall but fails the numeric cases is the wrong model for this project.

    Whichever wins, put it in config.yaml under nli.model_name.
"""

from __future__ import annotations

from dataclasses import dataclass

# Candidate checkpoints, all CPU-friendly. These are starting points, not a
# recommendation - the whole point of this spike is to measure rather than
# assume. Add or remove freely.
CANDIDATES = [
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
]


@dataclass
class Case:
    """One hand-written test: does this passage prove this claim?"""

    category: str
    premise: str  # the policy passage
    hypothesis: str  # the claim to check
    gold: str  # what the right answer is


# ---------------------------------------------------------------------------
# 20 hand-written cases, in the style of real university policy text.
#
# These are deliberately written by hand rather than sampled from a dataset.
# The whole point is to test the model on the exact kind of language, and the
# exact kind of mistake, that this system will meet in practice.
# ---------------------------------------------------------------------------

CASES: list[Case] = [
    # --- Straightforward support -------------------------------------------
    Case(
        "basic",
        "Students must attend at least 75% of all scheduled classes in each "
        "course to be eligible to sit the final examination.",
        "Students need 75% attendance to sit the final exam.",
        "ENTAILMENT",
    ),
    Case(
        "basic",
        "The library remains open from 8:00 AM to 11:00 PM on weekdays during "
        "the academic term.",
        "The library closes at 11 PM on weekdays.",
        "ENTAILMENT",
    ),
    Case(
        "basic",
        "A student found responsible for a second act of academic misconduct "
        "shall be suspended for a minimum of one semester.",
        "A second academic misconduct offence results in suspension.",
        "ENTAILMENT",
    ),
    Case(
        "basic",
        "Applications for the merit scholarship must be submitted before "
        "31 March each year.",
        "The merit scholarship deadline is 31 March.",
        "ENTAILMENT",
    ),
    # --- Numeric traps: THE most important category ------------------------
    Case(
        "numeric",
        "Students must attend at least 75% of all scheduled classes in each "
        "course to be eligible to sit the final examination.",
        "Students need 80% attendance to sit the final exam.",
        "CONTRADICTION",
    ),
    Case(
        "numeric",
        "Appeals against an examination result must be lodged within 14 days "
        "of the result being published.",
        "Students have 30 days to appeal an examination result.",
        "CONTRADICTION",
    ),
    Case(
        "numeric",
        "The scholarship provides a tuition waiver of 50% for the first year "
        "of study.",
        "The scholarship covers full tuition for the first year.",
        "CONTRADICTION",
    ),
    Case(
        "numeric",
        "A minimum cumulative grade point average of 3.0 is required to retain "
        "the scholarship.",
        "A GPA of 3.0 is required to keep the scholarship.",
        "ENTAILMENT",
    ),
    Case(
        "numeric",
        "Residents must vacate the hostel within 48 hours of their final "
        "examination.",
        "Residents must leave the hostel within 72 hours of their last exam.",
        "CONTRADICTION",
    ),
    # --- Exceptions and conditions -----------------------------------------
    Case(
        "exception",
        "Late submissions incur a penalty of 10% per day, except where a "
        "medical certificate has been approved by the faculty office.",
        "Late submissions are always penalised 10% per day.",
        "CONTRADICTION",
    ),
    Case(
        "exception",
        "Students may withdraw from a course without academic penalty up to "
        "the end of week 6, provided they have not already failed an "
        "assessment in that course.",
        "Students can always withdraw without penalty before week 6.",
        "CONTRADICTION",
    ),
    Case(
        "exception",
        "Attendance requirements may be waived for students representing the "
        "university in official sporting or cultural events.",
        "Attendance rules can be waived for official university representation.",
        "ENTAILMENT",
    ),
    # --- Neutral: the passage simply does not address the claim ------------
    Case(
        "neutral",
        "Students must attend at least 75% of all scheduled classes in each "
        "course to be eligible to sit the final examination.",
        "Students who miss classes must pay a fine.",
        "NEUTRAL",
    ),
    Case(
        "neutral",
        "The academic integrity policy applies to all coursework, examinations "
        "and research conducted under university supervision.",
        "Plagiarism results in immediate expulsion.",
        "NEUTRAL",
    ),
    Case(
        "neutral",
        "Hostel residents are responsible for keeping their allocated rooms "
        "clean and undamaged.",
        "Hostel fees are payable at the start of each semester.",
        "NEUTRAL",
    ),
    Case(
        "neutral",
        "Examination timetables are published no later than four weeks before "
        "the examination period begins.",
        "Examinations are held in the main auditorium.",
        "NEUTRAL",
    ),
    # --- Paraphrase: same meaning, very different words --------------------
    Case(
        "paraphrase",
        "No candidate shall be permitted to enter the examination hall after "
        "the first thirty minutes of the examination have elapsed.",
        "Students cannot enter the exam hall more than 30 minutes late.",
        "ENTAILMENT",
    ),
    Case(
        "paraphrase",
        "Any form of unauthorised collaboration on individual assessment tasks "
        "constitutes academic misconduct.",
        "Working with others on an individual assignment without permission is "
        "misconduct.",
        "ENTAILMENT",
    ),
    # --- Scope traps: right topic, wrong subject ---------------------------
    Case(
        "scope",
        "Postgraduate students must maintain a minimum attendance of 60% in "
        "all seminar courses.",
        "All students must maintain 60% attendance in seminars.",
        "CONTRADICTION",
    ),
    Case(
        "scope",
        "First-year undergraduate students are required to live in university "
        "accommodation unless granted an exemption.",
        "All undergraduate students must live in university accommodation.",
        "CONTRADICTION",
    ),
]


def normalise_label(raw: str) -> str:
    """Different checkpoints spell the labels differently. Map them to ours."""
    r = raw.lower().replace("-", "_")
    if "entail" in r or r in ("label_0", "0"):
        return "ENTAILMENT"
    if "contradict" in r or r in ("label_2", "2"):
        return "CONTRADICTION"
    if "neutral" in r or r in ("label_1", "1"):
        return "NEUTRAL"
    return raw.upper()


def evaluate(model_name: str) -> dict | None:
    """Run all 20 cases through one model and score it."""
    from transformers import pipeline

    print(f"\n  loading {model_name} ...", flush=True)
    try:
        clf = pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # -1 means CPU, keeping the GPU free for Qwen
            top_k=None,
        )
    except Exception as e:
        print(f"  FAILED to load: {type(e).__name__}: {e}")
        return None

    correct = 0
    by_cat: dict[str, list[int]] = {}
    failures: list[tuple[Case, str]] = []

    for case in CASES:
        try:
            # NLI models expect the premise and hypothesis as a pair. The
            # dict form is the portable way to pass that to a pipeline.
            out = clf({"text": case.premise, "text_pair": case.hypothesis})
            scores = out[0] if isinstance(out[0], list) else out
            best = max(scores, key=lambda d: d["score"])
            predicted = normalise_label(best["label"])
        except Exception as e:
            predicted = f"ERROR({type(e).__name__})"

        hit = predicted == case.gold
        correct += hit
        by_cat.setdefault(case.category, []).append(int(hit))
        if not hit:
            failures.append((case, predicted))

    total = len(CASES)
    print(f"  accuracy: {correct}/{total} = {correct / total:.0%}")
    print("  by category:")
    for cat, hits in sorted(by_cat.items()):
        marker = "  <-- most important" if cat == "numeric" else ""
        print(f"    {cat:<12} {sum(hits)}/{len(hits)}{marker}")

    if failures:
        print(f"  got {len(failures)} wrong:")
        for case, pred in failures[:5]:
            print(f"    [{case.category}] expected {case.gold}, said {pred}")
            print(f'      claim: "{case.hypothesis[:70]}"')

    return {
        "model": model_name,
        "accuracy": correct / total,
        "numeric": sum(by_cat.get("numeric", [])) / max(len(by_cat.get("numeric", [])), 1),
    }


def main() -> int:
    print(__doc__)
    print("=" * 78)
    print(f"Testing {len(CANDIDATES)} candidate models on {len(CASES)} hand-written cases")
    print("Everything runs on CPU. First run downloads models - be patient.")
    print("=" * 78)

    results = [r for name in CANDIDATES if (r := evaluate(name)) is not None]

    if not results:
        print("\nNo model loaded successfully. Check your internet connection.")
        return 1

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'OVERALL':<9} {'NUMERIC':<9} MODEL")
    for r in sorted(results, key=lambda d: -d["accuracy"]):
        print(f"  {r['accuracy']:<9.0%} {r['numeric']:<9.0%} {r['model']}")

    best = max(results, key=lambda d: (d["numeric"], d["accuracy"]))
    print(f"\n  Best on the numeric cases: {best['model']}")
    print("  -> put this in config.yaml under nli.model_name")
    print("\n  Rule of thumb: below ~70% overall, or below ~60% on numeric,")
    print("  plan to add an LLM judge as a second checking tier in Phase 4.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
