"""
Tests for Phase 4: the verifier.

The NLI model itself is replaced by a stub here, so this whole file runs in
milliseconds with no model download. That is deliberate and matches how the
rest of the suite works: these tests check the *decision logic* - what happens
when the checks disagree - not whether the model is any good. Model quality is
measured on real passages by the evaluation harness, not asserted in a unit
test.

Several cases below encode the specific failure this project exists to catch:
a claim that reads as perfectly supported but has the wrong number in it.
"""

from __future__ import annotations

import json

from policyverify.config import load_config
from policyverify.schema import (
    CheckResults,
    Chunk,
    Claim,
    ClaimVerdict,
    PolicyType,
    RetrievedChunk,
    VerdictStatus,
)
from policyverify.verify.citation import check_citation, check_citations
from policyverify.verify.nli import (
    CONTRADICTION,
    ENTAILMENT,
    NEUTRAL,
    NLIChecker,
    NLIResult,
    combine_unit_results,
    evidence_units,
    find_exact_quote,
    flatten,
    normalise_label,
    sentence_windows,
    split_sentences,
    split_windows,
)
from policyverify.verify.numeric import check_numbers, extract_numbers
from policyverify.verify.verifier import policy_says, verify_claim, verify_claims

CITATION = "srm/attendance/7.3-minimum_attendance"
PASSAGE = (
    "A student must maintain a minimum attendance record of at least 75% in "
    "individual courses, exclusive of leave of absence due to medical reasons."
)


class StubNLI:
    """Returns a fixed verdict, so decision logic can be tested in isolation."""

    def __init__(self, label: str, entailment: float = 0.0, contradiction: float = 0.0):
        self._result = NLIResult(
            label=label,
            entailment=entailment,
            neutral=max(0.0, 1.0 - entailment - contradiction),
            contradiction=contradiction,
        )
        self.calls = 0

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        self.calls += 1
        return self._result


def _retrieved(citation: str = CITATION, text: str = PASSAGE, part: int = 0) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{citation}#{part}",
        text=text,
        citation_id=citation,
        university="srm",
        university_name="SRM Institute of Science and Technology",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Academic Regulations > Minimum Attendance",
        source_url="https://example.srmist.edu.in/regs",
    )
    return RetrievedChunk(chunk=chunk, score=0.9, rank=0)


# ---------------------------------------------------------------------------
# numeric guard
# ---------------------------------------------------------------------------


def test_extract_numbers_handles_percentages_and_plain_values():
    numbers = extract_numbers("at least 75% within 14 days and a GPA of 3.0")
    values = [(n.value, n.is_percent) for n in numbers]
    assert (75.0, True) in values
    assert (14.0, False) in values
    assert (3.0, False) in values


def test_extract_numbers_handles_the_spaced_percent_pdfs_produce():
    numbers = extract_numbers("a minimum of 75 % attendance")
    assert numbers[0].value == 75.0
    assert numbers[0].is_percent


def test_extract_numbers_handles_thousands_separators():
    assert extract_numbers("a fine of 1,200 rupees")[0].value == 1200.0


def test_numeric_check_passes_when_numbers_match():
    result = check_numbers("Students need 75% attendance.", [PASSAGE])
    assert result.ok


def test_numeric_check_catches_the_75_vs_80_trap():
    """The error that actually harms a student: fluent, plausible, and wrong
    by one digit."""
    result = check_numbers("Students need 80% attendance.", [PASSAGE])
    assert not result.ok
    assert "80%" in result.detail


def test_numeric_check_does_not_confuse_a_percentage_with_a_bare_number():
    """"75 days" is not evidence for "75%"."""
    result = check_numbers("Students need 75% attendance.", ["a period of 75 days"])
    assert not result.ok


def test_numeric_check_passes_when_the_claim_has_no_numbers():
    result = check_numbers("Attendance is compulsory.", [PASSAGE])
    assert result.ok
    assert "no numbers" in result.detail


def test_numeric_check_matches_across_several_premises():
    """A claim may legitimately combine two cited passages."""
    result = check_numbers(
        "Students need 75% attendance and may appeal within 14 days.",
        [PASSAGE, "An appeal must be lodged within 14 days of the result."],
    )
    assert result.ok


def test_numeric_check_ignores_small_list_markers():
    """"(i) 1." style numbering is not a fact being asserted, and flagging it
    produced false alarms without catching anything real."""
    result = check_numbers("Rule 1 applies to all students.", ["Some text with no digits."])
    assert result.ok


# ---------------------------------------------------------------------------
# citation checks
# ---------------------------------------------------------------------------


def test_citation_resolves_when_it_was_retrieved():
    check = check_citation(CITATION, [_retrieved()])
    assert check.exists and check.was_retrieved
    assert not check.fabricated


def test_citation_marked_fabricated_when_it_exists_nowhere():
    check = check_citation("srm/attendance/99-invented", [_retrieved()], store=None)
    assert check.fabricated
    assert "fabricated" in check.describe()


def test_malformed_citation_is_not_well_formed():
    check = check_citation("not-a-citation", [_retrieved()])
    assert not check.well_formed
    assert check.fabricated


def test_citation_that_exists_in_corpus_but_was_not_retrieved():
    """A different failure from fabrication - the section is real, the model
    just cited something it was never shown."""

    class StubStore:
        def get_by_citation(self, citation_id):
            return [_retrieved("srm/attendance/9.9-elsewhere").chunk]

    check = check_citation("srm/attendance/9.9-elsewhere", [_retrieved()], store=StubStore())
    assert check.exists
    assert not check.was_retrieved
    assert not check.fabricated
    assert "not among the retrieved" in check.describe()


def test_store_failure_is_not_reported_as_fabrication():
    """Blaming the model for our own broken store would be a false accusation."""

    class BrokenStore:
        def get_by_citation(self, citation_id):
            raise RuntimeError("index unavailable")

    check = check_citation("srm/attendance/9.9-elsewhere", [], store=BrokenStore())
    assert not check.exists  # cannot confirm it
    # but the check completed rather than raising
    assert check.well_formed


def test_check_citations_handles_several_at_once():
    checks = check_citations([CITATION, "srm/x/made-up"], [_retrieved()])
    assert len(checks) == 2
    assert checks[0].exists
    assert checks[1].fabricated


# ---------------------------------------------------------------------------
# NLI helpers
# ---------------------------------------------------------------------------


def test_split_windows_leaves_short_passages_alone():
    assert split_windows("Short passage.") == ["Short passage."]


def test_split_windows_splits_long_passages():
    text = " ".join(f"Sentence number {i} about policy." for i in range(200))
    windows = split_windows(text, window_chars=400, overlap=50)
    assert len(windows) > 1
    assert all(len(w) <= 400 + 50 for w in windows)


def test_split_windows_preserves_content():
    """Truncation would silently drop the sentence that supports a claim -
    the exact silent failure this splitting exists to prevent."""
    sentences = [f"Rule {i} is unique and specific." for i in range(40)]
    text = " ".join(sentences)
    joined = " ".join(split_windows(text, window_chars=300, overlap=40))
    for sentence in sentences:
        assert sentence in joined


def test_split_windows_handles_empty_text():
    assert split_windows("") == []


def test_normalise_label_maps_checkpoint_variants():
    assert normalise_label("ENTAILMENT") == ENTAILMENT
    assert normalise_label("contradiction") == CONTRADICTION
    assert normalise_label("LABEL_1") == NEUTRAL


# ---------------------------------------------------------------------------
# verifier: how the checks combine
# ---------------------------------------------------------------------------


def test_supported_when_nli_entails_and_numbers_match():
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, entailment=0.95), config=load_config()
    )
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.score == 0.95
    assert verdict.checks.citation_supports is True


def test_refuted_when_nli_contradicts():
    claim = Claim(text="Students need no attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim,
        [_retrieved()],
        nli=StubNLI(CONTRADICTION, contradiction=0.97),
        config=load_config(),
    )
    assert verdict.status is VerdictStatus.REFUTED
    assert "contradicts" in verdict.explanation


def test_numeric_mismatch_vetoes_an_otherwise_supported_claim():
    """The core guard. Even when the passage reads as supporting the claim,
    a number that is not in the passage withholds support - fluent agreement
    does not make a figure right."""
    claim = Claim(text="Students need 80% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, entailment=0.97), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.checks.numeric_ok is False
    assert "numbers do not match" in verdict.explanation


def test_numeric_guard_cannot_grant_support_on_its_own():
    """Matching digits is not proof of meaning. Only NLI can grant support."""
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(NEUTRAL, entailment=0.10), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL


def test_claim_with_no_citation_is_not_supported():
    claim = Claim(text="Attendance is 75%.", citation_ids=[])
    verdict = verify_claim(claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.99), config=load_config())
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.score == 0.0
    assert "No citation given" in verdict.explanation


def test_claim_citing_only_a_fabricated_section_is_not_supported():
    claim = Claim(text="Attendance is 75%.", citation_ids=["srm/attendance/99-invented"])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.99), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL
    assert verdict.checks.citation_exists is False
    assert "fabricated" in verdict.explanation


def test_fabricated_citation_does_not_block_a_valid_one_on_the_same_claim():
    claim = Claim(
        text="Students need 75% attendance.",
        citation_ids=[CITATION, "srm/attendance/99-invented"],
    )
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.95), config=load_config()
    )
    assert verdict.status is VerdictStatus.SUPPORTED
    assert "could not be resolved" in verdict.explanation


def test_weak_entailment_stays_unsure_rather_than_supported():
    claim = Claim(text="Attendance is compulsory.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, entailment=0.30), config=load_config()
    )
    assert verdict.status is VerdictStatus.NEUTRAL


def test_verdict_records_the_evidence_it_used():
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(
        claim, [_retrieved()], nli=StubNLI(ENTAILMENT, 0.95), config=load_config()
    )
    assert verdict.evidence_chunk_ids == [f"{CITATION}#0"]


def test_verify_claims_reuses_one_model_across_claims():
    claims = [
        Claim(text="Students need 75% attendance.", citation_ids=[CITATION]),
        Claim(text="Attendance is compulsory.", citation_ids=[CITATION]),
    ]
    nli = StubNLI(ENTAILMENT, entailment=0.95)
    verdicts = verify_claims(claims, [_retrieved()], nli=nli, config=load_config())
    assert len(verdicts) == 2
    assert nli.calls == 2  # one premise each, not reloaded per claim


# ---------------------------------------------------------------------------
# evidence units: sentences, short windows, and the table fallback
# ---------------------------------------------------------------------------

# Text as it comes out of a PDF: hard line breaks in the middle of sentences.
PDF_TEXT = (
    "A student must maintain a minimum attendance\nrecord of at least 75% in "
    "individual courses.\n  Condonation of attendance\nshall not exceed 10%. "
    "The Dean may grant\nleave on medical grounds."
)


def test_flatten_removes_pdf_line_breaks():
    assert "\n" not in flatten(PDF_TEXT)
    assert "attendance record" in flatten(PDF_TEXT)


def test_split_sentences_handles_pdf_newlines():
    sentences = split_sentences(PDF_TEXT)
    assert sentences == [
        "A student must maintain a minimum attendance record of at least 75% in "
        "individual courses.",
        "Condonation of attendance shall not exceed 10%.",
        "The Dean may grant leave on medical grounds.",
    ]


def test_split_sentences_does_not_split_on_lowercase_after_a_full_stop():
    """"e.g. the" is not a sentence boundary."""
    assert len(split_sentences("Fees, e.g. the hostel fee, are due in June.")) == 1


def test_table_text_without_full_stops_falls_back_to_overlapping_slices():
    """A table flattened out of a PDF has no sentence punctuation. It must
    still be scored - in pieces - rather than handed over as one wall of text
    or, worse, skipped."""
    table = " ".join(f"Grade O{i} marks 91-100 points 10" for i in range(40))
    pieces = split_sentences(table)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)
    # Nothing is dropped: the end of the table is covered...
    assert table.endswith(pieces[-1])
    # ...and neighbouring slices overlap, so a row cut at a boundary is
    # still whole in one of them.
    assert pieces[0][-40:] in pieces[1]


def test_sentence_windows_slide_over_neighbours():
    windows = sentence_windows(["A.", "B.", "C.", "D."], max_sentences=3)
    assert windows == ["A. B.", "B. C.", "C. D.", "A. B. C.", "B. C. D."]


def test_evidence_units_are_sentences_then_windows_without_repeats():
    units = evidence_units("One rule. Two rule. One rule.", max_sentences=3)
    assert units[:2] == ["One rule.", "Two rule."]
    assert len(units) == len(set(units))
    assert "One rule. Two rule." in units


def test_evidence_units_sentences_only_when_asked():
    units = evidence_units(PDF_TEXT, max_sentences=1)
    assert units == split_sentences(PDF_TEXT)


# ---------------------------------------------------------------------------
# the exact-quote shortcut
# ---------------------------------------------------------------------------


def test_exact_quote_is_found_across_pdf_line_breaks_and_punctuation():
    claim = "A student must maintain a minimum attendance record of at least 75 % in individual courses"
    evidence = find_exact_quote(PDF_TEXT, claim)
    # The evidence is the one sentence containing the quote, not the passage.
    assert evidence == split_sentences(PDF_TEXT)[0]


def test_exact_quote_too_short_does_not_count():
    """"at least 75%" appears in many passages that say different things."""
    assert find_exact_quote(PDF_TEXT, "at least 75% in individual courses") is None


def test_exact_quote_requires_an_actual_match():
    assert find_exact_quote(PDF_TEXT, "A student must maintain a minimum attendance of 80% in all courses") is None


def test_exact_quote_must_match_whole_words():
    text = "Students who attendance record minimum thresholds courses are listed below."
    assert find_exact_quote(text, "dents who attendance record minimum thresholds courses") is None


def test_exact_quote_preceded_by_a_negation_does_not_count():
    text = "It is not the case that students may retake any failed examination in the summer term."
    assert find_exact_quote(text, "students may retake any failed examination in the summer term") is None


class FakePipeline:
    """Stands in for the transformers pipeline: entails only premises that
    contain a given phrase, and records how it was called."""

    def __init__(self, entailing_phrase: str = "", contradicting_phrase: str = ""):
        self.entailing_phrase = entailing_phrase
        self.contradicting_phrase = contradicting_phrase
        self.calls: list[int] = []

    def __call__(self, inputs, batch_size=None):
        self.calls.append(len(inputs))
        out = []
        for pair in inputs:
            premise = pair["text"]
            if self.entailing_phrase and self.entailing_phrase in premise:
                e, n, c = 0.9, 0.05, 0.05
            elif self.contradicting_phrase and self.contradicting_phrase in premise:
                e, n, c = 0.05, 0.05, 0.9
            else:
                e, n, c = 0.05, 0.9, 0.05
            out.append(
                [
                    {"label": "entailment", "score": e},
                    {"label": "neutral", "score": n},
                    {"label": "contradiction", "score": c},
                ]
            )
        return out


def _checker(pipeline: FakePipeline) -> NLIChecker:
    checker = NLIChecker(load_config())
    checker._pipeline = pipeline
    return checker


def test_check_uses_the_quote_shortcut_without_calling_the_model():
    fake = FakePipeline()
    result = _checker(fake).check(
        PDF_TEXT, "Condonation of attendance shall not exceed 10% in any case at all"
    )
    # Not a quote (extra words), so the model IS called here...
    assert fake.calls
    fake = FakePipeline()
    result = _checker(fake).check(
        PDF_TEXT, "A student must maintain a minimum attendance record of at least 75%"
    )
    # ...but this one is a quote, so it is not.
    assert fake.calls == []
    assert result.label == ENTAILMENT and result.entailment == 1.0
    assert result.exact_quote
    assert result.window.startswith("A student must maintain")


def test_check_scores_sentences_in_one_batch_and_stops_when_one_entails():
    fake = FakePipeline(entailing_phrase="Dean may grant")
    result = _checker(fake).check(PDF_TEXT, "The Dean can approve medical leave.")
    assert fake.calls == [3]  # three sentences, one call, windows not needed
    assert result.label == ENTAILMENT
    # The FULL text of the deciding sentence, not a 120-character prefix.
    assert result.window == "The Dean may grant leave on medical grounds."


def test_check_falls_back_to_windows_when_no_sentence_entails():
    # Only the window joining the first two sentences contains this phrase.
    fake = FakePipeline(entailing_phrase="courses. Condonation")
    result = _checker(fake).check(PDF_TEXT, "Attendance is 75% and condonation 10%.")
    assert fake.calls == [3, 3]  # sentences, then 2 pairs + 1 triple
    assert result.label == ENTAILMENT
    assert "Condonation" in result.window


def test_combine_records_the_strongest_contradiction_even_when_support_wins():
    support = NLIResult(ENTAILMENT, 0.8, 0.1, 0.1, window="rule")
    against = NLIResult(CONTRADICTION, 0.0, 0.05, 0.95, window="exception")
    combined = combine_unit_results([against, support], threshold=0.5)
    assert combined.label == ENTAILMENT and combined.window == "rule"
    assert combined.max_contradiction == 0.95
    assert combined.max_contradiction_text == "exception"


# ---------------------------------------------------------------------------
# aggregation across cited passages
# ---------------------------------------------------------------------------

OTHER = "srm/attendance/7.5-condonation"


class ByPremiseNLI:
    """Returns a fixed result per premise text, so several passages can each
    say something different."""

    def __init__(self, results: dict[str, NLIResult]):
        self.results = results

    def check(self, premise: str, hypothesis: str) -> NLIResult:
        # The verifier puts the section heading in front of the chunk text
        # (premise_with_heading); key on the chunk text itself.
        if premise not in self.results:
            premise = premise.split("\n", 1)[-1]
        return self.results[premise]


def _result(label: str, e: float, c: float, window: str) -> NLIResult:
    return NLIResult(label=label, entailment=e, neutral=max(0.0, 1 - e - c), contradiction=c, window=window)


def test_aggregation_does_not_depend_on_the_order_of_premises():
    """A strong contradiction must not be overwritten by a neutral from a
    later chunk that happens to have marginally higher entailment."""
    a, b = "passage A.", "passage B."
    nli = ByPremiseNLI(
        {
            a: _result(CONTRADICTION, 0.02, 0.95, "A says the opposite."),
            b: _result(NEUTRAL, 0.10, 0.05, "B is unrelated."),
        }
    )
    claim = Claim(text="Attendance is optional.", citation_ids=[CITATION])
    forward = [_retrieved(text=a, part=0), _retrieved(text=b, part=1)]
    backward = list(reversed(forward))
    cfg = load_config()
    v1 = verify_claim(claim, forward, nli=nli, config=cfg)
    v2 = verify_claim(claim, backward, nli=nli, config=cfg)
    assert v1.status is v2.status is VerdictStatus.REFUTED
    assert v1.score == v2.score == 0.95
    assert v1.checks.evidence_text == v2.checks.evidence_text == "A says the opposite."


def test_two_citations_that_disagree_are_reported_as_a_conflict():
    support_text, against_text = "Section one says 75%.", "Section two text."
    nli = ByPremiseNLI(
        {
            support_text: _result(ENTAILMENT, 0.75, 0.05, "Students need 75% attendance."),
            against_text: _result(CONTRADICTION, 0.05, 0.95, "No minimum attendance applies."),
        }
    )
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION, OTHER])
    retrieved = [_retrieved(CITATION, support_text), _retrieved(OTHER, against_text)]
    verdict = verify_claim(claim, retrieved, nli=nli, config=load_config())

    assert verdict.status is VerdictStatus.NEUTRAL  # not SUPPORTED
    assert verdict.checks.conflict is True
    assert verdict.checks.citation_supports is False
    assert verdict.checks.evidence_text == "Students need 75% attendance."
    assert verdict.checks.evidence_chunk_id == f"{CITATION}#0"
    assert verdict.checks.contradicting_text == "No minimum attendance applies."
    assert verdict.checks.contradicting_chunk_id == f"{OTHER}#0"
    # Both sections are named and both sentences quoted.
    assert CITATION in verdict.explanation and OTHER in verdict.explanation
    assert "No minimum attendance applies." in verdict.explanation
    assert "Students need 75% attendance." in verdict.explanation


def test_weak_contradiction_from_another_citation_is_not_a_conflict():
    """Below conflict_threshold, a loosely related section does not block
    well-supported claims."""
    support_text, against_text = "Section one says 75%.", "Section two text."
    nli = ByPremiseNLI(
        {
            support_text: _result(ENTAILMENT, 0.9, 0.05, "Students need 75% attendance."),
            against_text: _result(CONTRADICTION, 0.1, 0.7, "Something else."),
        }
    )
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION, OTHER])
    retrieved = [_retrieved(CITATION, support_text), _retrieved(OTHER, against_text)]
    verdict = verify_claim(claim, retrieved, nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.checks.conflict is False


def test_contradiction_within_one_citation_is_not_a_conflict():
    """A section that states a rule and its exception is normal: two chunks of
    the SAME section disagreeing must not block the claim."""
    rule, exception = "The rule chunk says 75%.", "The exception chunk."
    nli = ByPremiseNLI(
        {
            rule: _result(ENTAILMENT, 0.75, 0.05, "Students need 75% attendance."),
            exception: _result(CONTRADICTION, 0.02, 0.97, "Medical cases are exempt."),
        }
    )
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    retrieved = [_retrieved(text=rule, part=0), _retrieved(text=exception, part=1)]
    verdict = verify_claim(claim, retrieved, nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.checks.conflict is False


def test_same_citation_listed_twice_is_not_a_conflict():
    rule, exception = "The rule chunk says 75%.", "The exception chunk."
    nli = ByPremiseNLI(
        {
            rule: _result(ENTAILMENT, 0.75, 0.05, "Students need 75% attendance."),
            exception: _result(CONTRADICTION, 0.02, 0.97, "Medical cases are exempt."),
        }
    )
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION, CITATION])
    retrieved = [_retrieved(text=rule, part=0), _retrieved(text=exception, part=1)]
    verdict = verify_claim(claim, retrieved, nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.SUPPORTED


def test_supported_verdict_records_the_evidence_sentence():
    nli = ByPremiseNLI({PASSAGE: _result(ENTAILMENT, 0.92, 0.02, "A student must maintain 75%.")})
    claim = Claim(text="Students need 75% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(claim, [_retrieved()], nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.checks.evidence_text == "A student must maintain 75%."
    assert verdict.checks.evidence_chunk_id == f"{CITATION}#0"
    assert verdict.checks.conflict is False


def test_refuted_verdict_records_the_contradicting_sentence():
    nli = ByPremiseNLI({PASSAGE: _result(CONTRADICTION, 0.01, 0.96, "At least 75% is required.")})
    claim = Claim(text="No attendance is required.", citation_ids=[CITATION])
    verdict = verify_claim(claim, [_retrieved()], nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.REFUTED
    assert verdict.checks.evidence_text == "At least 75% is required."


def test_refuted_explanation_quotes_the_policy_sentence_and_section():
    nli = ByPremiseNLI({PASSAGE: _result(CONTRADICTION, 0.01, 0.96, "At least 75% is required.")})
    claim = Claim(text="Only 50% is required.", citation_ids=[CITATION])
    verdict = verify_claim(claim, [_retrieved()], nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.REFUTED
    assert 'The policy says: "At least 75% is required."' in verdict.explanation
    assert CITATION in verdict.explanation
    # Honest wording: it is a quoted sentence, nothing is generated.
    assert "word for word" in verdict.explanation
    # Already in the explanation, so the helper does not repeat it.
    assert policy_says(verdict) == []


def test_policy_says_fills_in_for_an_old_explanation_without_the_sentence():
    """Answers saved before the explanation carried the sentence still show it."""
    verdict = ClaimVerdict(
        claim=Claim(text="Only 50% is required.", citation_ids=[CITATION]),
        status=VerdictStatus.REFUTED,
        score=0.9,
        explanation="The cited policy text contradicts this claim (confidence 0.90).",
        checks=CheckResults(
            evidence_text="At least 75% is required.", evidence_chunk_id=f"{CITATION}#0"
        ),
    )
    lines = policy_says(verdict)
    assert len(lines) == 1
    assert 'The policy says: "At least 75% is required."' in lines[0]
    assert f"({CITATION})" in lines[0]


def test_policy_says_for_a_conflict_names_both_sections():
    verdict = ClaimVerdict(
        claim=Claim(text="75% is required.", citation_ids=[CITATION, OTHER]),
        status=VerdictStatus.NEUTRAL,
        score=0.4,
        explanation="The cited sources disagree.",
        checks=CheckResults(
            conflict=True,
            evidence_text="Students need 75%.",
            evidence_chunk_id=f"{CITATION}#0",
            contradicting_text="No minimum applies.",
            contradicting_chunk_id=f"{OTHER}#2",
        ),
    )
    lines = policy_says(verdict)
    assert len(lines) == 2
    assert CITATION in lines[0] and "Students need 75%." in lines[0]
    assert OTHER in lines[1] and "No minimum applies." in lines[1]


def test_policy_says_is_empty_without_evidence_fields_or_when_supported():
    old = ClaimVerdict(
        claim=Claim(text="x", citation_ids=[CITATION]),
        status=VerdictStatus.REFUTED,
        score=0.9,
        explanation="old",
        checks=CheckResults(evidence_text=None, contradicting_text=None),
    )
    assert policy_says(old) == []
    supported = old.model_copy(
        update={
            "status": VerdictStatus.SUPPORTED,
            "checks": CheckResults(evidence_text="A sentence.", evidence_chunk_id=f"{CITATION}#0"),
        }
    )
    assert policy_says(supported) == []


def test_neutral_verdict_has_no_evidence_sentence():
    verdict = verify_claim(
        Claim(text="Attendance is compulsory.", citation_ids=[CITATION]),
        [_retrieved()],
        nli=StubNLI(NEUTRAL, entailment=0.10),
        config=load_config(),
    )
    assert verdict.checks.evidence_text is None


def test_exact_quote_support_is_named_in_the_explanation():
    quote = NLIResult(ENTAILMENT, 1.0, 0.0, 0.0, window=PASSAGE, exact_quote=True)
    nli = ByPremiseNLI({PASSAGE: quote})
    claim = Claim(text="Students need at least 75% in individual courses.", citation_ids=[CITATION])
    verdict = verify_claim(claim, [_retrieved()], nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.SUPPORTED
    assert verdict.checks.exact_quote is True
    assert "word for word" in verdict.explanation


def test_exact_quote_still_goes_through_the_numeric_guard():
    quote = NLIResult(ENTAILMENT, 1.0, 0.0, 0.0, window=PASSAGE, exact_quote=True)
    nli = ByPremiseNLI({PASSAGE: quote})
    claim = Claim(text="Students need 80% attendance.", citation_ids=[CITATION])
    verdict = verify_claim(claim, [_retrieved()], nli=nli, config=load_config())
    assert verdict.status is VerdictStatus.NEUTRAL


def test_old_check_results_json_without_the_new_fields_still_loads():
    """Run logs saved before evidence_text and conflict existed must load."""
    old = {
        "claim": {"text": "Students need 75% attendance.", "citation_ids": [CITATION]},
        "status": "SUPPORTED",
        "score": 0.95,
        "evidence_chunk_ids": [f"{CITATION}#0"],
        "checks": {
            "nli_label": "entailment",
            "nli_score": 0.95,
            "numeric_ok": True,
            "numeric_detail": "all numbers found",
            "citation_exists": True,
            "citation_supports": True,
        },
        "explanation": "The cited policy text supports this claim (confidence 0.95).",
    }
    verdict = ClaimVerdict.model_validate_json(json.dumps(old))
    assert verdict.checks.evidence_text is None
    assert verdict.checks.conflict is False
    assert CheckResults.model_validate(old["checks"]).exact_quote is False
