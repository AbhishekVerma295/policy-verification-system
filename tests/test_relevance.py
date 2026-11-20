"""
Tests for the relevance gate (verify/relevance.py).

No model, index or network: the similarity scorer and the language model are
both replaced by fakes. These tests check the plumbing - which signal decides
when, what is recorded, and what happens when the judge fails. Whether the
thresholds are RIGHT is measured on the tuning and fresh questions, not
asserted here.
"""

from __future__ import annotations

import json

from conftest import FakeLLM
from policyverify.config import load_config
from policyverify.llm import LLMError
from policyverify.schema import Claim, ClaimVerdict, VerdictStatus
from policyverify.verify.relevance import (
    JUDGE_SCHEMA,
    OFF_TOPIC_NOTE,
    EmbedderSimilarity,
    ask_judge,
    assess_relevance,
    similarity_for_store,
)

CITATION = "srm/attendance/7.3-minimum_attendance"


def _verdict(text="Students need 75% attendance.", status=VerdictStatus.SUPPORTED, score=0.95):
    return ClaimVerdict(
        claim=Claim(text=text, citation_ids=[CITATION]),
        status=status,
        score=score,
        explanation="checked",
    )


class FakeSimilarity:
    """Returns the scores it was given, in order, whatever it is asked."""

    def __init__(self, scores):
        self._scores = list(scores)
        self.calls = 0

    def scores(self, question, claims):
        self.calls += 1
        assert len(claims) == len(self._scores), "one score per claim"
        return list(self._scores)


class BrokenSimilarity:
    def scores(self, question, claims):
        raise RuntimeError("embedding model exploded")


class RaisingLLM:
    def generate(self, prompt, *, json_mode=True, schema=None):
        raise LLMError("ollama is down")


def _judge(answer: bool) -> FakeLLM:
    return FakeLLM([json.dumps({"answered": answer})])


def _cfg():
    config = load_config()
    config.relevance.enabled = True
    config.relevance.similarity_accept = 0.75
    config.relevance.similarity_reject = 0.55
    config.relevance.similarity_fallback = 0.65
    return config


# ---------------------------------------------------------------------------
# The two clear-cut cases never reach the model
# ---------------------------------------------------------------------------


def test_clearly_relevant_claim_is_accepted_without_asking_the_model():
    llm = _judge(False)  # would say "no" if asked - it must not be asked
    report = assess_relevance("q", [_verdict()], llm=llm, similarity=FakeSimilarity([0.9]), config=_cfg())
    marked = report.verdicts[0]
    assert marked.checks.relevant is True
    assert marked.checks.relevance_source == "similarity"
    assert marked.checks.relevance_similarity == 0.9
    assert llm.call_count == 0
    assert not report.judge_called and not report.judge_failed


def test_clearly_irrelevant_claim_is_rejected_without_asking_the_model():
    llm = _judge(True)
    report = assess_relevance("q", [_verdict()], llm=llm, similarity=FakeSimilarity([0.3]), config=_cfg())
    marked = report.verdicts[0]
    assert marked.checks.relevant is False
    assert marked.checks.relevance_source == "similarity"
    assert llm.call_count == 0


def test_rejected_claim_says_why_in_its_explanation_and_stays_supported():
    """The claim is TRUE, so its status must not change - only the reason it
    is withheld is new, and it is said plainly."""
    report = assess_relevance("q", [_verdict()], llm=_judge(True), similarity=FakeSimilarity([0.3]), config=_cfg())
    marked = report.verdicts[0]
    assert marked.status is VerdictStatus.SUPPORTED
    assert marked.explanation.startswith(OFF_TOPIC_NOTE)
    assert "checked" in marked.explanation, "the original explanation is kept"


# ---------------------------------------------------------------------------
# The borderline case asks the model, once
# ---------------------------------------------------------------------------


def test_borderline_answer_goes_to_the_model_and_yes_means_relevant():
    llm = _judge(True)
    report = assess_relevance("q", [_verdict()], llm=llm, similarity=FakeSimilarity([0.65]), config=_cfg())
    assert report.verdicts[0].checks.relevant is True
    assert report.verdicts[0].checks.relevance_source == "judge"
    assert report.judge_called
    assert llm.call_count == 1


def test_borderline_answer_and_model_says_no_means_not_relevant():
    report = assess_relevance("q", [_verdict()], llm=_judge(False), similarity=FakeSimilarity([0.65]), config=_cfg())
    assert report.verdicts[0].checks.relevant is False
    assert report.verdicts[0].checks.relevance_source == "judge"


def test_the_model_is_asked_once_per_answer_not_once_per_claim():
    llm = _judge(True)
    claims = [_verdict("A."), _verdict("B."), _verdict("C.")]
    assess_relevance("q", claims, llm=llm, similarity=FakeSimilarity([0.6, 0.65, 0.7]), config=_cfg())
    assert llm.call_count == 1


def test_judge_gets_a_schema_the_question_and_the_statements():
    llm = _judge(True)
    assess_relevance(
        "Who is the Vice Chancellor?",
        [_verdict("The Vice-Chancellor gives the welcome address.")],
        llm=llm,
        similarity=FakeSimilarity([0.65]),
        config=_cfg(),
    )
    assert llm.schemas == [JUDGE_SCHEMA]
    assert "Who is the Vice Chancellor?" in llm.prompts[0]
    assert "The Vice-Chancellor gives the welcome address." in llm.prompts[0]


def test_judge_prompt_tells_the_model_a_false_premise_correction_counts():
    """The main risk of the gate: a question that assumes something untrue is
    answered by the TRUE rule, and that must count as answering it."""
    llm = _judge(True)
    assess_relevance("q", [_verdict()], llm=llm, similarity=FakeSimilarity([0.65]), config=_cfg())
    assert "wrong assumption" in llm.prompts[0]
    assert "real rule or figure" in llm.prompts[0]


# ---------------------------------------------------------------------------
# Several claims
# ---------------------------------------------------------------------------


def test_a_tangent_beside_a_relevant_claim_is_withheld():
    claims = [_verdict("On topic."), _verdict("Unrelated tangent.")]
    report = assess_relevance("q", claims, llm=_judge(True), similarity=FakeSimilarity([0.9, 0.3]), config=_cfg())
    assert [v.checks.relevant for v in report.verdicts] == [True, False]


def test_all_claims_off_topic_when_the_best_one_is():
    claims = [_verdict("A."), _verdict("B.")]
    report = assess_relevance("q", claims, llm=_judge(True), similarity=FakeSimilarity([0.4, 0.3]), config=_cfg())
    assert [v.checks.relevant for v in report.verdicts] == [False, False]


def test_only_claims_that_would_be_kept_are_judged():
    """Refuted, unconfirmed and weakly supported claims are already removed by
    verification; the gate must leave them exactly as they were."""
    claims = [
        _verdict("Supported."),
        _verdict("Refuted.", VerdictStatus.REFUTED, 0.99),
        _verdict("Unconfirmed.", VerdictStatus.NEUTRAL, 0.1),
        _verdict("Weak.", VerdictStatus.SUPPORTED, 0.1),
    ]
    sim = FakeSimilarity([0.9])  # exactly one candidate -> exactly one score
    report = assess_relevance("q", claims, llm=_judge(True), similarity=sim, config=_cfg())
    assert report.verdicts[0].checks.relevant is True
    assert [v.checks.relevant for v in report.verdicts[1:]] == [None, None, None]
    assert [v.explanation for v in report.verdicts[1:]] == [v.explanation for v in claims[1:]]


def test_the_gate_never_drops_or_reorders_claims():
    claims = [_verdict("A."), _verdict("B.", VerdictStatus.REFUTED, 0.9), _verdict("C.")]
    report = assess_relevance("q", claims, llm=_judge(True), similarity=FakeSimilarity([0.3, 0.9]), config=_cfg())
    assert [v.claim.text for v in report.verdicts] == ["A.", "B.", "C."]


def test_input_verdicts_are_not_modified():
    original = _verdict()
    assess_relevance("q", [original], llm=_judge(True), similarity=FakeSimilarity([0.1]), config=_cfg())
    assert original.checks.relevant is None
    assert original.explanation == "checked"


def test_no_supported_claims_means_nothing_to_judge():
    llm = _judge(True)
    report = assess_relevance(
        "q", [_verdict(status=VerdictStatus.REFUTED)], llm=llm, similarity=FakeSimilarity([]), config=_cfg()
    )
    assert llm.call_count == 0
    assert report.verdicts[0].checks.relevant is None


def test_gate_off_changes_nothing():
    config = _cfg()
    config.relevance.enabled = False
    report = assess_relevance("q", [_verdict()], llm=_judge(False), similarity=FakeSimilarity([0.1]), config=config)
    assert report.verdicts[0].checks.relevant is None


# ---------------------------------------------------------------------------
# When the judge fails
# ---------------------------------------------------------------------------


def test_unusable_judge_reply_falls_back_to_similarity_and_says_so():
    config = _cfg()
    for sim, expected in ((0.70, True), (0.60, False)):
        llm = FakeLLM(["this is not json"])
        report = assess_relevance("q", [_verdict()], llm=llm, similarity=FakeSimilarity([sim]), config=config)
        marked = report.verdicts[0]
        assert marked.checks.relevant is expected
        assert marked.checks.relevance_source == "similarity_fallback"
        assert report.judge_failed


def test_reply_with_the_wrong_shape_counts_as_unusable():
    for reply in ('{"claims": []}', '{"answered": "yes"}', "[]"):
        llm = FakeLLM([reply])
        assert ask_judge(llm, "q", ["s"]) is None


def test_unreachable_model_falls_back_to_similarity_and_does_not_raise():
    report = assess_relevance(
        "q", [_verdict()], llm=RaisingLLM(), similarity=FakeSimilarity([0.7]), config=_cfg()
    )
    assert report.judge_failed
    assert report.verdicts[0].checks.relevant is True
    assert report.verdicts[0].checks.relevance_source == "similarity_fallback"


def test_failed_similarity_is_survived_by_asking_the_model():
    report = assess_relevance(
        "q", [_verdict()], llm=_judge(False), similarity=BrokenSimilarity(), config=_cfg()
    )
    assert report.verdicts[0].checks.relevant is False
    assert report.verdicts[0].checks.relevance_source == "judge"
    assert report.notes


def test_no_similarity_and_no_judge_leaves_claims_unchecked_and_flags_it():
    """Failing open, but loudly: the claim stays as verification left it, and
    the report says the gate could not run."""
    report = assess_relevance("q", [_verdict()], llm=RaisingLLM(), similarity=None, config=_cfg())
    assert report.judge_failed
    assert report.verdicts[0].checks.relevant is True
    assert report.verdicts[0].checks.relevance_source == "unchecked"


def test_without_similarity_the_model_alone_decides():
    report = assess_relevance("q", [_verdict()], llm=_judge(False), similarity=None, config=_cfg())
    assert report.verdicts[0].checks.relevant is False
    assert report.verdicts[0].checks.relevance_source == "judge"


# ---------------------------------------------------------------------------
# The similarity scorer
# ---------------------------------------------------------------------------


class FakeEmbedder:
    def encode_query(self, text):
        return [1.0, 0.0]

    def encode_documents(self, texts):
        return [[0.6, 0.8] if "off" in t else [1.0, 0.0] for t in texts]


def test_embedder_similarity_is_the_dot_product_of_normalised_vectors():
    scores = EmbedderSimilarity(FakeEmbedder()).scores("q", ["on topic", "off topic"])
    assert scores == [1.0, 0.6]


def test_embedder_similarity_of_no_claims_is_empty():
    assert EmbedderSimilarity(FakeEmbedder()).scores("q", []) == []


def test_store_without_an_embedder_gives_no_similarity():
    class BareStore:
        pass

    assert similarity_for_store(BareStore()) is None


def test_store_with_an_embedder_gives_a_similarity():
    class Store:
        embedder = FakeEmbedder()

    assert isinstance(similarity_for_store(Store()), EmbedderSimilarity)


# ---------------------------------------------------------------------------
# The cross-encoder judge (relevance.judge = "cross_encoder")
# ---------------------------------------------------------------------------


class FakeCrossEncoder:
    """Returns a fixed probability per claim text."""

    def __init__(self, probs):
        self.probs = probs

    def scores(self, question, claims):
        return [self.probs[c] for c in claims]


class BrokenCrossEncoder:
    def scores(self, question, claims):
        raise RuntimeError("model not downloaded")


def test_cross_encoder_scores_each_claim_on_its_own_and_asks_no_model():
    """Measured: the 3B judge read 'fees will not be refunded' as no answer."""
    answer = _verdict("Fees once paid will not be refunded normally.")
    tangent = _verdict("All visitors must sign the visitor's book.")
    llm = FakeLLM([json.dumps({"answered": False})])
    report = assess_relevance(
        "Are hostel fees refundable?", [answer, tangent], llm=llm,
        similarity=FakeSimilarity([0.6, 0.6]), config=load_config(),
        cross_encoder=FakeCrossEncoder({answer.claim.text: 0.99, tangent.claim.text: 0.01}),
    )
    kept, dropped = report.verdicts
    assert kept.checks.relevant is True and kept.checks.relevance_score == 0.99
    assert kept.checks.relevance_source == "cross_encoder"
    assert dropped.checks.relevant is False
    assert llm.prompts == [], "the language model is not asked"


def test_a_failing_cross_encoder_falls_back_to_the_gate_and_says_so():
    llm = FakeLLM([json.dumps({"answered": True})])
    report = assess_relevance(
        "q", [_verdict()], llm=llm, similarity=FakeSimilarity([0.7]),
        config=load_config(), cross_encoder=BrokenCrossEncoder(),
    )
    assert report.judge_failed
    assert report.verdicts[0].checks.relevance_source == "judge"


def test_judge_views_restate_counts_and_use_the_corpus_definition_of_detained():
    """Measured: the 74-of-100 'detained' question scored 0.000 as asked and
    0.976 in the normalised form."""
    from policyverify.verify.relevance import judge_views

    views = judge_views("i have attended 74 classes out of 100, will i get detained?")
    assert views[0] == "i have attended 74 classes out of 100, will i get detained?"
    assert views[1] == "i have 74% attendance, will i not be allowed to write the examinations?"
    assert judge_views("Are male visitors allowed in female hostels?") == [
        "Are male visitors allowed in female hostels?"]


def test_the_cross_encoder_takes_the_best_score_over_the_views():
    from policyverify.verify.relevance import CrossEncoderRelevance

    class _Model:
        def predict(self, pairs, show_progress_bar=False):
            # High only for the normalised wording.
            return [5.0 if "not be allowed" in q else -5.0 for q, _ in pairs]

    ce = CrossEncoderRelevance("unused")
    ce._model = _Model()
    [p] = ce.scores("i attended 74 classes out of 100, will i get detained?", ["claim"])
    assert p > 0.99


# ---------------------------------------------------------------------------
# Name questions: "which X" / "who is" must be answered by naming something
# ---------------------------------------------------------------------------

_OWN = ["SRM Institute of Science and Technology", "srm"]


def test_naming_only_the_university_does_not_answer_which_software():
    """Measured in the evaluation (un-011): answered from this claim, while
    the documents never name the software."""
    from policyverify.verify.relevance import asks_for_a_name, names_something_new

    q = "Which plagiarism detection software does the university use?"
    assert asks_for_a_name(q)
    assert not names_something_new(
        "SRM Institute of Science and Technology will use the best tools / software "
        "to detect plagiarism.", q, _OWN)
    assert not names_something_new("SRMIST uses the best software available.", q, _OWN)
    assert names_something_new("The university uses Turnitin to detect plagiarism.", q, _OWN)


def test_a_named_scholarship_answers_which_scholarship():
    from policyverify.verify.relevance import names_something_new

    q = "I got rank 300 in SRMJEEE. Which scholarship can I get?"
    assert names_something_new(
        "Candidates with SRMJEEE rank 101-500 can get the SRM Merit Scholarship.", q, _OWN)


def test_who_should_i_contact_is_not_a_name_question():
    from policyverify.verify.relevance import asks_for_a_name

    assert not asks_for_a_name("I fell sick at 2 am. Who should I contact?")
    assert asks_for_a_name("Who is the current Vice Chancellor?")


def test_the_name_check_overrides_a_high_cross_encoder_score():
    claim = _verdict("SRM Institute of Science and Technology will use the best tools / "
                     "software to detect plagiarism.")
    report = assess_relevance(
        "Which plagiarism detection software does the university use?", [claim],
        similarity=FakeSimilarity([0.75]), config=load_config(),
        cross_encoder=FakeCrossEncoder({claim.claim.text: 0.997}), own_names=_OWN,
    )
    marked = report.verdicts[0]
    assert marked.checks.relevant is False
    assert marked.checks.relevance_source == "cross_encoder+name_check"


def test_which_needs_a_name_only_before_a_named_kind_of_thing():
    from policyverify.verify.relevance import asks_for_a_name

    assert asks_for_a_name("Which plagiarism detection software does the university use?")
    assert asks_for_a_name("Which bus routes serve the Kattankulathur campus?")
    # Answered by a description, not a name.
    assert not asks_for_a_name("Which of my published articles can be excluded from the check?")
    assert not asks_for_a_name("I got rank 300. Which scholarship can I get?")
