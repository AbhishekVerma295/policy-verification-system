"""Tests for the written answer (compose.py): every sentence must be backed
by the verified facts, and an unsupported sentence never reaches the user."""

from __future__ import annotations

from policyverify.compose import compose_answer
from policyverify.llm import LLMError
from policyverify.schema import Claim, ClaimVerdict, VerdictStatus


def _kept(*texts):
    return [
        ClaimVerdict(claim=Claim(text=t, citation_ids=["srm/residence/refund"]),
                     status=VerdictStatus.SUPPORTED, score=1.0)
        for t in texts
    ]


class _LLM:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def generate(self, prompt, *, json_mode=True, schema=None):
        assert json_mode is False, "the answer is prose"
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class _NLI:
    """Entails a sentence only when every word in it appears in the premise."""

    class _R:
        def __init__(self, e):
            self.entailment = e

    def check(self, premise, hypothesis):
        words = {w.strip(".,").lower() for w in hypothesis.split()}
        known = {w.strip(".,").lower() for w in premise.split()} | {"no", "yes"}
        return self._R(1.0 if words <= known else 0.0)


FACTS = _kept("Fees once paid will not be refunded normally.",
              "Refund is permitted if the student is selected for semester abroad.")


def test_a_supported_answer_is_kept_with_its_fact_markers():
    llm = _LLM("No, fees once paid will not be refunded normally. [1] "
               "Refund is permitted if the student is selected for semester abroad. [2]")
    out = compose_answer("Are hostel fees refundable?", FACTS, llm, _NLI())
    assert out.text.startswith("No, fees once paid will not be refunded normally.")
    assert "[2]" in out.text and out.dropped_sentences == []
    # The model is shown the verified facts - not the passages.
    assert "[1] Fees once paid will not be refunded normally." in llm.prompts[0]


def test_an_invented_sentence_is_removed():
    llm = _LLM("No, fees will not be refunded normally. [1] "
               "You can also ask the bank for a loan to cover the fees. [1]")
    out = compose_answer("Are hostel fees refundable?", FACTS, llm, _NLI())
    assert "bank" not in out.text
    assert out.dropped_sentences == ["You can also ask the bank for a loan to cover the fees."]


def test_nothing_supported_means_no_written_answer():
    llm = _LLM("Hostel fees are always refundable within thirty days. [1]")
    assert compose_answer("Are hostel fees refundable?", FACTS, llm, _NLI()) is None


def test_no_model_or_no_facts_means_no_written_answer():
    assert compose_answer("q", FACTS, _LLM(LLMError("down")), _NLI()) is None
    assert compose_answer("q", [], _LLM("anything"), _NLI()) is None


def test_an_answer_never_opens_on_a_dangling_connective():
    """Measured: the first sentence was removed and the answer began 'However, ...'."""
    llm = _LLM("Yes, you are fine. [9] However, fees once paid will not be refunded normally. [1]")
    out = compose_answer("Are hostel fees refundable?", FACTS, llm, _NLI())
    assert out.text.startswith("Fees once paid will not be refunded normally.")


def test_the_model_never_sees_the_askers_attendance_figures():
    llm = _LLM("Fees once paid will not be refunded normally. [1]")
    compose_answer("I attended 74 classes out of 100, will I be detained?", FACTS, llm, _NLI())
    assert "74" not in llm.prompts[0]
    assert "a certain number of classes out of the total" in llm.prompts[0]
