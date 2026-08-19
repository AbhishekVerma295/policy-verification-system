"""
Shared test fixtures.

The important one here is FakeLLM. Real language models are slow, need a GPU,
and give slightly different answers each time - all three of which make them
useless inside a test suite. FakeLLM returns canned answers instantly, so the
tests run anywhere in under a second, including on a machine with no GPU.

The rule this encodes: tests check that the *plumbing* is correct. Whether the
answers are any good is a question for the evaluation harness in Phase 8, and
it is measured on the adversarial set, not asserted in a unit test.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from policyverify.schema import (
    Chunk,
    Document,
    PolicyType,
    Section,
    SourceFormat,
)


class FakeLLM:
    """A stand-in for the real model that returns whatever you tell it to.

    Used from Phase 3 onwards, wherever a test needs something shaped like a
    model response without paying for an actual model.

        llm = FakeLLM(['{"claims": []}'])
        llm.generate("any prompt")   -> '{"claims": []}'

    It also records every prompt it was given, so a test can assert that the
    retrieved passages actually made it into the prompt.
    """

    def __init__(self, responses: list[str] | None = None):
        self.responses = list(responses or [])
        self.prompts: list[str] = []
        self._index = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            return json.dumps({"claims": []})
        # Repeat the last response once the scripted ones run out, so a test
        # does not fall over just because it called one extra time.
        response = self.responses[min(self._index, len(self.responses) - 1)]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        return len(self.prompts)


@pytest.fixture
def fake_llm() -> FakeLLM:
    """A FakeLLM that returns one supported claim."""
    return FakeLLM(
        [
            json.dumps(
                {
                    "claims": [
                        {
                            "text": "Students must attend at least 75% of classes.",
                            "citation_ids": ["uni_a/attendance/4.2"],
                        }
                    ]
                }
            )
        ]
    )


@pytest.fixture
def sample_section() -> Section:
    return Section(
        number="4.2",
        heading="Minimum Attendance",
        text=(
            "Students must attend at least 75% of all scheduled classes in "
            "each course to be eligible to sit the final examination."
        ),
        path=["Attendance Policy", "Requirements", "Minimum Attendance"],
        level=3,
    )


@pytest.fixture
def sample_document(sample_section: Section) -> Document:
    return Document(
        doc_id="uni_a/attendance",
        university="uni_a",
        university_name="Example University A",
        policy_type=PolicyType.ATTENDANCE,
        title="Attendance and Absence Policy",
        source_url="https://example-a.edu/policies/attendance",
        source_format=SourceFormat.HTML,
        retrieved_at=date(2026, 8, 14),
        checksum="a" * 64,
        sections=[sample_section],
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="uni_a/attendance/4.2#0",
        text=(
            "Students must attend at least 75% of all scheduled classes in "
            "each course to be eligible to sit the final examination."
        ),
        citation_id="uni_a/attendance/4.2",
        university="uni_a",
        university_name="Example University A",
        policy_type=PolicyType.ATTENDANCE,
        section_path="Attendance Policy > Requirements > Minimum Attendance",
        source_url="https://example-a.edu/policies/attendance",
    )
