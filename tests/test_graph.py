"""Tests for the section graph (graph.py)."""

from __future__ import annotations

from policyverify.graph import SectionGraph, build_section_graph
from policyverify.schema import Chunk, PolicyType


def _chunk(citation, text, entities=(), path=None):
    return Chunk(
        chunk_id=f"{citation}#0",
        text=text,
        citation_id=citation,
        university="srm",
        university_name="SRM",
        policy_type=PolicyType(citation.split("/")[1]),
        section_path=path or f"Policy > {citation.rsplit('/', 1)[-1]}",
        source_url="https://example.edu",
        entities=list(entities),
    )


def test_an_explicit_reference_becomes_a_link_both_ways():
    """Real case: 8.2 says 'as indicated in the table under section R8.1'."""
    chunks = [
        _chunk("srm/attendance/8.1-grading", "The letter grades are as follows."),
        _chunk("srm/attendance/8.2-awarding", "Grades as indicated in the table under section R8.1."),
    ]
    g = build_section_graph(chunks, {})
    assert g.related("srm/attendance/8.2-awarding")[0]["citation"] == "srm/attendance/8.1-grading"
    assert g.related("srm/attendance/8.1-grading")[0]["kind"] == "refers_to"


def test_a_reference_to_a_section_that_does_not_exist_is_not_a_link():
    chunks = [_chunk("srm/attendance/3.5-summer", "R 3.6 During the Break of Studies ...")]
    assert build_section_graph(chunks, {}).edges == {}


def test_distinctive_entities_link_sections_and_common_ones_do_not():
    weights = {"parent": 0.9, "student": 0.05}
    chunks = [
        _chunk("srm/residence/guest", "Parent stays.", ["parent", "student"]),
        _chunk("srm/residence/visitors", "Visitors.", ["parent", "student"]),
        _chunk("srm/attendance/7.3-min", "Attendance.", ["student"]),
    ]
    g = build_section_graph(chunks, weights)
    related = g.related("srm/residence/guest")
    assert [e["citation"] for e in related] == ["srm/residence/visitors"]
    assert related[0]["why"] == "both concern parent"
    assert g.related("srm/attendance/7.3-min") == [], "'student' is too common to link anything"


def test_the_graph_round_trips_through_disk(tmp_path):
    chunks = [
        _chunk("srm/attendance/8.1-grading", "Grades."),
        _chunk("srm/attendance/8.2-awarding", "See section R8.1."),
    ]
    build_section_graph(chunks, {}).save(tmp_path)
    loaded = SectionGraph.load(tmp_path)
    assert loaded.related("srm/attendance/8.2-awarding")[0]["citation"] == "srm/attendance/8.1-grading"
    assert SectionGraph.load(tmp_path / "missing") is None
