"""
Tests for the keyword index (indexing/bm25.py) and the rank-fusion rule
(retrieve.reciprocal_rank_fusion).

Pure functions over a handful of hand-written chunks: no model, no index, no
network. Whether BM25 actually helps on the real corpus is measured against
the evaluation questions, not asserted here; these tests pin down that the
arithmetic does what the docstrings say.
"""

from __future__ import annotations

from policyverify.indexing.bm25 import STOP_WORDS, BM25Index, tokenize
from policyverify.retrieve import reciprocal_rank_fusion
from policyverify.schema import Chunk, PolicyType


def _chunk(name: str, text: str, university="srm", policy=PolicyType.ATTENDANCE,
           heading: str | None = None) -> Chunk:
    return Chunk(
        chunk_id=f"{university}/{policy.value}/{name}#0",
        text=text,
        citation_id=f"{university}/{policy.value}/{name}",
        university=university,
        university_name="Some University",
        policy_type=policy,
        section_path=heading or f"Policy > {name}",
        source_url="https://example.edu/p",
    )


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("To 'Pass' a course: 50%!") == ["pass", "course", "50"]


def test_tokenize_splits_hyphenated_words():
    assert tokenize("re-examination") == ["re", "examination"]


def test_tokenize_drops_stop_words_but_keeps_meaningful_short_words():
    words = tokenize("What is the minimum attendance? Not less than 75")
    assert "what" not in words and "the" not in words
    # "not" and "minimum" change the meaning of a rule; they must survive.
    assert "not" in words and "minimum" in words
    assert "not" not in STOP_WORDS


def test_tokenize_empty_text():
    assert tokenize("") == []
    assert tokenize("the of and") == []


# ---------------------------------------------------------------------------
# BM25 scoring
# ---------------------------------------------------------------------------


def _corpus() -> list[Chunk]:
    return [
        _chunk("attendance", "Students must attend 75% of the classes in each course."),
        _chunk("grades", "To pass a course a student must secure a minimum of 50% marks."),
        _chunk("hostel", "Students in the hostel must return before the curfew."),
        _chunk("condonation", "Condonation of attendance shortage is up to 10% on medical grounds."),
    ]


def test_rare_keyword_ranks_its_chunk_first():
    index = BM25Index(_corpus())
    hits = index.search("what is condonation?", k=4)
    assert hits[0][0].chunk_id == "srm/attendance/condonation#0"


def test_the_marks_question_finds_the_pass_rule():
    """The measured failure that motivated hybrid search: the embedding
    search ranked this section about 10th."""
    index = BM25Index(_corpus())
    hits = index.search("What is minimum marks needed to pass a course ?", k=4)
    assert hits[0][0].chunk_id == "srm/attendance/grades#0"


def test_stop_words_do_not_dominate():
    """A chunk stuffed with 'the', 'what', 'is' must not outrank the one that
    contains the actual subject of the question."""
    noisy = _chunk("noisy", "what is the the the what is is it this that of to in on")
    index = BM25Index([*_corpus(), noisy])
    hits = index.search("what is the curfew", k=5)
    assert hits[0][0].chunk_id == "srm/attendance/hostel#0"
    assert all(c.chunk_id != noisy.chunk_id for c, _ in hits)


def test_headings_are_searchable():
    chunk = _chunk("letter", "O 91-100, A+ 81-90.", heading="Method of Awarding Letter Grades")
    index = BM25Index([*_corpus(), chunk])
    hits = index.search("letter grades", k=1)
    assert hits[0][0].chunk_id == chunk.chunk_id


def test_repeating_a_word_saturates():
    """k1 makes the tenth mention worth much less than the first."""
    once = _chunk("once", "fee refund")
    many = _chunk("many", "fee " * 10 + "refund")
    index = BM25Index([once, many, *_corpus()])
    scores = dict((c.chunk_id, s) for c, s in index.search("fee", k=10))
    assert scores[many.chunk_id] < 3 * scores[once.chunk_id]


def test_query_with_no_known_words_returns_nothing():
    index = BM25Index(_corpus())
    assert index.search("zebra xylophone", k=5) == []


def test_empty_index_is_harmless():
    index = BM25Index([])
    assert len(index) == 0
    assert index.search("anything", k=5) == []


def test_ties_are_broken_by_chunk_id():
    a = _chunk("b_second", "library hours", heading="Library")
    b = _chunk("a_first", "library hours", heading="Library")
    index = BM25Index([a, b])
    hits = index.search("library", k=2)
    assert hits[0][1] == hits[1][1]
    assert [c.chunk_id for c, _ in hits] == [b.chunk_id, a.chunk_id]


def test_k_limits_the_result():
    index = BM25Index(_corpus())
    assert len(index.search("students", k=1)) == 1


def test_filters_restrict_but_do_not_rescore():
    """University and policy filters mean what they mean on the embedding
    side: they decide which chunks are eligible, not how they score."""
    other_uni = _chunk("attendance", "Students must attend 75% of classes.", university="uni_b")
    other_policy = _chunk("rules", "Students must attend 75% of classes.",
                          policy=PolicyType.EXAMINATION)
    index = BM25Index([*_corpus(), other_uni, other_policy])

    unfiltered = dict((c.chunk_id, s) for c, s in index.search("attend classes", k=10))

    by_uni = index.search("attend classes", k=10, university="uni_b")
    assert [c.chunk_id for c, _ in by_uni] == [other_uni.chunk_id]
    assert by_uni[0][1] == unfiltered[other_uni.chunk_id]

    by_policy = index.search("attend classes", k=10, policy_type=PolicyType.EXAMINATION)
    assert [c.chunk_id for c, _ in by_policy] == [other_policy.chunk_id]
    # a plain string works too, as it does for VectorStore.search
    assert index.search("attend classes", k=10, policy_type="examination") == by_policy

    both = index.search("attend classes", k=10, university="srm", policy_type="attendance")
    assert all(c.university == "srm" and c.policy_type is PolicyType.ATTENDANCE for c, _ in both)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def test_agreement_beats_a_single_first_place():
    """Ranked 2nd by both searches beats ranked 1st by only one of them."""
    dense = ["solo_dense", "both", "x"]
    keyword = ["solo_kw", "both", "y"]
    fused = reciprocal_rank_fusion([dense, keyword], rrf_k=60)
    assert fused[0] == "both"


def test_fusion_includes_items_from_either_list():
    fused = reciprocal_rank_fusion([["a"], ["b"]])
    assert set(fused) == {"a", "b"}


def test_fusion_ties_are_deterministic():
    """'a' and 'b' each come first in one list: equal sums. The order must
    not depend on dict ordering or on which list is passed first."""
    one = reciprocal_rank_fusion([["b", "a"], ["a", "b"]])
    two = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])
    assert one == two == ["a", "b"]


def test_fusion_tie_between_mirrored_items_falls_back_to_id():
    """x and y tie exactly (each is 1st in one list and 3rd in the other), so
    the id decides between them. z, 2nd in both, scores 2/62 - a hair below
    1/61 + 1/63 - so it comes last."""
    fused = reciprocal_rank_fusion([["x", "z", "y"], ["y", "z", "x"]])
    assert fused == ["x", "y", "z"]


def test_fusion_tie_prefers_the_better_single_rank():
    """With rrf_k=0 an exact tie is easy to build: 'zz' is 1st in one list
    (1/1) and 'aa' is 2nd in both (1/2 + 1/2). The one that reached first
    place wins, even though its id sorts later."""
    fused = reciprocal_rank_fusion([["zz", "aa"], ["other", "aa"]], rrf_k=0)
    assert fused.index("zz") < fused.index("aa")


def test_duplicate_in_one_list_counts_once():
    fused = reciprocal_rank_fusion([["a", "a", "b"], ["b"]])
    assert fused[0] == "b"


def test_fusion_of_nothing():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
