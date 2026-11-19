"""
Tests for calibration.py: calibration error, temperature scaling and
risk-coverage tables.

Pure arithmetic on hand-made lists, so no model, index or network is needed.
The property that matters most for the system is that temperature scaling is
MONOTONE: applied to the confidence a user sees, it must never change which
claims pass a threshold, only the number shown.
"""

from __future__ import annotations

import pytest

from policyverify.calibration import (
    apply_temperature,
    brier,
    calibration_bins,
    ece,
    fit_temperature,
    format_bins,
    format_risk_coverage,
    nll,
    risk_coverage,
)

# ---------------------------------------------------------------------------
# Brier and ECE
# ---------------------------------------------------------------------------


def test_brier_is_zero_for_perfect_scores_and_one_for_perfectly_wrong():
    assert brier([1.0, 0.0], [True, False]) == 0.0
    assert brier([0.0, 1.0], [True, False]) == 1.0


def test_brier_of_a_coin_flip_is_a_quarter():
    assert brier([0.5, 0.5], [True, False]) == pytest.approx(0.25)


def test_ece_is_zero_when_every_bin_matches_its_accuracy():
    # Bin 0.6-0.8: mean score 0.75, 3 of 4 supported.
    scores = [0.75, 0.75, 0.75, 0.75]
    labels = [True, True, True, False]
    assert ece(scores, labels, n_bins=5) == pytest.approx(0.0)


def test_ece_weights_each_bin_by_its_count():
    # One bin of 3 at 0.9 that is all false (gap 0.9), one bin of 1 at 0.1
    # that is false (gap 0.1): (3 * 0.9 + 1 * 0.1) / 4 = 0.7.
    scores = [0.9, 0.9, 0.9, 0.1]
    labels = [False, False, False, False]
    assert ece(scores, labels, n_bins=5) == pytest.approx(0.7)


def test_bins_report_counts_and_put_one_in_the_last_bin():
    bins = calibration_bins([0.0, 0.19, 0.5, 1.0], [False, False, True, True], n_bins=5)
    assert [b.count for b in bins] == [2, 0, 1, 0, 1]
    assert bins[1].mean_score is None and bins[1].gap is None
    assert bins[4].share_supported == 1.0


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        brier([0.5], [True, False])
    with pytest.raises(ValueError):
        ece([1.5], [True])
    with pytest.raises(ValueError):
        brier([], [])


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def test_temperature_one_changes_nothing():
    for s in (0.0, 0.2, 0.5, 0.97, 1.0):
        assert apply_temperature(s, 1.0) == s


def test_higher_temperature_pulls_towards_one_half():
    assert 0.5 < apply_temperature(0.9, 2.0) < 0.9
    assert 0.1 < apply_temperature(0.1, 2.0) < 0.5
    assert apply_temperature(0.5, 3.0) == pytest.approx(0.5)


def test_lower_temperature_pushes_away_from_one_half():
    assert apply_temperature(0.9, 0.5) > 0.9
    assert apply_temperature(0.1, 0.5) < 0.1


def test_temperature_never_reorders_scores():
    """So a threshold on the raw score always equals some threshold on the
    scaled one: scaling the shown confidence cannot change what is kept."""
    scores = [0.0, 0.001, 0.2, 0.49, 0.5, 0.51, 0.8, 0.999, 1.0]
    for t in (0.3, 0.77, 1.5, 4.0):
        scaled = [apply_temperature(s, t) for s in scores]
        assert scaled == sorted(scaled)


def test_temperature_must_be_positive():
    with pytest.raises(ValueError):
        apply_temperature(0.5, 0.0)


def test_fit_softens_over_confident_scores():
    # Scores of 0.95 / 0.05 that are right only 70% of the time: the best
    # temperature is above 1, and it lowers the NLL.
    scores = [0.95] * 10 + [0.05] * 10
    labels = [True] * 7 + [False] * 3 + [False] * 7 + [True] * 3
    t = fit_temperature(scores, labels)
    assert t > 1.0
    assert nll(scores, labels, t) < nll(scores, labels, 1.0)
    # sigmoid(logit(0.95) / t) should land near the true 0.7.
    assert apply_temperature(0.95, t) == pytest.approx(0.7, abs=0.02)


def test_fit_sharpens_under_confident_scores():
    scores = [0.7] * 10 + [0.3] * 10
    labels = [True] * 10 + [False] * 10
    assert fit_temperature(scores, labels) < 1.0


# ---------------------------------------------------------------------------
# Risk-coverage
# ---------------------------------------------------------------------------


def test_risk_coverage_counts_accepted_and_wrong():
    scores = [0.99, 0.9, 0.6, 0.3]
    labels = [True, True, False, True]
    rows = risk_coverage([0.5, 0.95], labels, lambda t: [s >= t for s in scores])
    first, second = rows
    assert (first.accepted, first.wrong, first.true_kept) == (3, 1, 2)
    assert first.risk == pytest.approx(1 / 3)
    assert first.coverage == pytest.approx(0.75)
    assert first.recall == pytest.approx(2 / 3)
    assert (second.accepted, second.wrong) == (1, 0)
    assert second.risk == 0.0


def test_risk_is_none_when_nothing_is_accepted():
    rows = risk_coverage([0.5], [True, False], lambda t: [False, False])
    assert rows[0].risk is None
    assert rows[0].coverage == 0.0


def test_risk_coverage_rejects_a_wrong_length():
    with pytest.raises(ValueError):
        risk_coverage([0.5], [True, False], lambda t: [True])


def test_formatting_mentions_every_row_and_bin():
    bins_text = format_bins(calibration_bins([0.1, 0.9], [False, True], n_bins=2))
    assert "0.0-0.5" in bins_text and "0.5-1.0" in bins_text
    table = format_risk_coverage(
        risk_coverage([0.5, 0.9], [True], lambda t: [True]), "entail_t"
    )
    assert "entail_t" in table
    assert "0.500" in table and "0.900" in table
