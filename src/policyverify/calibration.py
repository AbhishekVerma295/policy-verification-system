"""
calibration.py - is the verifier's confidence a probability, and where
should the pass marks sit?

Two different questions, both answered from the same list of
(score, truly supported?) pairs collected on the TUNING half of
eval/claims.jsonl (see scripts/calibrate.py):

CALIBRATION: DOES "0.9" MEAN "RIGHT 9 TIMES IN 10"?
    The NLI model's entailment probability is a softmax output. Nothing
    guarantees that claims scored 0.9 are true 90% of the time; modern
    classifiers are usually over-confident (Guo et al., ICML 2017). Two
    measures are reported:

      ECE     expected calibration error. Claims are put into a few equal-
              width score bins; in each bin the gap between the mean score
              and the share of claims that are really supported is taken,
              weighted by how many claims fell in the bin. 0 = perfect.
              With ~40 claims only a handful of bins is meaningful, and the
              bin counts are always printed, because a bin of 2 claims says
              almost nothing.
      Brier   mean squared difference between score and outcome (1 or 0).
              Needs no bins, so it is the steadier number on little data.

TEMPERATURE SCALING, AND WHY IT CANNOT CHANGE WHICH CLAIMS ARE KEPT
    Temperature scaling divides the score's log-odds by one number T:

        calibrated = sigmoid(logit(score) / T)

    T > 1 pulls confident scores towards 0.5, T < 1 pushes them out. It is
    the simplest calibration method there is (one parameter), which is the
    only kind worth fitting on ~40 claims. It is also MONOTONE: it never
    reorders two claims, so any threshold on the raw score is exactly the
    same as some threshold on the calibrated score. Applying it to the
    confidence a user sees therefore cannot change which claims are kept -
    only the number printed next to them.

    T is fitted by minimising the negative log-likelihood on the same claims
    it is measured on. That is optimistic: the "after" numbers are the best
    this one parameter can do on these claims, not what it would do on new
    ones. scripts/calibrate.py says so whenever it prints them.

    This is a 2-class (supported or not) temperature on the claim-level
    score, not the textbook 3-class temperature on the NLI logits: the
    verifier keeps only probabilities, and its claim score is a maximum over
    many sentences, so a per-sentence temperature would also change WHICH
    sentence wins. The 2-class form is the one that can be applied to the
    number the user sees.

RISK-COVERAGE: WHAT DOES A STRICTER PASS MARK BUY?
    For each candidate threshold: how many claims would be accepted
    (coverage) and what share of the accepted ones are actually false
    (risk). Raising the threshold should lower risk and lower coverage; the
    question is whether the risk drops enough to be worth the true claims
    thrown away. Wrongly rejected true claims show up end to end as false
    abstentions, so coverage is not free to give up.

Nothing here loads a model or reads a file. Every function takes plain
lists, so tests run in milliseconds.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Scores of exactly 0 or 1 (the exact-quote shortcut returns 1.0) have
# infinite log-odds. They are clipped this far from the ends first.
_EPS = 1e-6


@dataclass
class Bin:
    """One score bin of a calibration table."""

    low: float
    high: float
    count: int
    mean_score: float | None
    share_supported: float | None

    @property
    def gap(self) -> float | None:
        if self.count == 0 or self.mean_score is None or self.share_supported is None:
            return None
        return abs(self.mean_score - self.share_supported)


@dataclass
class RiskRow:
    """One line of a risk-coverage table."""

    threshold: float
    accepted: int
    wrong: int
    total: int
    supported_total: int

    @property
    def coverage(self) -> float:
        """Share of ALL claims that were accepted."""
        return self.accepted / self.total if self.total else 0.0

    @property
    def risk(self) -> float | None:
        """Share of ACCEPTED claims that are false. None if nothing accepted."""
        return self.wrong / self.accepted if self.accepted else None

    @property
    def true_kept(self) -> int:
        return self.accepted - self.wrong

    @property
    def recall(self) -> float | None:
        """Share of truly supported claims that were accepted."""
        return self.true_kept / self.supported_total if self.supported_total else None


def _check(scores: Sequence[float], labels: Sequence[bool]) -> None:
    if len(scores) != len(labels):
        raise ValueError(f"{len(scores)} scores but {len(labels)} labels")
    for s in scores:
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"score {s} is not a probability")


def brier(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Mean squared error between score and outcome. Lower is better."""
    _check(scores, labels)
    if not scores:
        raise ValueError("no scores")
    return sum((s - float(y)) ** 2 for s, y in zip(scores, labels, strict=True)) / len(scores)


def calibration_bins(
    scores: Sequence[float], labels: Sequence[bool], n_bins: int = 5
) -> list[Bin]:
    """Equal-width bins over [0, 1]. The last bin includes 1.0."""
    _check(scores, labels)
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    members: list[list[int]] = [[] for _ in range(n_bins)]
    for i, s in enumerate(scores):
        members[min(int(s * n_bins), n_bins - 1)].append(i)
    out = []
    for b, idx in enumerate(members):
        mean = sum(scores[i] for i in idx) / len(idx) if idx else None
        share = sum(1 for i in idx if labels[i]) / len(idx) if idx else None
        out.append(Bin(b / n_bins, (b + 1) / n_bins, len(idx), mean, share))
    return out


def ece(scores: Sequence[float], labels: Sequence[bool], n_bins: int = 5) -> float:
    """Expected calibration error with `n_bins` equal-width bins."""
    bins = calibration_bins(scores, labels, n_bins)
    n = len(scores)
    if not n:
        raise ValueError("no scores")
    return sum(b.count / n * b.gap for b in bins if b.gap is not None)


def apply_temperature(score: float, temperature: float) -> float:
    """sigmoid(logit(score) / T). T = 1 returns the score unchanged.

    Monotone in `score` for any T > 0, so it never reorders claims.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return score
    p = min(max(score, _EPS), 1.0 - _EPS)
    z = math.log(p / (1.0 - p)) / temperature
    # Numerically safe sigmoid.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def nll(scores: Sequence[float], labels: Sequence[bool], temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the labels under the scaled scores."""
    _check(scores, labels)
    total = 0.0
    for s, y in zip(scores, labels, strict=True):
        p = min(max(apply_temperature(s, temperature), _EPS), 1.0 - _EPS)
        total -= math.log(p if y else 1.0 - p)
    return total / len(scores)


def fit_temperature(
    scores: Sequence[float],
    labels: Sequence[bool],
    low: float = 0.05,
    high: float = 20.0,
    steps: int = 400,
) -> float:
    """The T in [low, high] with the lowest NLL on these scores.

    A plain search over a log-spaced grid, then a finer grid around the best
    point. One parameter on a bounded range needs nothing cleverer, and it
    keeps this dependency-free. If every claim has the same label the NLL
    keeps falling towards an end of the range, and that end is returned.
    """
    _check(scores, labels)
    if not scores:
        raise ValueError("no scores")

    def search(lo: float, hi: float) -> float:
        a, b = math.log(lo), math.log(hi)
        grid = [math.exp(a + (b - a) * k / steps) for k in range(steps + 1)]
        return min(grid, key=lambda t: (nll(scores, labels, t), abs(math.log(t))))

    coarse = search(low, high)
    ratio = (high / low) ** (2.0 / steps)
    return search(max(low, coarse / ratio), min(high, coarse * ratio))


def risk_coverage(
    thresholds: Sequence[float],
    labels: Sequence[bool],
    accepted_at: Callable[[float], Sequence[bool]],
) -> list[RiskRow]:
    """One row per threshold.

    `accepted_at(t)` returns, for every claim in `labels` order, whether it
    would be accepted at threshold t. A callable rather than a score list,
    because accepting a claim is not always "score >= t": the verifier also
    applies the numeric veto and the conflict rule, and the script re-runs
    it at each threshold.
    """
    rows = []
    supported_total = sum(1 for y in labels if y)
    for t in thresholds:
        acc = list(accepted_at(t))
        if len(acc) != len(labels):
            raise ValueError("accepted_at returned the wrong number of flags")
        accepted = sum(acc)
        wrong = sum(1 for a, y in zip(acc, labels, strict=True) if a and not y)
        rows.append(RiskRow(t, accepted, wrong, len(labels), supported_total))
    return rows


def format_bins(bins: list[Bin]) -> str:
    """A calibration table for the terminal, with every bin's count."""
    lines = [f"  {'bin':<11} {'claims':>6} {'mean score':>11} {'really supported':>17} {'gap':>6}"]
    for b in bins:
        mean = "-" if b.mean_score is None else f"{b.mean_score:.3f}"
        share = "-" if b.share_supported is None else f"{b.share_supported:.3f}"
        gap = "-" if b.gap is None else f"{b.gap:.3f}"
        lines.append(f"  {b.low:.1f}-{b.high:.1f}     {b.count:>6} {mean:>11} {share:>17} {gap:>6}")
    return "\n".join(lines)


def format_risk_coverage(rows: list[RiskRow], name: str = "threshold") -> str:
    """A risk-coverage table for the terminal."""
    lines = [
        f"  {name:<10} {'accepted':>8} {'wrong':>6} {'risk':>7} {'coverage':>9} "
        f"{'true kept':>10} {'recall':>7}"
    ]
    for r in rows:
        risk = "   n/a" if r.risk is None else f"{r.risk * 100:5.1f}%"
        recall = "   n/a" if r.recall is None else f"{r.recall * 100:5.1f}%"
        lines.append(
            f"  {r.threshold:<10.3f} {r.accepted:>8} {r.wrong:>6} {risk:>7} "
            f"{r.coverage * 100:8.1f}% {r.true_kept:>4}/{r.supported_total:<5} {recall:>7}"
        )
    return "\n".join(lines)
