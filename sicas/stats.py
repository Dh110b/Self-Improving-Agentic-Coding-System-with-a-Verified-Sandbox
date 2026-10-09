"""Small stats helpers (stdlib only): mean/std and percentile bootstrap CIs."""
from __future__ import annotations

import math
import random


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def bootstrap_ci(xs: list[float], n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Resample the *units* (e.g. tasks)."""
    if not xs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(xs)
    means = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def paired_bootstrap_diff(a: list[float], b: list[float], n_boot: int = 2000,
                          seed: int = 0) -> tuple[float, float, float]:
    """CI of mean(a - b) for paired per-unit scores. Returns (mean_diff, lo, hi)."""
    assert len(a) == len(b)
    d = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(d, n_boot=n_boot, seed=seed)
    return (sum(d) / len(d) if d else 0.0), lo, hi
