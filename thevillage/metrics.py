from __future__ import annotations
import itertools
import random
from collections.abc import Sequence


def gini(xs: Sequence[float]) -> float:
    xs = sorted(float(x) for x in xs)
    n, total = len(xs), sum(xs)
    if n == 0 or total == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * total) - (n + 1) / n

def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def bootstrap_ci(xs: Sequence[float], n: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    ms = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    return ms[int(alpha / 2 * n)], ms[int((1 - alpha / 2) * n) - 1]

def paired_sign_flip_p(a: Sequence[float], b: Sequence[float]) -> float:
    d = [x - y for x, y in zip(a, b, strict=True)]
    obs = abs(sum(d))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(d)):
        total += 1
        hits += abs(sum(s * x for s, x in zip(signs, d, strict=True))) >= obs - 1e-12
    return hits / total
