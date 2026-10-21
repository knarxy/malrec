"""How sure a prediction is.

A prediction is an expected rating; what the user sees next to it is the
range their actual rating is likely to fall in, and the chance it is a 9 or
10. Both come from the distribution of past prediction errors - split-
conformal style, so no distributional assumption:

  * accounts with their own temporal holdout: their own errors on their
    newest ratings (model.train_hybrid)
  * small accounts: errors of held-out population users with a list the same
    size (service.fit_size_calibration), top-of-list errors where available,
    since list cards show the top of the list

Errors are stored as 21 quantiles (0, 5, ..., 100 %). What a card shows
("most likely 7-8", or one number) is the shortest run of whole ratings
around the likeliest one that holds half of the error distribution: out of
time it held 56-66 % of later ratings for both sources, where the 5-95 %
band shown until 2026-10 ("likely 5-8") read as "you might rate it a 5"
(experiments/exp_range.py, results_range_2026-10-20.txt). The bands stay for
the internal checks (`coverage`, `malrec report`, `coverage80` in each
size-calibration row).
"""
from __future__ import annotations

import numpy as np

QS = np.arange(0, 101, 5) / 100.0
# The internal "8 in 10" bands. Own holdout: the 5-95 % band, nominally 90 %,
# but errors grow as taste drifts - strictly out of time it holds ~80 % of
# later ratings (2026-10-20). Population-based: the 10-90 % band (76-83 %).
LOW, HIGH = 1, 19                  # indices of the 5 % and 95 % quantiles
POP_BAND = (2, 18)


def residual_quantiles(pred, truth) -> list[float]:
    e = np.asarray(truth, dtype=float) - np.asarray(pred, dtype=float)
    return [round(float(x), 3) for x in np.quantile(e, QS)]


LIKELY_MASS = 0.5


def likely_set(pred: float, qa: np.ndarray, mass: float = LIKELY_MASS) -> tuple[int, int]:
    """The shortest run of whole ratings around the likeliest one that holds
    `mass` of the error distribution (ratings below 1 count as 1, above 10 as 10)."""
    edges = [float(np.interp(k + 0.5 - pred, qa, QS, left=0.0, right=1.0)) for k in range(11)]
    probs = np.diff(edges)
    probs[0] += edges[0]
    probs[-1] += 1.0 - edges[-1]
    lo = hi = int(np.argmax(probs))
    cum = probs[lo]
    while cum < mass and (lo > 0 or hi < 9):
        left = probs[lo - 1] if lo > 0 else -1.0
        right = probs[hi + 1] if hi < 9 else -1.0
        if right >= left:
            hi += 1
            cum += probs[hi]
        else:
            lo -= 1
            cum += probs[lo]
    return lo + 1, hi + 1


def band(pred: float | None, q: list[float] | None) -> dict | None:
    """{'low', 'high', 'p9'} for a displayed prediction, or None: the likely
    set and the chance of a 9 or 10."""
    if q is None or pred is None or not np.isfinite(pred):
        return None
    qa = np.asarray(q, dtype=float)
    lo, hi = likely_set(pred, qa)
    # P(rating >= 9) = P(error >= 8.5 - pred), from the error quantile function
    cdf = float(np.interp(8.5 - pred, qa, QS, left=0.0, right=1.0))
    return {"low": lo, "high": hi, "p9": round(1.0 - cdf, 2)}


def likely_coverage(q: list[float], pred, truth) -> float:
    """Share of actual ratings inside the shown likely set - above one half."""
    qa = np.asarray(q, dtype=float)
    sets = [likely_set(float(x), qa) for x in np.asarray(pred, dtype=float)]
    t = np.asarray(truth, dtype=float)
    return float(np.mean([lo <= y <= hi for (lo, hi), y in zip(sets, t, strict=True)]))


def coverage(q: list[float], pred, truth, idx: tuple[int, int] | None = None) -> float:
    """Share of actual ratings inside the 80 % band - should be about 0.8."""
    qa = np.asarray(q, dtype=float)
    p, t = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    ends = [_ends(x, qa, idx) for x in p]
    lo = np.array([e[0] for e in ends])
    hi = np.array([e[1] for e in ends])
    return float(np.mean((t >= lo) & (t <= hi)))


def _ends(pred: float, qa: np.ndarray, idx: tuple[int, int] | None = None) -> tuple[int, int]:
    """The whole ratings inside the band. Rounding both ends outward
    would make an "80 %" range cover ~92 % of integer ratings."""
    a, b = idx or (LOW, HIGH)
    lo = int(np.clip(np.ceil(pred + qa[a] - 1e-9), 1, 10))
    hi = int(np.clip(np.floor(pred + qa[b] + 1e-9), 1, 10))
    if hi < lo:                        # band narrower than one step
        lo = hi = int(np.clip(round(pred), 1, 10))
    return lo, hi


def user_quantiles(user_id: int) -> list[float] | None:
    """The error quantiles stored with the user's latest model."""
    from .db import one
    row = one("SELECT metrics->'uncertainty' AS u FROM model_run WHERE user_id=%s"
              " ORDER BY id DESC LIMIT 1", (user_id,))
    return ((row or {}).get("u") or {}).get("q")
