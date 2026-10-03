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

Errors are stored as 21 quantiles (0, 5, ..., 100 %). The "likely" range is
the 5-95 % band, chosen so that its real, out-of-sample coverage is about
80 % (`malrec report`, and `coverage80` in each size-calibration row).
"""
from __future__ import annotations

import numpy as np

QS = np.arange(0, 101, 5) / 100.0
# The shown range is the 5-95 % band of past errors. Nominally 90 %, but
# errors grow as taste drifts, so measured strictly out of time it covers
# ~84 % of later ratings on the reference profile (the 10-90 % band: 70 %) -
# i.e. "8 times out of 10", as the app says (2026-09-27, `malrec report`).
LOW, HIGH = 1, 19                  # indices of the 5 % and 95 % quantiles
# Population-based ranges do not drift the same way: there the 5-95 % band
# covered 89-92 % of held-out users' later ratings (global_model 8), the
# 10-90 % band is the honest "8 in 10".
POP_BAND = (2, 18)


def residual_quantiles(pred, truth) -> list[float]:
    e = np.asarray(truth, dtype=float) - np.asarray(pred, dtype=float)
    return [round(float(x), 3) for x in np.quantile(e, QS)]


def band(pred: float, q: list[float] | None, idx: tuple[int, int] | None = None) -> dict | None:
    """{'low', 'high', 'p9'} for a displayed prediction, or None. `idx` are
    the quantile indices of the band (default LOW, HIGH)."""
    if q is None or pred is None or not np.isfinite(pred):
        return None
    qa = np.asarray(q, dtype=float)
    lo, hi = _ends(pred, qa, idx)
    # P(rating >= 9) = P(error >= 8.5 - pred), from the error quantile function
    cdf = float(np.interp(8.5 - pred, qa, QS, left=0.0, right=1.0))
    return {"low": lo, "high": max(hi, lo), "p9": round(1.0 - cdf, 2)}


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


def user_quantiles(user_id: int) -> tuple[list[float] | None, tuple[int, int]]:
    """The error quantiles stored with the user's latest model, and the band
    to show: own holdout 5-95 %, population-based 10-90 %."""
    from .db import one
    row = one("SELECT metrics->'uncertainty' AS u FROM model_run WHERE user_id=%s"
              " ORDER BY id DESC LIMIT 1", (user_id,))
    u = (row or {}).get("u") or {}
    own = str(u.get("source", "")).startswith("own")
    return u.get("q"), ((LOW, HIGH) if own else POP_BAND)
