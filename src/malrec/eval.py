"""Offline evaluation.

The primary metric is a TEMPORAL holdout: train on the user's older ratings,
predict their most recent ones. Random K-fold is also available but must not be
used to judge recency weighting - it lets the model train on 2026 ratings to
predict a 2018 one, which is exactly the leakage that hides taste drift.

Measured on the reference profile (143 ratings), random K-fold said recency weighting
*hurt* (rho 0.615 -> 0.578) while the temporal holdout showed it clearly helps
(rho 0.654 -> 0.765). Two controls confirmed the effect is drift and not just
a smaller effective sample: weighting OLD ratings up scored 0.659, and randomly
subsampling to the same effective n scored 0.680.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from .config import settings
from .db import query
from .features import build_vocabulary, implicit_rows, load_rows, vectorise

log = logging.getLogger(__name__)


@dataclass
class Split:
    train_ids: list[int]
    test_ids: list[int]


def _rated_in_time_order(user_id: int) -> list[dict]:
    """Oldest first. finished_at is when they actually watched it; updated_at
    is the fallback and is always populated."""
    return query(
        """
        SELECT mal_id, score,
               coalesce(finished_at::timestamptz, updated_at, now()) AS at
          FROM list_entry
         WHERE user_id = %s AND score > 0
         ORDER BY at ASC
        """,
        (user_id,),
    )


def _weights(rows: list[dict], cutoff, half_life: float | None, floor: float) -> np.ndarray:
    """Ages are measured from the CUTOFF, never from today: at prediction time
    the model could only know about ratings made before it."""
    if half_life is None or half_life <= 0:
        return np.ones(len(rows))
    out = []
    for r in rows:
        age = max((cutoff - r["at"]).total_seconds() / 31_557_600.0, 0.0)
        out.append(floor + (1 - floor) * 0.5 ** (age / half_life))
    return np.array(out)


def spearman(a, b) -> float:
    """Spearman's rho with tied values given their average rank.

    Scores are integers, so ties are the norm, not an edge case. An earlier
    version ranked with argsort-of-argsort, which breaks ties by position in
    the array; that gave a constant prediction a non-zero correlation (0.12
    on sampled users) and nudged every reported rho. A constant prediction
    now scores exactly 0.
    """
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return 0.0
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def ndcg_at(pred, truth, k: int = 10) -> float:
    pred, truth = np.asarray(pred, float), list(truth)
    if not truth:
        return 0.0
    order = np.argsort(-pred)[:k]
    dcg = sum((2 ** truth[i] - 1) / math.log2(r + 2) for r, i in enumerate(order))
    ideal = sorted(truth, reverse=True)[:k]
    idcg = sum((2 ** t - 1) / math.log2(r + 2) for r, t in enumerate(ideal))
    return float(dcg / idcg) if idcg else 0.0


def _fit_predict(user_id, train, test, half_life, floor, algo, alpha, cutoff):
    """Fit on `train` (plus implicit rows predating the cutoff) and score `test`.

    The implicit rows go through the identical path the product uses, so the
    holdout measures the model that actually ships rather than a simplified
    one. Anything the user only touched after the cutoff is excluded.
    """
    w_scored = _weights(train, cutoff, half_life, floor)
    mean = float(np.average([r["score"] for r in train], weights=np.maximum(w_scored, 1e-6)))         if len(train) else 7.0
    extra = implicit_rows(user_id, mean, cutoff=cutoff, half_life=half_life)

    ids = [r["mal_id"] for r in train + test] + [r["mal_id"] for r in extra]
    rows_by_id = load_rows(user_id, ids, half_life=half_life)
    train = [r for r in train if r["mal_id"] in rows_by_id]
    test = [r for r in test if r["mal_id"] in rows_by_id]
    extra = [r for r in extra if r["mal_id"] in rows_by_id]
    if len(train) < 10 or not test:
        return None, None

    tr_rows = [rows_by_id[r["mal_id"]] for r in train + extra]
    te_rows = [rows_by_id[r["mal_id"]] for r in test]
    vocab = build_vocabulary(tr_rows)               # fit on train only
    Xtr, Xte = vectorise(tr_rows, vocab), vectorise(te_rows, vocab)
    ytr = np.array([float(r["score"]) for r in train + extra])
    yte = np.array([float(r["score"]) for r in test])
    w = np.concatenate([_weights(train, cutoff, half_life, floor),
                        np.array([float(r["w"]) for r in extra])])

    if algo == "lgbm":
        from .model import _fit_lgbm
        est = _fit_lgbm(Xtr, ytr, w)
    else:
        est = Ridge(alpha=alpha).fit(Xtr, ytr, sample_weight=w)
    return est.predict(Xte), yte


def temporal_holdout(user_id: int, holdouts: tuple[int, ...] = (30, 40, 50),
                     half_life: float | None = None, floor: float | None = None,
                     algo: str = "ridge", alpha: float = 30.0) -> dict:
    """Train on older ratings, score the most recent `h`. Averaged over several
    holdout sizes because any single cut is noisy at ~150 ratings."""
    cfg = settings()
    half_life = cfg.recency_half_life_years if half_life is None else half_life
    floor = cfg.recency_floor if floor is None else floor

    rated = _rated_in_time_order(user_id)
    results = []
    for h in holdouts:
        if len(rated) < h + 25:
            continue
        train, test = rated[:-h], rated[-h:]
        cutoff = test[0]["at"]
        pred, yte = _fit_predict(user_id, train, test, half_life, floor, algo, alpha, cutoff)
        if pred is None:
            continue
        results.append({
            "holdout": h,
            "spearman": spearman(pred, yte),
            "rmse": float(np.sqrt(np.mean((pred - yte) ** 2))),
            "ndcg@10": ndcg_at(pred, yte, 10),
        })
    if not results:
        return {"note": "not enough rated history for a temporal holdout",
                "n_rated": len(rated)}
    return {
        "method": "temporal_holdout",
        "n_rated": len(rated),
        "half_life_years": half_life,
        "spearman": round(float(np.mean([r["spearman"] for r in results])), 4),
        "rmse": round(float(np.mean([r["rmse"] for r in results])), 4),
        "ndcg@10": round(float(np.mean([r["ndcg@10"] for r in results])), 4),
        "per_holdout": results,
    }


def kfold(user_id: int, folds: int = 10, half_life: float | None = None,
          algo: str = "ridge", alpha: float = 30.0) -> dict:
    """Random K-fold. Fine for choosing a regularisation strength; misleading
    for anything time-dependent."""
    cfg = settings()
    half_life = cfg.recency_half_life_years if half_life is None else half_life
    rated = _rated_in_time_order(user_id)
    if len(rated) < folds * 3:
        return {"note": "too few ratings", "n_rated": len(rated)}
    idx = np.arange(len(rated))
    preds, truth = np.zeros(len(rated)), np.array([float(r["score"]) for r in rated])
    now = max(r["at"] for r in rated)
    for tr, te in KFold(folds, shuffle=True, random_state=0).split(idx):
        p, _ = _fit_predict(user_id, [rated[i] for i in tr], [rated[i] for i in te],
                            half_life, cfg.recency_floor, algo, alpha, now)
        if p is not None:
            preds[te] = p
    return {"method": "kfold", "folds": folds, "n_rated": len(rated),
            "spearman": round(spearman(preds, truth), 4),
            "rmse": round(float(np.sqrt(np.mean((preds - truth) ** 2))), 4),
            "ndcg@10": round(ndcg_at(preds, truth, 10), 4)}


def tune_recency(user_id: int,
                 half_lives: tuple[float | None, ...] = (None, 4.0, 2.0, 1.0, 0.75, 0.5, 0.35),
                 floors: tuple[float, ...] = (0.05, 0.15)) -> list[dict]:
    """Grid-search the decay on the temporal holdout. Reported by
    `malrec tune-recency`; the winner belongs in config as the new default."""
    out = []
    for hl in half_lives:
        for fl in floors:
            if hl is None and fl != floors[0]:
                continue                     # floor is meaningless with no decay
            m = temporal_holdout(user_id, half_life=hl, floor=fl)
            out.append({"half_life": hl, "floor": fl,
                        "spearman": m.get("spearman"), "rmse": m.get("rmse"),
                        "ndcg@10": m.get("ndcg@10")})
    out.sort(key=lambda r: -(r["spearman"] or -1))
    return out


def compare_algos(user_id: int) -> list[dict]:
    return [{"algo": a, **temporal_holdout(user_id, algo=a)} for a in ("ridge", "lgbm")]


def fit_calibration(user_id: int, holdouts: tuple[int, ...] = (30, 40, 50),
                    half_life: float | None = None, floor: float | None = None,
                    algo: str = "ridge", alpha: float = 30.0) -> dict:
    """Least-squares map from raw prediction to actual score.

    Fitted on out-of-sample holdout predictions, never on the training fit,
    or it would simply learn the in-sample shrinkage. Returns (1.0, 0.0) when
    there is too little history to fit anything trustworthy.
    """
    cfg = settings()
    half_life = cfg.recency_half_life_years if half_life is None else half_life
    floor = cfg.recency_floor if floor is None else floor
    rated = _rated_in_time_order(user_id)

    preds: list[float] = []
    truth: list[float] = []
    for h in holdouts:
        if len(rated) < h + 25:
            continue
        train, test = rated[:-h], rated[-h:]
        p, y = _fit_predict(user_id, train, test, half_life, floor, algo, alpha,
                            test[0]["at"])
        if p is not None:
            preds.extend(p.tolist())
            truth.extend(y.tolist())

    if len(preds) < 20 or float(np.std(preds)) < 1e-6:
        return {"slope": 1.0, "intercept": 0.0}
    p_arr, t_arr = np.array(preds), np.array(truth)
    slope, intercept = np.polyfit(p_arr, t_arr, 1)
    # A non-positive slope would invert the ranking; fall back to identity.
    if slope <= 0:
        return {"slope": 1.0, "intercept": 0.0}
    fitted = np.clip(slope * p_arr + intercept, 1.0, 10.0)
    return {
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "rmse_raw": round(float(np.sqrt(((p_arr - t_arr) ** 2).mean())), 4),
        "rmse_calibrated": round(float(np.sqrt(((fitted - t_arr) ** 2).mean())), 4),
        "pred_sd_raw": round(float(p_arr.std()), 3),
        "actual_sd": round(float(t_arr.std()), 3),
    }


def tune_implicit(user_id: int, holdouts: tuple[int, ...] = (25, 30, 35, 40, 45, 50, 55, 60)
                  ) -> list[dict]:
    """Grid-search the pseudo-rating offsets and weight for unscored entries.

    The user's own data cannot supply these: a typical profile has a handful
    of scored entries per status at most. So they are fitted against the same
    temporal holdout that decides whether the signal is worth using, and the
    per-split win count is reported so a lucky average is visible as such.
    """
    cfg = settings()
    saved_w, saved_off = cfg.implicit_weight, dict(cfg.implicit_offsets)
    out = []
    try:
        cfg.implicit_weight = 0.0
        base = temporal_holdout(user_id, holdouts=holdouts)
        base_per = [r["spearman"] for r in base["per_holdout"]]
        out.append({"weight": 0.0, "offsets": {}, "spearman": base["spearman"],
                    "rmse": base["rmse"], "ndcg@10": base["ndcg@10"], "wins": None})

        for w in (0.3, 0.5, 0.8):
            for dropped, on_hold in ((-1.0, -0.3), (-1.5, -0.5), (-2.0, -1.0), (-2.5, -1.5)):
                cfg.implicit_weight = w
                cfg.implicit_offsets = {"dropped": dropped, "on_hold": on_hold,
                                        "completed": 0.0}
                m = temporal_holdout(user_id, holdouts=holdouts)
                per = [r["spearman"] for r in m["per_holdout"]]
                wins = sum(1 for a, b in zip(per, base_per) if a > b)
                out.append({"weight": w, "offsets": dict(cfg.implicit_offsets),
                            "spearman": m["spearman"], "rmse": m["rmse"],
                            "ndcg@10": m["ndcg@10"], "wins": f"{wins}/{len(per)}"})
    finally:
        cfg.implicit_weight, cfg.implicit_offsets = saved_w, saved_off
    out.sort(key=lambda r: -(r["spearman"] or -1))
    return out


PROSPECTIVE_SQL = """
    SELECT DISTINCT ON (l.mal_id) l.mal_id, l.predicted, l.shown_at, le.score
      FROM rec_log l
      JOIN list_entry le ON le.user_id = l.user_id AND le.mal_id = l.mal_id
     WHERE l.user_id = %s AND le.score > 0 AND le.updated_at > l.shown_at
     ORDER BY l.mal_id, l.shown_at
"""


def prospective(user_id: int) -> dict:
    """The one test no model can have seen: predictions as they were shown
    (rec_log) against scores given afterwards. Grows with every rating the
    nightly sync brings in; below 10 pairs it only reports the count."""
    rows = query(PROSPECTIVE_SQL, (user_id,))
    out: dict = {"pairs": len(rows)}
    if len(rows) < 10:
        out["note"] = "fewer than 10 recommended titles rated since they were shown"
        return out
    p = np.array([r["predicted"] for r in rows], dtype=float)
    t = np.array([r["score"] for r in rows], dtype=float)
    out.update({"spearman": round(spearman(p, t), 4),
                "bias": round(float((p - t).mean()), 3),
                "rmse": round(float(np.sqrt(((p - t) ** 2).mean())), 3),
                "since": min(r["shown_at"] for r in rows)})
    return out
