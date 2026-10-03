"""Taste model: learn the user's scoring function from their own ratings.

Two estimators are available. Ridge is the default because with ~150 ratings
and ~400 mostly-sparse features it is better behaved than a tree ensemble and
trains instantly; LightGBM is offered for users with much longer lists, and
`train` can pick whichever the evaluation harness scores higher.
"""
from __future__ import annotations

import io
import logging
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from .config import settings
from .db import conn, one, scalar
from .features import Vocabulary, training_set, vectorise

log = logging.getLogger(__name__)

RIDGE_ALPHAS = (10.0, 30.0, 100.0, 300.0, 1000.0)


@dataclass
class TasteModel:
    estimator: object
    vocab: Vocabulary
    algo: str
    params: dict
    metrics: dict
    baseline: float                       # recency-weighted personal mean
    calibration: tuple[float, float] = (1.0, 0.0)   # slope, intercept

    def predict(self, rows: list[dict]) -> np.ndarray:
        """Raw model output. Use this for ranking."""
        if not rows:
            return np.zeros(0)
        return self.estimator.predict(vectorise(rows, self.vocab))

    def predict_score(self, rows: list[dict]) -> np.ndarray:
        """Calibrated 1-10 estimate, for showing to a person.

        Ridge shrinks predictions toward the training mean, so the raw output
        is both biased and far too narrow: on this data it predicted a spread
        of 0.50 where the real spread was 1.27, and sat +0.64 high. A linear
        map fitted on the temporal holdout takes RMSE from 1.18 to 0.92, which
        is better than quoting the community score. Being monotonic it leaves
        the ranking untouched.
        """
        return self.calibrate(self.predict(rows))

    def calibrate(self, raw: np.ndarray) -> np.ndarray:
        a, b = self.calibration
        return np.clip(a * np.asarray(raw, dtype=float) + b, 1.0, 10.0)

    def dumps(self) -> bytes:
        buf = io.BytesIO()
        pickle.dump({"estimator": self.estimator, "vocab": self.vocab, "algo": self.algo,
                     "params": self.params, "metrics": self.metrics,
                     "baseline": self.baseline, "calibration": self.calibration}, buf)
        return buf.getvalue()

    @staticmethod
    def loads(blob: bytes) -> TasteModel:
        d = pickle.loads(blob)
        # artefacts stored before calibration existed
        d.setdefault("calibration", (1.0, 0.0))
        return TasteModel(**d)


def _fit_ridge(X, y, w, alpha: float) -> Ridge:
    return Ridge(alpha=alpha).fit(X, y, sample_weight=w)


def _fit_lgbm(X, y, w, **kw):
    import lightgbm as lgb
    params = {"objective": "regression", "learning_rate": 0.05, "num_leaves": 15,
              "min_data_in_leaf": 8, "feature_fraction": 0.7, "bagging_fraction": 0.8,
              "bagging_freq": 1, "verbose": -1, "n_estimators": 300}
    params.update(kw)
    return lgb.LGBMRegressor(**params).fit(X, y, sample_weight=w)


def choose_alpha(X, y, w, alphas=RIDGE_ALPHAS, folds: int = 10) -> float:
    """Pick the ridge penalty by weighted CV error. Uses plain K-fold because
    this is a variance question, not the drift question the temporal holdout
    in eval.py answers."""
    from sklearn.model_selection import KFold
    best, best_err = alphas[0], float("inf")
    for a in alphas:
        errs = []
        for tr, te in KFold(min(folds, len(y)), shuffle=True, random_state=0).split(X):
            m = _fit_ridge(X[tr], y[tr], w[tr], a)
            r = m.predict(X[te]) - y[te]
            errs.append(float(np.average(r ** 2, weights=w[te])))
        err = float(np.mean(errs))
        if err < best_err:
            best, best_err = a, err
    return best


def _hybrid_available() -> bool:
    if settings().model_mode == "personal":
        return False
    from .recsys.service import active_global
    return active_global() is not None


def _build_scorer(user_id: int, params: dict, metrics: dict, calibration):
    half_life = (params or {}).get("half_life")
    """Fit the user's hybrid model on their current list (well under a
    second) and wrap it so the ranking code can use it like a TasteModel."""
    from .db import query
    from .recsys.scorer import HybridScorer
    from .recsys.service import fit_for_user
    um = fit_for_user(user_id, half_life=half_life)
    if um is None:
        return None
    rated = query("""SELECT le.mal_id, le.score, a.title FROM list_entry le
                       JOIN anime a ON a.mal_id = le.mal_id
                      WHERE le.user_id=%s AND le.score>0""", (user_id,))
    cal = (metrics or {}).get("calibration") or {}
    list_cal = ((cal["top_slope"], cal["top_intercept"])
                if "top_slope" in cal else None)
    return HybridScorer(um, settings().model_mode, params, metrics, tuple(calibration),
                        {r["mal_id"]: float(r["score"]) for r in rated},
                        {r["mal_id"]: r["title"] for r in rated}, list_calibration=list_cal)


HALF_LIFE_CANDIDATES = (0.35, 0.5)
HALF_LIFE_MIN_RATED = 100


def choose_half_life(user_id: int) -> float:
    """Recency half-life for this user's personal model, from their own 8
    temporal splits. Measured (exp_signals --part mix/combo/final): 0.35 years
    lifts the reference profile's rank correlation 0.765 -> 0.780 (7 of 8
    splits), while held-out users lose 0.001-0.003 on average - so it is
    chosen per user, and only with enough history to decide; ties keep the
    configured default."""
    from .recsys.service import GATE_SPLITS, temporal_holdout
    default = settings().recency_half_life_years
    n = scalar("SELECT count(*) FROM list_entry WHERE user_id=%s AND score>0", (user_id,)) or 0
    if n < HALF_LIFE_MIN_RATED:
        return default
    best, best_rho = default, None
    for hl in sorted(set(HALF_LIFE_CANDIDATES) | {default}, key=lambda h: h != default):
        r = temporal_holdout(user_id, GATE_SPLITS, half_life=hl)
        rho = r.get("spearman")
        if rho is not None and (best_rho is None or rho > best_rho + 0.002):
            best, best_rho = hl, rho
    log.info("half-life for user %s: %s (rho %s)", user_id, best, best_rho)
    return best


def _uncertainty(user_id: int, preds, truth, cal: dict) -> dict | None:
    """Error quantiles behind the "likely 7-9" range: the user's own holdout
    errors when there are enough, else those of population users with a list
    this size (see malrec.uncertainty)."""
    from .recsys.service import active_global, size_quantiles
    from .uncertainty import residual_quantiles
    if len(preds) >= 40 and "rmse_raw" in cal:
        shown = np.clip(cal["slope"] * np.asarray(preds) + cal["intercept"], 1, 10)
        return {"q": residual_quantiles(shown, truth), "source": f"own, {len(preds)} ratings"}
    n = scalar("SELECT count(*) FROM list_entry WHERE user_id=%s AND score>0", (user_id,)) or 0
    gm = active_global()
    q = size_quantiles((gm.meta if gm else {}).get("size_calibration"), n)
    return {"q": q, "source": f"population, {n} ratings"} if q else None


def train_hybrid(user_id: int, persist: bool = True):
    """Evaluate on the user's own newest ratings, calibrate on those
    out-of-sample predictions, then fit on everything."""
    from .recsys.service import (
        active_global,
        calibration_from,
        size_calibration,
        temporal_holdout,
    )
    mode = settings().model_mode
    half_life = choose_half_life(user_id)
    metrics = temporal_holdout(user_id, half_life=half_life)
    preds, truth = metrics.pop("_preds", []), metrics.pop("_truth", [])
    cal = calibration_from(preds, truth)
    if "rmse_raw" not in cal:
        # No usable holdout of their own (too new, or one that would invert
        # the order): use the map measured on population users with a list
        # this size.
        n = scalar("SELECT count(*) FROM list_entry WHERE user_id=%s AND score>0",
                   (user_id,)) or 0
        gm = active_global()
        table = (gm.meta if gm else {}).get("size_calibration")
        pc = size_calibration(table, n)
        if pc is not None:
            cal = {"slope": round(pc[0], 4), "intercept": round(pc[1], 4),
                   "source": f"population, {n} ratings"}
            top = size_calibration(table, n, prefix="top_")
            if top is not None:
                cal |= {"top_slope": round(top[0], 4), "top_intercept": round(top[1], 4)}
    metrics = {**metrics, "calibration": cal, "uncertainty": _uncertainty(user_id, preds, truth, cal)}
    scorer = _build_scorer(user_id, {"half_life": half_life}, metrics,
                           (cal["slope"], cal["intercept"]))
    if scorer is None:
        raise ValueError(f"user {user_id} has no scored entries to learn from")
    um = scorer.um
    params = {"mode": mode, "half_life": half_life,
              "personal_share": round(float(um.lam), 3)
              if mode in ("blend", "sized") else None,
              "alpha": scorer.personal.alpha if scorer.personal is not None else None,
              "n_scored": sum(1 for v in scorer._scores.values() if v > 0)}
    scorer.params = params
    log.info("trained %s for user %s: %s", mode, user_id, metrics)
    run_id = None
    if persist:
        blob = pickle.dumps({"kind": "hybrid", "mode": mode, "params": params,
                             "metrics": metrics, "calibration": scorer.calibration})
        with conn() as c, c.cursor() as cur:
            from psycopg.types.json import Jsonb
            cur.execute(
                "INSERT INTO model_run (user_id, algo, params, metrics, feature_names,"
                " n_train, artifact) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (user_id, mode, Jsonb(params), Jsonb(metrics), [], params["n_scored"], blob))
            run_id = cur.fetchone()["id"]
            c.commit()
    return scorer, run_id


def train(user_id: int, algo: str = "ridge", persist: bool = True,
          half_life: float | None = None) -> tuple[TasteModel, int | None]:
    if algo == "ridge" and half_life is None and _hybrid_available():
        return train_hybrid(user_id, persist=persist)
    X, y, w, vocab, _ = training_set(user_id, half_life=half_life)
    baseline = float(np.average(y, weights=w))

    if algo == "ridge":
        alpha = choose_alpha(X, y, w)
        est = _fit_ridge(X, y, w, alpha)
        params = {"alpha": alpha, "half_life": half_life or settings().recency_half_life_years}
    elif algo == "lgbm":
        est = _fit_lgbm(X, y, w)
        params = {"half_life": half_life or settings().recency_half_life_years}
    else:
        raise ValueError(f"unknown algo {algo!r}")

    from .eval import fit_calibration, temporal_holdout
    metrics = temporal_holdout(user_id, algo=algo, half_life=half_life)
    cal = fit_calibration(user_id, algo=algo, half_life=half_life)
    metrics = {**metrics, "calibration": cal}
    model = TasteModel(est, vocab, algo, params, metrics, baseline,
                       (cal["slope"], cal["intercept"]))
    log.info("trained %s for user %s: %s", algo, user_id, metrics)

    run_id = None
    if persist:
        with conn() as c, c.cursor() as cur:
            from psycopg.types.json import Jsonb
            cur.execute(
                "INSERT INTO model_run (user_id, algo, params, metrics, feature_names,"
                " n_train, artifact) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
                (user_id, algo, Jsonb(params), Jsonb(metrics), vocab.names, len(y),
                 model.dumps()),
            )
            run_id = cur.fetchone()["id"]
            c.commit()
    return model, run_id


def load_latest(user_id: int) -> tuple[TasteModel, int] | tuple[None, None]:
    row = one(
        "SELECT id, artifact FROM model_run WHERE user_id=%s AND artifact IS NOT NULL"
        " ORDER BY trained_at DESC LIMIT 1",
        (user_id,),
    )
    if not row:
        return None, None
    d = pickle.loads(bytes(row["artifact"]))
    if isinstance(d, dict) and d.get("kind") == "hybrid":
        # The user model is refitted from the current list rather than stored:
        # it takes well under a second and can never go stale that way.
        if not _hybrid_available():
            return None, None
        scorer = _build_scorer(user_id, d["params"], d["metrics"], d["calibration"])
        return (scorer, row["id"]) if scorer is not None else (None, None)
    if _hybrid_available():
        return None, None        # legacy artefact; retrain on the current model
    return TasteModel.loads(bytes(row["artifact"])), row["id"]


_SCORER_TTL = 600.0
_scorers: dict[int, tuple[float, object, int | None]] = {}


def cached_model(user_id: int) -> tuple[object, int | None]:
    """load_or_train, remembered for a few minutes - for per-request work such
    as the "why this?" breakdown, where refitting on every click is waste."""
    import time
    hit = _scorers.get(user_id)
    if hit is not None and time.time() - hit[0] < _SCORER_TTL:
        return hit[1], hit[2]
    model, run_id = load_or_train(user_id)
    _scorers[user_id] = (time.time(), model, run_id)
    return model, run_id


def invalidate_scorer(user_id: int) -> None:
    _scorers.pop(user_id, None)


# The user model is refitted from the current list on every load, but what a
# full train chooses - calibration and likely ranges by list size, recency
# half-life - is kept from the run. Once a list has grown this much since,
# those choices are out of date (22 ratings calibrated as 11), so the next
# build retrains.
RETRAIN_GROWTH = 1.25
RETRAIN_MIN_NEW = 5


def outgrown(user_id: int, run_id: int | None) -> bool:
    if run_id is None:
        return False
    was = scalar("SELECT n_train FROM model_run WHERE id=%s", (run_id,)) or 0
    now = scalar("SELECT count(*) FROM list_entry WHERE user_id=%s AND score>0", (user_id,)) or 0
    return was > 0 and now >= max(was + RETRAIN_MIN_NEW, was * RETRAIN_GROWTH)


def load_or_train(user_id: int, algo: str = "ridge") -> tuple[TasteModel, int | None]:
    model, run_id = load_latest(user_id)
    if model is None:
        return train(user_id, algo=algo)
    if outgrown(user_id, run_id):
        log.info("user %s: list outgrew model run %s, retraining", user_id, run_id)
        invalidate_scorer(user_id)
        return train(user_id, algo=algo)
    return model, run_id


def feature_importance(model: TasteModel, top: int = 25) -> list[tuple[str, float]]:
    """What the model actually learned - used for the /explain endpoint and to
    sanity-check that a retrain has not gone sideways. Empty when there is
    no personal model: a short list whose blend gives it no weight (under
    ~40 ratings with the handover at 150) is never fitted one."""
    if getattr(model, "vocab", None) is None or getattr(model, "estimator", None) is None:
        return []
    names = model.vocab.names
    if hasattr(model.estimator, "coef_"):
        vals = np.asarray(model.estimator.coef_, dtype=float)
    elif hasattr(model.estimator, "feature_importances_"):
        vals = np.asarray(model.estimator.feature_importances_, dtype=float)
    else:
        return []
    order = np.argsort(-np.abs(vals))[:top]
    return [(names[i], float(vals[i])) for i in order if i < len(names)]
