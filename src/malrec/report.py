"""One evaluation report: `malrec report` (run monthly by the refresh job).

For every app user with enough history:
  gate        rank correlation on the 8 temporal splits, next to two trivial
              baselines on the same splits - the community score and the
              user's mean plus the population item bias - so a change can
              never be judged only against its own predecessor
  coverage    how often actual ratings fell inside the shown "likely" range,
              strictly out of sample: error quantiles from the 4 older
              splits, checked on the 4 newer ones (target 0.80)
  prospective predictions as shown vs scores given afterwards
Plus the active population model, its per-size range coverage, and the
prospective check (the active model refitted as of the first recommendation,
on the ratings given since) - the test a candidate model has to pass.
The heavier population-level comparison is experiments/exp_audit.py.
"""
from __future__ import annotations

import numpy as np

from .config import settings
from .db import one, query
from .eval import prospective, spearman
from .recsys.service import (
    GATE_SPLITS,
    active_global,
    fit_for_user,
    item_store,
    prospective_compare,
)
from .uncertainty import coverage, residual_quantiles

RATED_SQL = """SELECT mal_id, score::float AS score,
                      coalesce(finished_at::timestamptz, updated_at, now()) AS at
                 FROM list_entry WHERE user_id=%s AND score>0 ORDER BY at"""


def user_report(user_id: int) -> dict:
    run = one("SELECT params, metrics FROM model_run WHERE user_id=%s ORDER BY id DESC LIMIT 1",
              (user_id,)) or {}
    params, metrics = run.get("params") or {}, run.get("metrics") or {}
    cal = metrics.get("calibration") or {}
    a, b = float(cal.get("slope", 1.0)), float(cal.get("intercept", 0.0))
    rated = query(RATED_SQL, (user_id,))
    out: dict = {"ratings": len(rated), "half_life": params.get("half_life"),
                 "prospective": prospective(user_id)}
    if len(rated) < max(GATE_SPLITS) + 25:
        out["gate"] = "not enough history"
        return out
    store = item_store()
    rho = {"model": [], "community score": [], "mean + item bias": []}
    old_err, new_shown, new_truth = [], [], []
    for h in GATE_SPLITS:
        test = rated[-h:]
        um = fit_for_user(user_id, cutoff=test[0]["at"], half_life=params.get("half_life"))
        items = [r["mal_id"] for r in test]
        truth = np.array([r["score"] for r in test])
        p = um.predict(items)
        mal = np.array([(store.rows.get(m) or {}).get("mal_mean") or 7.5 for m in items], float)
        mb = um.fold.mu + um.fold.signals(items)["bias"]
        ok = ~np.isnan(p)
        for k, v in (("model", p), ("community score", mal), ("mean + item bias", mb)):
            rho[k].append(round(spearman(v[ok], truth[ok]), 4))
        shown = np.clip(a * p[ok] + b, 1, 10)
        if h >= 45:
            old_err.append((shown, truth[ok]))
        else:
            new_shown.extend(shown.tolist()); new_truth.extend(truth[ok].tolist())
    out["gate"] = {k: {"mean": round(float(np.mean(v)), 4), "per_split": v}
                   for k, v in rho.items()}
    if old_err:
        q = residual_quantiles(np.concatenate([s for s, _ in old_err]),
                               np.concatenate([t for _, t in old_err]))
        out["range_coverage_80"] = round(coverage(q, new_shown, new_truth), 3)
    return out


def report() -> dict:
    gm = active_global()
    out: dict = {"population_model": None}
    if gm is not None:
        gid = one("SELECT id FROM global_model WHERE active")["id"]
        out["population_model"] = {
            "id": gid, "users": gm.meta.get("users"),
            "range_coverage_80_by_size": {r["n"]: r.get("coverage80")
                                         for r in gm.meta.get("size_calibration", [])}}
        if settings().malrec_user:
            out["prospective_check"] = prospective_compare(gm, gm, settings().malrec_user)
    out["users"] = {r["mal_username"]: user_report(r["id"])
                    for r in query("SELECT id, mal_username FROM app_user ORDER BY id")}
    return out
