"""Experiment: does the unscored half of the list carry usable signal?

107 of 250 entries have no score. Dropping something is an opinion; so is
stalling on it, finishing it without rating, or queueing it. The question is
whether turning those into weak pseudo-ratings helps predict real ones.

The user's own per-status means are useless here (n=1 for dropped, on_hold and
plan_to_watch), so the offsets are global hyperparameters fitted on the same
temporal holdout that decides whether to keep them at all.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge

from malrec.config import settings
from malrec.db import one, query
from malrec.eval import ndcg_at, spearman
from malrec.features import build_vocabulary, load_rows, vectorise

UID = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
CFG = settings()
HOLDOUTS = (25, 30, 35, 40, 45, 50, 55, 60)

SCORED = query(
    """SELECT mal_id, score, coalesce(finished_at::timestamptz, updated_at, now()) AS at
         FROM list_entry WHERE user_id=%s AND score>0 ORDER BY at ASC""", (UID,))
IMPLICIT = query(
    """SELECT mal_id, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
         FROM list_entry WHERE user_id=%s AND score=0 ORDER BY at ASC""", (UID,))

ALL_IDS = [r["mal_id"] for r in SCORED] + [r["mal_id"] for r in IMPLICIT]
ROWS = load_rows(UID, ALL_IDS)
MU = float(np.mean([r["score"] for r in SCORED]))


def recency(at, cutoff, hl=None, floor=None):
    hl = CFG.recency_half_life_years if hl is None else hl
    floor = CFG.recency_floor if floor is None else floor
    age = max((cutoff - at).total_seconds() / 31_557_600.0, 0.0)
    return floor + (1 - floor) * 0.5 ** (age / hl)


def run(offsets: dict[str, float], weight: float, holdouts=HOLDOUTS) -> dict:
    """offsets: status -> pseudo-score relative to the user's mean.
    weight: how much an implicit row counts against a real rating."""
    rhos, nds, rmses = [], [], []
    for h in holdouts:
        train_s, test_s = SCORED[:-h], SCORED[-h:]
        cutoff = test_s[0]["at"]
        rows, ys, ws = [], [], []

        for r in train_s:
            row = ROWS.get(r["mal_id"])
            if row is None:
                continue
            rows.append(row); ys.append(float(r["score"]))
            ws.append(recency(r["at"], cutoff))

        for r in IMPLICIT:
            off = offsets.get(r["status"])
            # only what was already known at prediction time
            if off is None or r["at"] >= cutoff:
                continue
            row = ROWS.get(r["mal_id"])
            if row is None:
                continue
            rows.append(row)
            ys.append(float(np.clip(MU + off, 1.0, 10.0)))
            ws.append(weight * recency(r["at"], cutoff))

        te_rows = [ROWS[r["mal_id"]] for r in test_s if r["mal_id"] in ROWS]
        yte = np.array([float(r["score"]) for r in test_s if r["mal_id"] in ROWS])
        if len(rows) < 20 or len(te_rows) < 5:
            continue

        vocab = build_vocabulary(rows)
        Xtr, Xte = vectorise(rows, vocab), vectorise(te_rows, vocab)
        est = Ridge(alpha=30.0).fit(Xtr, np.array(ys), sample_weight=np.array(ws))
        p = est.predict(Xte)
        rhos.append(spearman(p, yte))
        nds.append(ndcg_at(p, list(yte), 10))
        rmses.append(float(np.sqrt(np.mean((p - yte) ** 2))))
    return {"rho": float(np.mean(rhos)), "ndcg": float(np.mean(nds)),
            "rmse": float(np.mean(rmses)), "per": rhos}


def show(label, res, base=None):
    d = ""
    if base:
        deltas = np.array(res["per"]) - np.array(base["per"])
        d = f"  Δ{deltas.mean():+.3f}  wins {int((deltas > 0).sum())}/{len(deltas)}"
    print(f"{label:<44}{res['rho']:7.3f}{res['ndcg']:8.3f}{res['rmse']:8.3f}{d}")


if __name__ == "__main__":
    print(f"{'configuration':<44}{'rho':>7}{'nDCG@10':>8}{'rmse':>8}")
    print("-" * 78)
    base = run({}, 0.0)
    show("scored only (baseline)", base)

    print("\n-- one status at a time (weight 0.3) --")
    singles = {
        "dropped  (mu - 2.5)": {"dropped": -2.5},
        "on_hold  (mu - 1.0)": {"on_hold": -1.0},
        "completed unscored (mu + 0.0)": {"completed": 0.0},
        "watching (mu + 0.2)": {"watching": 0.2},
        "plan_to_watch (mu + 0.1)": {"plan_to_watch": 0.1},
    }
    for label, off in singles.items():
        show(label, run(off, 0.3), base)

    print("\n-- negatives only, weight sweep --")
    neg = {"dropped": -2.5, "on_hold": -1.0}
    for w in (0.15, 0.3, 0.5, 0.8):
        show(f"dropped+on_hold  w={w}", run(neg, w), base)

    print("\n-- offset sweep for the negatives (w=0.5) --")
    for d_off, h_off in [(-1.5, -0.5), (-2.0, -1.0), (-2.5, -1.0), (-3.0, -1.5), (-3.5, -2.0)]:
        show(f"dropped {d_off}  on_hold {h_off}", run({"dropped": d_off, "on_hold": h_off}, 0.5), base)

    print("\n-- combinations around the best negatives (w=0.5) --")
    best_neg = {"dropped": -1.5, "on_hold": -0.5}
    combos = {
        "negatives only": best_neg,
        "+ completed unscored": {**best_neg, "completed": 0.0},
        "+ plan_to_watch": {**best_neg, "plan_to_watch": 0.1},
        "+ completed + watching": {**best_neg, "completed": 0.0, "watching": 0.2},
        "everything": {**best_neg, "completed": 0.0, "watching": 0.2, "plan_to_watch": 0.1},
    }
    for label, off in combos.items():
        show(label, run(off, 0.5), base)
