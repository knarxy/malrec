"""Which displayed number is honest for a user with a long, drifting history?

The stored map (slope, intercept) is fitted on out-of-sample predictions from
models trained on *older* data. When the user's ratings drift (reference profile:
oldest 83 average 8.24, newest 30 average 7.37), those models overpredicted,
so the intercept learns a downward correction - and is then applied to a
final model that has already seen the recent ratings. The correction happens
twice, and a rating slightly above the displayed number can move it down.

Nested, strictly out of time: for each outer split the model is trained on
everything before it; any map may only use that history; it is scored on the
outer split's ratings.

  identity     the model's own output
  stored       least squares on inner holdouts (what production does)
  anchored     stored slope, intercept re-set so the final model's in-sample
               predictions for the newest 30 training ratings average what
               they actually were - the level comes from recent reality, the
               spread from the holdout
  shifted      slope 1, same level anchoring
"""
from __future__ import annotations

import argparse

import numpy as np

from malrec.config import settings
from malrec.db import one, query
from malrec.recsys.hybrid import fit_user
from malrec.recsys.service import (
    active_global,
    calibration_from,
    item_store,
    user_examples,
    user_listed,
)

OUTER = (25, 30, 35, 40, 45, 50, 55, 60)
INNER = (30, 40, 50)
ANCHOR = 30


def fit_at(uid, cutoff, gm, store):
    scored, implicit = user_examples(uid, cutoff)
    return fit_user(gm.pop, gm.stacker, store, scored, implicit, mode=settings().model_mode,
                    listed=user_listed(uid, cutoff))


def main(user: str, level: str = ""):
    settings().level_trend = level
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (user,))["id"]
    gm, store = active_global(), item_store()
    rated = query("""SELECT mal_id, score::float s, coalesce(finished_at::timestamptz, updated_at, now()) at
                       FROM list_entry WHERE user_id=%s AND score>0 ORDER BY at""", (uid,))
    res = {k: ([], []) for k in ("identity", "stored", "anchored", "shifted")}
    for h in OUTER:
        hist, test = rated[:-h], rated[-h:]
        cutoff = test[0]["at"]
        final = fit_at(uid, cutoff, gm, store)
        raw_test = final.predict([r["mal_id"] for r in test])
        truth = np.array([r["s"] for r in test])
        # inner holdouts within the history only
        pp, tt = [], []
        for hi in INNER:
            if len(hist) < hi + 25:
                continue
            inner = hist[-hi:]
            m = fit_at(uid, inner[0]["at"], gm, store)
            p = m.predict([r["mal_id"] for r in inner])
            ok = ~np.isnan(p)
            pp += p[ok].tolist(); tt += np.array([r["s"] for r in inner])[ok].tolist()
        c = calibration_from(pp, tt)
        a, b = c["slope"], c["intercept"]
        anc = hist[-ANCHOR:]
        raw_anc = final.predict([r["mal_id"] for r in anc])
        ok = ~np.isnan(raw_anc)
        act = float(np.mean([r["s"] for r in anc]))
        maps = {"identity": (1.0, 0.0), "stored": (a, b),
                "anchored": (a, act - a * float(np.mean(raw_anc[ok]))),
                "shifted": (1.0, act - float(np.mean(raw_anc[ok])))}
        ok = ~np.isnan(raw_test)
        for k, (sl, ic) in maps.items():
            res[k][0].extend(np.clip(sl * raw_test[ok] + ic, 1, 10).tolist())
            res[k][1].extend(truth[ok].tolist())
    print(f"{user}: {len(OUTER)} outer splits, strictly out of time, level_trend={level!r}")
    print(f"{'map':<10}{'bias':>7}{'RMSE':>7}{'top-10% bias':>14}{'bottom-50% bias':>17}")
    for k, (shown, truth) in res.items():
        q, t = np.array(shown), np.array(truth)
        e = q - t
        o = np.argsort(-q)
        top, bot = o[:max(1, len(o) // 10)], o[len(o) // 2:]
        print(f"{k:<10}{e.mean():>+7.2f}{np.sqrt((e ** 2).mean()):>7.3f}"
              f"{e[top].mean():>+14.2f}{e[bot].mean():>+17.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=settings().malrec_user)
    ap.add_argument("--level", default="", help="level_trend variant to evaluate")
    a = ap.parse_args()
    main(a.user, a.level)
