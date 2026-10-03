"""The gate: does the hybrid lose anything for an app user with a long history?

Same protocol that produced the baseline - 8 temporal holdouts (25..60 most
recent ratings), implicit rows included, recency weights from the cutoff -
so current and hybrid are compared split by split on identical data.

The population model is fitted on the CF sample only; app users are excluded
from that sample at fetch time, so nothing about the evaluated user leaks in.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
from sklearn.linear_model import Ridge

from malrec.config import settings
from malrec.db import one, query
from malrec.eval import _fit_predict, _rated_in_time_order, ndcg_at, spearman
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_user
from malrec.recsys.items import ItemStore

CFG = settings()
H = (25, 30, 35, 40, 45, 50, 55, 60)


def recency(at: dt.datetime, cutoff: dt.datetime) -> float:
    age = max((cutoff - at).total_seconds() / 31_557_600.0, 0.0)
    return CFG.recency_floor + (1 - CFG.recency_floor) * 0.5 ** (age / CFG.recency_half_life_years)


def fit_global(seed: int = 0, params: CFParams | None = None):
    """Population model on 85% of the CF sample, stacker on the rest."""
    from experiments.exp_population import fit_stacker, load_cf
    data = load_cf()
    users = sorted(data)
    np.random.default_rng(seed).shuffle(users)
    cut = int(len(users) * 0.85)
    ua, ia, ra = [], [], []
    for u in users[:cut]:
        for m, s, _ in data[u]:
            ua.append(u); ia.append(m); ra.append(s)
    pop = PopulationModel(params or CFParams()).fit(np.array(ua), np.array(ia), np.array(ra))
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: data[u] for u in users[cut:]}, store)
    return pop, stacker, store, len(users)


def implicit_before(user_id: int, cutoff, mean: float) -> list[Example]:
    rows = query("""SELECT mal_id, status, finished_at, updated_at,
                           coalesce(finished_at::timestamptz, updated_at, now()) AS at
                      FROM list_entry WHERE user_id=%s AND score=0""", (user_id,))
    out = []
    for r in rows:
        off = CFG.implicit_offsets.get(r["status"])
        if off is None or r["at"] >= cutoff:
            continue
        out.append(Example(r["mal_id"], float(min(max(mean + off, 1), 10)),
                           CFG.implicit_weight * recency(r["at"], cutoff)))
    return out


def gate(username: str, pop, stacker, store, variants: dict) -> dict:
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (username,))["id"]
    rated = _rated_in_time_order(uid)
    res = {name: {"rho": [], "ndcg": [], "rmse": []} for name in ["current", *variants]}
    for h in H:
        if len(rated) < h + 25:
            continue
        train, test = rated[:-h], rated[-h:]
        cutoff = test[0]["at"]
        truth = np.array([float(r["score"]) for r in test])
        # current production model, exactly as the baseline was measured
        p, y = _fit_predict(uid, train, test, CFG.recency_half_life_years, CFG.recency_floor,
                            "ridge", 30.0, cutoff)
        preds = {"current": (p, y)}
        scored = [Example(r["mal_id"], float(r["score"]), recency(r["at"], cutoff)) for r in train]
        mean = float(np.average([e.score for e in scored], weights=[e.weight for e in scored]))
        imp = implicit_before(uid, cutoff, mean)
        items = [r["mal_id"] for r in test]
        for name, opts in variants.items():
            if opts.get("current_leakfree"):
                preds[name] = (current_leakfree(store, scored, imp, items,
                                                taste_from_all=opts.get("taste_all"),
                                                uid=uid), truth)
                continue
            um = fit_user(pop, stacker, store, scored, imp if opts.get("implicit") else [],
                          alpha=opts.get("alpha"))
            pr = um.stack(items) if opts.get("stack_only") else um.predict(items)
            preds[name] = (pr, truth)
        for name, (pp, yy) in preds.items():
            res[name]["rho"].append(spearman(pp, yy))
            res[name]["ndcg"].append(ndcg_at(pp, list(yy), 10))
            res[name]["rmse"].append(float(np.sqrt(np.mean((pp - yy) ** 2))))
    return res


def current_leakfree(store, scored, imp, items, taste_from_all=False, uid=None):
    """The production model (content ridge, alpha 30, implicit rows), built
    through ItemStore. With taste_from_all the taste vector is taken from every
    rating, as the SQL path does - including the held-out ones."""
    from malrec.features import build_vocabulary, vectorise
    if taste_from_all:
        allr = query("""SELECT mal_id, score, recency_weight(finished_at, updated_at,
                         %s::real, %s::real) w FROM list_entry WHERE user_id=%s AND score>0""",
                     (CFG.recency_half_life_years, CFG.recency_floor, uid))
        taste = store.taste_vector({r["mal_id"]: float(r["score"]) for r in allr},
                                   {r["mal_id"]: float(r["w"]) for r in allr})
    else:
        taste = store.taste_vector({e.mal_id: e.score for e in scored},
                                   {e.mal_id: e.weight for e in scored})
    ex = scored + imp
    rows = store.user_rows([e.mal_id for e in ex], taste)
    by = {e.mal_id: e for e in ex}
    ex = [by[r["mal_id"]] for r in rows]
    vocab = build_vocabulary(rows)
    est = Ridge(alpha=30.0).fit(vectorise(rows, vocab), [e.score for e in ex],
                                sample_weight=[e.weight for e in ex])
    return est.predict(vectorise(store.user_rows(items, taste), vocab))


VARIANTS = {
    "current via ItemStore, taste from ALL ratings": {"current_leakfree": True, "taste_all": True},
    "current via ItemStore, taste from TRAIN only": {"current_leakfree": True},
    "stack only": {"stack_only": True, "implicit": True},
    "hierarchical (CV alpha)": {"implicit": True},
    "hierarchical (alpha 30)": {"implicit": True, "alpha": 30.0},
    "hierarchical, no implicit": {"implicit": False},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=settings().malrec_user)
    args = ap.parse_args()
    pop, stacker, store, n = fit_global()
    print(f"population model from {n} sampled users, {len(pop.items)} items")
    res = gate(args.user, pop, stacker, store, VARIANTS)
    base = res["current"]["rho"]
    print(f"\n{args.user}  (8 temporal splits, tie-corrected Spearman)")
    print(f"{'model':<48}{'rho':>8}{'nDCG@10':>9}{'RMSE':>8}   wins vs current")
    for name, r in res.items():
        w = "" if name == "current" else f"   {sum(a > b for a, b in zip(r['rho'], base))}/{len(base)}"
        print(f"{name:<48}{np.mean(r['rho']):8.4f}{np.mean(r['ndcg']):9.4f}"
              f"{np.mean(r['rmse']):8.4f}{w}")
