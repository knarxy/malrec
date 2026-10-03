"""Answers to the 2026-09 audit, on the deployed configuration.

Built on exp_baselines.py (the audit's script), but measuring what runs:
EASE in the population model, relevance dial 0.5, Safe Bets (no novelty,
long-memory mix 0.5) and Discover (novelty, minus Safe Bets' 60) as separate
lists.

Rating order, per list size (rho / RMSE on newest ratings):
  MAL mean, mean+bias, STACK, SHIPPED (size-weighted blend),
  LATE (same blend, handover centred at 150 instead of 80),
  CHOSEN (blend weight picked on the user's own newest training ratings -
          the existing "blend" mode with the personal model)

Retrieval, recall@50 of liked future titles, plain and inverse-propensity
weighted (IPS). People mostly watch popular titles, so plain recall rewards
popularity; IPS weights each liked title by 1 / P(watched), estimated as the
share of sampled lists containing it, which asks "did we find what they liked,
however obscure" (Schnabel et al. 2016, self-normalised):
  popularity, relevance only, SAFE, SAFE+POP(beta) = Safe Bets with a
  popularity term beta * (1 - personal share) * z(-log popularity), DISCOVER,
  and SAFE u DISCOVER.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from experiments.exp_signals import CFG, keyed, listed_weights
from malrec.config import settings
from malrec.db import one, query
from malrec.eval import spearman
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import (
    Example,
    fit_stacker,
    fit_user,
    recency_weight,
    size_weight,
    split_user,
)
from malrec.recsys.items import ItemStore

BUDGETS = (10, 25, 60, 150, 400)
BETAS = (0.5, 1.0)


def z(x):
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x))
    return (x - np.nanmean(x)) / sd if sd > 1e-9 else np.zeros_like(x)


def recalls(key, pool, targets, prop, k=50):
    s = np.where(np.isnan(key), -np.inf, key)
    top = {pool[i] for i in np.argsort(-s)[:k]}
    return _rec(top, targets, prop)


def _rec(top, targets, prop):
    if not targets:
        return None, None
    hit = top & targets
    w = {t: 1.0 / prop.get(t, 1e-3) for t in targets}
    return len(hit) / len(targets), sum(w[t] for t in hit) / sum(w.values())


def evaluate(pop, stacker, store, scored, implicit, listed_fn, pool, items, truth, targets,
             prop, popul, malmean, cutoff, n_hist, base=(), cal=(), size_n=None):
    R, Q = {}, {}
    listed = listed_fn()
    kw = {"listed": listed, "size_n": size_n}
    sz = fit_user(pop, stacker, store, scored, implicit, mode="sized", **kw)
    st = fit_user(pop, stacker, store, scored, implicit, mode="stack", **kw)
    late = fit_user(pop, stacker, store, scored, implicit, mode="sized", size_n0=150.0, **kw)
    ch = fit_user(pop, stacker, store, scored, implicit, mode="blend", blend_of="personal",
                  listed=listed)
    sig = st.fold.signals(items)
    a, b = cal if cal else (1.0, 0.0)
    for name, p in {"MAL mean": np.array([malmean.get(m, np.nan) for m in items]),
                    "mean+bias": st.fold.mu + sig["bias"], "STACK": st.predict(items),
                    "SHIPPED": sz.predict(items), "LATE": late.predict(items),
                    "CHOSEN": ch.predict(items)}.items():
        if name not in ("MAL mean",):
            p = a * p + b
        R[name] = p
    Q["lam"] = {"SHIPPED": sz.lam, "LATE": late.lam, "CHOSEN": ch.lam}
    # retrieval on the deployed lists
    if targets:
        nw = CFG.novelty_weight
        CFG.novelty_weight = 0.0
        safe, _ = keyed(pop, stacker, store, scored, implicit, listed_fn, pool, items,
                        base + ("mix:0.5",), cutoff, *cal, size_n=size_n)
        CFG.novelty_weight = nw
        disc, _ = keyed(pop, stacker, store, scored, implicit, listed_fn, pool, items, base,
                        cutoff, *cal, size_n=size_n)
        lam = size_weight(size_n or len(scored), 80.0, 15.0)
        popz = z([-math.log10(popul.get(m, 99999)) for m in pool])
        keys = {"popularity": np.array([-popul.get(m, 99999) for m in pool], dtype=float),
                "relevance only": sz.relevance(pool), "SAFE": safe}
        for beta in BETAS:
            keys[f"SAFE+POP {beta}"] = z(safe) + beta * (1 - lam) * popz
        for name, k in keys.items():
            Q[name] = recalls(k, pool, targets, prop)
        s = np.where(np.isnan(safe), -np.inf, safe)
        safe60 = np.argsort(-s)[:60]
        d = np.where(np.isnan(disc), -np.inf, disc).copy()
        d[safe60] = -np.inf
        top_d = {pool[i] for i in np.argsort(-d)[:50]}
        Q["DISCOVER"] = _rec(top_d, targets, prop)
        Q["SAFE u DISC"] = _rec(top_d | {pool[i] for i in safe60}, targets, prop)
    return R, Q


def report(title, rows, ret):
    print(f"\n== {title} ==")
    print(f"{'rating':<12}{'rho':>7}{'RMSE':>7}{'lam':>6}")
    for n, v in rows.items():
        lam = f"{np.mean(v['lam']):6.2f}" if v.get("lam") else f"{'':>6}"
        print(f"{n:<12}{np.mean(v['rho']):7.3f}{np.mean(v['rmse']):7.3f}{lam}")
    print(f"{'retrieval':<14}{'recall@50':>10}{'IPS':>8}")
    for n, v in ret.items():
        print(f"{n:<14}{np.mean(v['plain']):10.3f}{np.mean(v['ips']):8.3f}")


def collect(R, Q, truth, rows, ret):
    ok = ~np.isnan(R["SHIPPED"])
    if ok.sum() >= 5 and truth[ok].std() > 0:
        for n, p in R.items():
            p = np.where(np.isnan(p), 7.5, p)
            rows[n]["rho"].append(spearman(p[ok], truth[ok]))
            rows[n]["rmse"].append(float(np.sqrt(np.mean((p[ok] - truth[ok]) ** 2))))
            if n in Q["lam"]:
                rows[n]["lam"].append(Q["lam"][n])
    for n, v in Q.items():
        if n != "lam" and v and v[0] is not None:
            ret[n]["plain"].append(v[0]); ret[n]["ips"].append(v[1])


def main(max_eval: int, seed: int):
    t0 = time.time()
    CFG.relevance_weight_personal = 0.5
    rng = np.random.default_rng(seed)
    entries = load_entries()
    users = sorted(u for u, es in entries.items() if sum(e["score"] > 0 for e in es) >= 15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    ua, ia, ra, oa, oi = [], [], [], [], []
    for u in A:
        for e in entries[u]:
            oa.append(u); oi.append(e["mal_id"])
            if e["score"] > 0:
                ua.append(u); ia.append(e["mal_id"]); ra.append(float(e["score"]))
    pop = PopulationModel(CFParams()).fit(np.array(ua), np.array(ia), np.array(ra))
    pop.fit_occurrence(np.array(oa), np.array(oi))
    pop.fit_ease(np.array(oa), np.array(oi))
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: [(e["mal_id"], float(e["score"]), e["at"])
                                    for e in entries[u] if e["score"] > 0] for u in B},
                          store, seed=seed)
    # propensity: share of fit users' lists containing the title
    lists = defaultdict(set)
    for u_, m_ in zip(oa, oi):
        lists[m_].add(u_)
    prop = {m: (len(s) + 1) / (len(A) + 2) for m, s in lists.items()}
    pool_all = pool_ids(store)
    malmean = {m: float(r.get("mal_mean") or 7.5) for m, r in store.rows.items()}
    popul = {m: float(r.get("mal_popularity") or 99999) for m, r in store.rows.items()}
    print(f"{len(users)} users; EASE on; dial 0.5 [{time.time() - t0:.0f}s]", flush=True)

    for budget in BUDGETS:
        rows, ret = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for u in C[:max_eval]:
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, rng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            targets = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in rng.permutation(len(history))[:budget]]
            scored = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff),
                              e["at"]) for e in seen if e["score"] > 0]
            if len(scored) < 3:
                continue
            m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
            implicit = [Example(e["mal_id"], float(min(max(m_ + CFG.implicit_offsets[e["status"]],
                                                            1), 10)),
                                CFG.implicit_weight * recency_weight(e["at"], cutoff), e["at"])
                        for e in seen if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            items = [m for m, _, _ in test_s]
            truth = np.array([s for _, s, _ in test_s])
            R, Q = evaluate(pop, stacker, store, scored, implicit,
                            lambda s_=seen, c=cutoff: listed_weights(s_, c, 0.0), pool, items,
                            truth, targets if len(targets) >= 3 else set(), prop, popul,
                            malmean, cutoff, len(history))
            collect(R, Q, truth, rows, ret)
        report(f"sampled users with {budget} list entries ({len(rows['SHIPPED']['rho'])} users)"
               f" [{time.time() - t0:.0f}s]", rows, ret)

    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    cal = one("SELECT metrics->'calibration' c FROM model_run WHERE user_id=%s"
              " AND metrics ? 'calibration' ORDER BY id DESC LIMIT 1", (uid,))["c"]
    cal = (float(cal["slope"]), float(cal["intercept"]))
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    CFG.recency_half_life_years = 0.35
    rows, ret = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    per = defaultdict(list)
    for h in (25, 30, 35, 40, 45, 50, 55, 60):
        test = rated[-h:]
        cutoff = test[0]["at"]
        targets = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        test_ids = {r["mal_id"] for r in test}
        pool = [m for m in pool_all if m not in {r["mal_id"] for r in le} - test_ids]
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        scored = [Example(r["mal_id"], float(r["score"]), recency_weight(r["at"], cutoff), r["at"])
                  for r in hist if r["score"] > 0]
        m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
        implicit = [Example(r["mal_id"], float(min(max(m_ + CFG.implicit_offsets[r["status"]], 1),
                                                    10)),
                            CFG.implicit_weight * recency_weight(r["at"], cutoff), r["at"])
                    for r in hist if r["score"] == 0 and r["status"] in CFG.implicit_offsets]
        items = [r["mal_id"] for r in test]
        truth = np.array([float(r["score"]) for r in test])
        R, Q = evaluate(pop, stacker, store, scored, implicit,
                        lambda h_=hist, c=cutoff: listed_weights(h_, c, 0.0), pool, items, truth,
                        targets, prop, popul, malmean, cutoff, len(hist),
                        base=("hl:0.35",), cal=cal, size_n=len(rated))
        collect(R, Q, truth, rows, ret)
        for n, p in R.items():
            per[n].append(round(spearman(np.where(np.isnan(p), 7.5, p), truth), 4))
    report("reference profile, 8 temporal splits (half-life 0.35)", rows, ret)
    for n, v in per.items():
        print(f"  {n:<10} per split: {v}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
