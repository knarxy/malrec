"""Long memory in Safe Bets only: what happens to both tabs together?

Safe Bets = predicted score + relevance, no novelty; optionally its order
blends in a no-decay model's key (weight w). Discover = today's key with
novelty, over the pool minus Safe Bets' 60 titles (as production builds it).
Reported per variant: recall@50 and top-20 median popularity of each tab, and
the share of liked future titles caught by either (Safe 60 + Discover 50).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from experiments.exp_signals import CFG, keyed, listed_weights
from malrec.config import settings
from malrec.db import one, query
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, recency_weight, split_user
from malrec.recsys.items import ItemStore

WEIGHTS = (0.0, 0.5, 0.7)


def tabs(pop, stacker, store, scored, implicit, listed_fn, pool, test_items, cutoff, targets,
         *cal, size_n=None, base=()):
    out = {}
    nw = CFG.novelty_weight
    CFG.novelty_weight = 1.0
    disc_key, _ = keyed(pop, stacker, store, scored, implicit, listed_fn, pool, test_items, base,
                        cutoff, *cal, size_n=size_n)
    for w in WEIGHTS:
        CFG.novelty_weight = 0.0
        flags = base + ((f"mix:{w}",) if w else ())
        safe_key, _ = keyed(pop, stacker, store, scored, implicit, listed_fn, pool, test_items,
                            flags, cutoff, *cal, size_n=size_n)
        s = np.where(np.isnan(safe_key), -np.inf, safe_key)
        safe60 = np.argsort(-s)[:60]
        d = np.where(np.isnan(disc_key), -np.inf, disc_key).copy()
        d[safe60] = -np.inf
        disc50 = np.argsort(-d)[:50]
        ids_s = {pool[i] for i in safe60[:50]}
        ids_d = {pool[i] for i in disc50}
        both = {pool[i] for i in safe60} | ids_d
        n = max(len(targets), 1)
        out[f"safe mix={w}"] = {
            "safe_r50": len(ids_s & targets) / n, "disc_r50": len(ids_d & targets) / n,
            "union": len(both & targets) / n,
            "safe_pop": float(np.median([store.rows[pool[i]].get("mal_popularity") or 99999
                                         for i in safe60[:20]])),
            "disc_pop": float(np.median([store.rows[pool[i]].get("mal_popularity") or 99999
                                         for i in disc50[:20]]))}
    CFG.novelty_weight = nw
    return out


def show(title, M):
    print(f"\n== {title} ==")
    print(f"{'':<14}{'safe r@50':>10}{'safe pop':>9}{'disc r@50':>10}{'disc pop':>9}{'union':>8}")
    for k, v in M.items():
        print(f"{k:<14}{np.mean(v['safe_r50']):10.3f}{np.median(v['safe_pop']):9.0f}"
              f"{np.mean(v['disc_r50']):10.3f}{np.median(v['disc_pop']):9.0f}"
              f"{np.mean(v['union']):8.3f}")


def main(max_eval: int, seed: int):
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
    pop.fit_ease(np.array(oa), np.array(oi))          # as deployed
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: [(e["mal_id"], float(e["score"]), e["at"])
                                    for e in entries[u] if e["score"] > 0] for u in B},
                          store, seed=seed)
    pool_all = pool_ids(store)
    for budget in (25, 150, 400):
        M = defaultdict(lambda: defaultdict(list))
        for u in C[:max_eval]:
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, rng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            targets = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget or len(targets) < 3:
                continue
            seen = [history[i] for i in rng.permutation(len(history))[:budget]]
            scored = [Example(e["mal_id"], float(e["score"]), 1.0, e["at"])
                      for e in seen if e["score"] > 0]
            if len(scored) < 3:
                continue
            implicit = [Example(e["mal_id"], 0.0, 1.0, e["at"]) for e in seen
                        if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
            m_ = float(np.mean([x.score for x in scored]))
            implicit = [Example(x.mal_id, float(min(max(m_ + CFG.implicit_offsets[e["status"]], 1),
                                                        10)), 1.0, x.at)
                        for x, e in zip(implicit, [e for e in seen if e["score"] == 0
                                                   and e["status"] in CFG.implicit_offsets])]
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            res = tabs(pop, stacker, store, scored, implicit,
                       lambda s_=seen, c=cutoff: listed_weights(s_, c, 0.0), pool,
                       [m for m, _, _ in test_s], cutoff, targets)
            for k, v in res.items():
                for kk, vv in v.items():
                    M[k][kk].append(vv)
        show(f"sampled users with {budget} list entries ({len(M['safe mix=0.0']['union'])} users)", M)

    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    G = defaultdict(lambda: defaultdict(list))
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
                                                    10)), CFG.implicit_weight, r["at"])
                    for r in hist if r["score"] == 0 and r["status"] in CFG.implicit_offsets]
        res = tabs(pop, stacker, store, scored, implicit,
                   lambda h_=hist, c=cutoff: listed_weights(h_, c, 0.0), pool,
                   [r["mal_id"] for r in test], cutoff, targets, size_n=len(rated),
                   base=("hl:0.35",))                 # his per-user choice
        for k, v in res.items():
            for kk, vv in v.items():
                G[k][kk].append(vv)
    show("reference profile, 8 temporal splits (half-life 0.35, dial 0.5, EASE)", G)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
