"""Trivial baselines next to the shipped model, on the same held-out users.

Rating prediction (rho / nDCG@10 / RMSE on the user's newest ratings):
  MAL mean     community score as the prediction
  mean+bias    user's weighted mean + population item bias (fold-in mu + b_i)
  STACK        population stack only (mode="stack")
  SHIPPED      mode="sized" (what runs in production)

Retrieval (recall@50 of future liked titles from the whole pool):
  popularity   rank the pool by MAL popularity
  MAL mean     rank by community score
  SHIPPED      rank key as in exp_final (score + relevance + scaled novelty)
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from experiments.exp_final import _Share, z_over
from experiments.exp_retrieval import load_entries, pool_ids, recall
from malrec.config import settings
from malrec.db import query
from malrec.eval import ndcg_at, spearman
from malrec.rank import novelty, relevance_bonus
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore
from malrec.recsys.scorer import novelty_scale

CFG = settings()
BUDGETS = (10, 25, 60, 150, 400)


def main(max_eval: int, seed: int):
    t0 = time.time()
    leak = query("""SELECT count(*) AS n FROM cf_user u JOIN app_user a
                     ON lower(u.name) = lower(a.mal_username)""")[0]["n"]
    print(f"app users present in the CF sample: {leak}")
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
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: [(e["mal_id"], float(e["score"]), e["at"])
                                    for e in entries[u] if e["score"] > 0] for u in B},
                          store, seed=seed)
    pool_all = pool_ids(store)
    malmean = {m: float(r.get("mal_mean") or 7.5) for m, r in store.rows.items()}
    popul = {m: float(r.get("mal_popularity") or 99999) for m, r in store.rows.items()}
    print(f"{len(users)} users ({len(A)} fit / {len(B)} stack / {len(C)} eval) [{time.time()-t0:.0f}s]")

    for budget in BUDGETS:
        M = defaultdict(lambda: defaultdict(list))
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
            listed = {e["mal_id"]: recency_weight(e["at"], cutoff) for e in seen}
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            items = [m for m, _, _ in test_s]
            truth = np.array([s for _, s, _ in test_s])

            sz = fit_user(pop, stacker, store, scored, implicit, mode="sized", listed=listed)
            st = fit_user(pop, stacker, store, scored, implicit, mode="stack", listed=listed)
            sig = st.fold.signals(items)
            preds = {
                "MAL mean": np.array([malmean.get(m, np.nan) for m in items]),
                "mean+bias": st.fold.mu + sig["bias"],
                "STACK": st.predict(items),
                "SHIPPED": sz.predict(items),
            }
            # same rows for every method: those the shipped model can score
            ok = ~np.isnan(preds["SHIPPED"])
            if ok.sum() >= 5 and truth[ok].std() > 0:
                for name, p in preds.items():
                    p = np.where(np.isnan(p), 7.5, p)
                    M[name]["rho"].append(spearman(p[ok], truth[ok]))
                    M[name]["ndcg"].append(ndcg_at(p[ok], list(truth[ok]), 10))
                    M[name]["rmse"].append(float(np.sqrt(np.mean((p[ok] - truth[ok]) ** 2))))
                M["user mean"]["rmse"].append(float(np.sqrt(np.mean((st.fold.mu - truth[ok]) ** 2))))
            if len(targets) >= 3:
                nov = np.array([novelty(store.rows[m].get("mal_popularity")) for m in pool])
                shown = np.clip(sz.predict(pool), 1, 10)
                z = z_over(sz.fold.signals(pool)["rel"])
                base = shown + relevance_bonus(_Share(sz.lam), z)
                top = np.argsort(-np.nan_to_num(base, nan=-1e9))[:600]
                key = base + CFG.novelty_weight * novelty_scale(shown[top], CFG.novelty_ref_sd) * nov
                keys = {
                    "popularity": -np.array([popul[m] for m in pool]),
                    "MAL mean": np.array([malmean[m] for m in pool]),
                    "relevance only": z,
                    "SHIPPED": key,
                }
                for name, k in keys.items():
                    r = recall(k, pool, targets, 50)
                    if r is not None:
                        M[name]["r50"].append(r)
        n = len(M["SHIPPED"]["rho"])
        print(f"\n== {budget} list entries ({n} users) [{time.time()-t0:.0f}s] ==")
        print(f"{'':<16}{'rho':>7}{'nDCG@10':>9}{'RMSE':>7}{'recall@50':>11}")
        for name in ("user mean", "MAL mean", "mean+bias", "STACK", "SHIPPED", "popularity",
                     "relevance only"):
            m = M[name]
            f = lambda k, m=m: f"{np.mean(m[k]):7.3f}" if m[k] else f"{'-':>7}"
            print(f"{name:<16}{f('rho')}{f('ndcg'):>9}{f('rmse')}{f('r50'):>11}")
    print(f"\n[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
