"""Final validation: the configuration that would ship vs what runs today.

TODAY    per-user content ridge; rank = predicted score + novelty
SHIPPED  size-weighted blend of personal model and population stack;
         rank = predicted score + rank.relevance_bonus() + scaled novelty

Both are evaluated through the production functions (recsys.hybrid.fit_user,
rank.relevance_bonus, recsys.scorer.novelty_scale) with the current config,
so what is measured is what ships.

  sampled users  held out from the population model; by list size:
                 rating prediction on their future ratings, and retrieval of
                 the future titles they liked
  reference profile     8 temporal splits. The blend weight is the one production
                 applies at his real list size (size_n), since the gate is
                 about his live experience; the weight implied by the
                 truncated window is reported too.
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids, recall
from malrec.config import settings
from malrec.db import one, query
from malrec.eval import ndcg_at, spearman
from malrec.rank import novelty, relevance_bonus
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore
from malrec.recsys.scorer import novelty_scale

CFG = settings()
BUDGETS = (10, 25, 60, 150, 400)


class _Share:
    """Duck-type for relevance_bonus: only personal_share is read."""
    def __init__(self, lam):
        self.personal_share = lam


def z_over(x):
    sd = float(np.nanstd(x))
    return (x - np.nanmean(x)) / sd if sd > 1e-9 else np.zeros_like(x)


def evaluate(pop, stacker, store, scored, implicit, listed, pool, test_items, slope=1.0,
             icpt=0.0, size_n=None, rwps=()):
    """Returns {name: (rank_key over pool, predictions for test items)}."""
    nov = np.array([novelty(store.rows[m].get("mal_popularity")) for m in pool])
    out = {}
    # today
    per = fit_user(pop, stacker, store, scored, implicit, mode="personal", listed=listed)
    shown = np.clip(slope * per.predict(pool) + icpt, 1, 10)
    out["TODAY"] = (shown + CFG.novelty_weight * nov,
                    np.clip(slope * per.predict(test_items) + icpt, 1, 10))
    # shipped
    sz = fit_user(pop, stacker, store, scored, implicit, mode="sized", listed=listed,
                  size_n=size_n)
    shown = np.clip(slope * sz.predict(pool) + icpt, 1, 10)
    z = z_over(sz.fold.signals(pool)["rel"])
    pred_test = np.clip(slope * sz.predict(test_items) + icpt, 1, 10)
    for name, rwp in (("SHIPPED", None), *((f"dial {r:g}", r) for r in rwps)):
        base = shown + relevance_bonus(_Share(sz.lam), z, personal_weight=rwp)
        top = np.argsort(-np.nan_to_num(base, nan=-1e9))[:600]
        key = base + CFG.novelty_weight * novelty_scale(shown[top], CFG.novelty_ref_sd) * nov
        out[name] = (key, pred_test)
    out["_lam"] = sz.lam
    return out


def main(max_eval: int, seed: int, rwps: tuple[float, ...] = (), skip_sampled: bool = False):
    t0 = time.time()
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
    print(f"{len(users)} sampled users ({len(A)} fit / {len(B)} stack / {len(C)} eval), "
          f"pool {len(pool_all)} [{time.time() - t0:.0f}s]")
    print(f"config: relevance {CFG.relevance_weight} / personal {CFG.relevance_weight_personal},"
          f" novelty {CFG.novelty_weight} (ref sd {CFG.novelty_ref_sd})")

    def pops(key, pool):
        s = np.where(np.isnan(key), -np.inf, key)
        return [store.rows[pool[i]].get("mal_popularity") or 99999 for i in np.argsort(-s)[:20]]

    def top20(key, pool):
        s = np.where(np.isnan(key), -np.inf, key)
        return {pool[i] for i in np.argsort(-s)[:20]}

    for budget in () if skip_sampled else BUDGETS:
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
            test_items = [m for m, _, _ in test_s]
            truth = np.array([s for _, s, _ in test_s])
            res = evaluate(pop, stacker, store, scored, implicit, listed, pool, test_items)
            for name in ("TODAY", "SHIPPED"):
                key, pred = res[name]
                ok = ~np.isnan(pred)
                if ok.sum() >= 5 and truth[ok].std() > 0:
                    M[name]["rho"].append(spearman(pred[ok], truth[ok]))
                    M[name]["ndcg"].append(ndcg_at(pred[ok], list(truth[ok]), 10))
                    M[name]["rmse"].append(float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2))))
                if len(targets) >= 3:
                    r = recall(key, pool, targets, 50)
                    if r is not None:
                        M[name]["r50"].append(r)
                        M[name]["r100"].append(recall(key, pool, targets, 100))
                M[name]["pop"].extend(pops(key, pool))
                M[name]["tops"].append(top20(key, pool))
        if not M:
            continue
        print(f"\n== sampled users with {budget} list entries ({len(M['TODAY']['tops'])} users) ==")
        print(f"{'':<10}{'rho':>7}{'nDCG@10':>9}{'RMSE':>7}{'recall@50':>11}{'recall@100':>11}"
              f"{'overlap':>9}{'med pop#':>10}")
        for name in ("TODAY", "SHIPPED"):
            m = M[name]
            sets = m["tops"]
            pairs = list(itertools.islice(itertools.combinations(range(len(sets)), 2), 3000))
            jac = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs])
            print(f"{name:<10}{np.mean(m['rho']):7.3f}{np.mean(m['ndcg']):9.3f}"
                  f"{np.mean(m['rmse']):7.3f}{np.mean(m['r50']):11.3f}{np.mean(m['r100']):11.3f}"
                  f"{jac:9.3f}{np.median(m['pop']):10.0f}")

    # ------------------------------------------------------------ dennis --
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    cal = one("SELECT metrics->'calibration' c FROM model_run WHERE user_id=%s"
              " AND metrics ? 'calibration' ORDER BY id DESC LIMIT 1", (uid,))
    cal = (cal or {}).get("c") or {}
    slope, icpt = float(cal.get("slope", 1.0)), float(cal.get("intercept", 0.0))
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    n_full = len(rated)
    mean = float(np.mean([r["score"] for r in rated]))
    names = ("TODAY", "SHIPPED", *(f"dial {r:g}" for r in rwps))
    for label, size_n in ((f"at its real size (n={n_full})", n_full),
                          ("at the truncated window's size", None)):
        G = defaultdict(lambda: defaultdict(list))
        lams = []
        for h in (25, 30, 35, 40, 45, 50, 55, 60):
            test = rated[-h:]
            cutoff = test[0]["at"]
            targets = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
            test_ids = {r["mal_id"] for r in test}
            pool = [m for m in pool_all if m not in {r["mal_id"] for r in le} - test_ids]
            hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
            scored = [Example(r["mal_id"], float(r["score"]), recency_weight(r["at"], cutoff),
                              r["at"]) for r in hist if r["score"] > 0]
            m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
            implicit = [Example(r["mal_id"],
                                float(min(max(m_ + CFG.implicit_offsets[r["status"]], 1), 10)),
                                CFG.implicit_weight * recency_weight(r["at"], cutoff), r["at"])
                        for r in hist if r["score"] == 0 and r["status"] in CFG.implicit_offsets]
            listed = {r["mal_id"]: recency_weight(r["at"], cutoff) for r in hist}
            items = [r["mal_id"] for r in test]
            truth = np.array([float(r["score"]) for r in test])
            res = evaluate(pop, stacker, store, scored, implicit, listed, pool, items,
                           slope, icpt, size_n=size_n, rwps=rwps)
            lams.append(res["_lam"])
            for name in names:
                key, pred = res[name]
                G[name]["rho"].append(spearman(pred, truth))
                G[name]["ndcg"].append(ndcg_at(pred, list(truth), 10))
                G[name]["rmse"].append(float(np.sqrt(np.mean((pred - truth) ** 2))))
                G[name]["r20"].append(recall(key, pool, targets, 20) or 0.0)
                G[name]["r50"].append(recall(key, pool, targets, 50) or 0.0)
                G[name]["r100"].append(recall(key, pool, targets, 100) or 0.0)
                G[name]["pop"].extend(pops(key, pool))
        base = G["TODAY"]["rho"]
        print(f"\n== reference profile, 8 temporal splits, blend weight {label}: "
              f"personal share {np.mean(lams):.2f} ==")
        print(f"{'':<10}{'rho':>7}{'nDCG@10':>9}{'RMSE':>7}{'recall@20':>11}{'recall@50':>11}"
              f"{'recall@100':>11}{'med pop#':>10}   rho wins")
        for name in names:
            g = G[name]
            w = "" if name == "TODAY" else \
                f"   {sum(a > b + 1e-9 for a, b in zip(g['rho'], base))}/8 better, " \
                f"{sum(abs(a - b) <= 1e-9 for a, b in zip(g['rho'], base))}/8 equal"
            print(f"{name:<10}{np.mean(g['rho']):7.4f}{np.mean(g['ndcg']):9.4f}{np.mean(g['rmse']):7.3f}"
                  f"{np.mean(g['r20']):11.3f}{np.mean(g['r50']):11.3f}{np.mean(g['r100']):11.3f}{np.median(g['pop']):10.0f}{w}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dial", default="", help="comma-separated relevance_weight_personal values")
    ap.add_argument("--skip-sampled", action="store_true")
    a = ap.parse_args()
    main(a.max_eval, a.seed, tuple(float(x) for x in a.dial.split(",") if x), a.skip_sampled)
