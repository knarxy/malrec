"""Retrieval with the product's own ranking rule, novelty included.

exp_retrieval.py showed that adding co-occurrence relevance to the ranking
roughly triples recall. This checks the version that would actually ship:

    rank key = shown score + beta * relevance_z + novelty bonus

`shown` is the calibrated prediction (per-user calibration for the app user,
identity for sampled users, as production does below ~85 ratings). The
novelty bonus is the product's balanced setting, scaled by prediction spread.
Reported per ranking: recall, how personal the lists are (overlap), and how
much discovery they offer (median popularity rank of the top 20).
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids, recall, zscore
from malrec.config import settings
from malrec.db import one, query
from malrec.rank import novelty
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore
from malrec.recsys.scorer import novelty_scale

CFG = settings()
BUDGETS = (10, 25, 60, 400)


def keys(pop, stacker, store, scored, implicit, listed, pool, slope=1.0, intercept=0.0):
    """Ranking keys for each strategy, in display units."""
    nov = np.array([novelty(store.rows[m].get("mal_popularity")) for m in pool])
    fold = pop.fold_in({e.mal_id: e.score for e in scored},
                       {e.mal_id: e.weight for e in scored}, listed=listed)
    rel = zscore(fold.signals(pool)["rel"])

    def shown(mode):
        um = fit_user(pop, stacker, store, scored, implicit, mode=mode)
        um.fold = fold
        if um.inner is not None:
            um.inner.fold = fold
        raw = um.predict(pool)
        return np.clip(slope * raw + intercept, 1, 10) if mode != "stack" else raw

    def with_novelty(base):
        top = np.argsort(-np.nan_to_num(base, nan=-1e9))[:600]
        scale = novelty_scale(base[top], CFG.novelty_ref_sd)
        return base + CFG.novelty_weight * scale * nov

    personal = shown("personal")
    sized = shown("sized")
    out = {
        "TODAY: personal + novelty": with_novelty(personal),
        "personal (no novelty)": personal,
    }
    for b in (0.5, 1.0):
        out[f"sized + {b:g}rel (no novelty)"] = sized + b * rel
        out[f"sized + {b:g}rel + novelty"] = with_novelty(sized + b * rel)
        out[f"personal + {b:g}rel + novelty"] = with_novelty(personal + b * rel)
    return out


def summarise(title, res, tops, pops):
    print(f"\n== {title} ==")
    print(f"{'ranking':<34}{'recall@50':>10}{'recall@100':>11}{'overlap':>9}{'median pop#':>12}")
    for name, m in res.items():
        sets = tops.get(name, [])
        pairs = list(itertools.islice(itertools.combinations(range(len(sets)), 2), 3000))
        jac = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs]) \
            if pairs else float("nan")
        print(f"{name:<34}{np.mean(m['r50']):10.3f}{np.mean(m['r100']):11.3f}{jac:9.3f}"
              f"{np.median(pops[name]):12.0f}")


def main(max_eval: int, seed: int):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    entries = load_entries()
    users = sorted(u for u, es in entries.items() if sum(e["score"] > 0 for e in es) >= 15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.70), int(len(users) * 0.15)
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
    print(f"{len(users)} users; pool {len(pool_all)} [{time.time() - t0:.0f}s]")

    def pop_rank(sc, pool):
        s = np.where(np.isnan(sc), -np.inf, sc)
        return [store.rows[pool[i]].get("mal_popularity") or 99999 for i in np.argsort(-s)[:20]]

    for budget in BUDGETS:
        res = defaultdict(lambda: {"r50": [], "r100": []})
        tops, pops = defaultdict(list), defaultdict(list)
        for u in C[:max_eval]:
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, rng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            targets = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            if len(targets) < 3:
                continue
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            seen = [history[i] for i in rng.permutation(len(history))[:budget]]
            scored = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff), e["at"])
                      for e in seen if e["score"] > 0]
            if len(scored) < 3:
                continue
            m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
            implicit = [Example(e["mal_id"], float(min(max(m_ + CFG.implicit_offsets[e["status"]], 1), 10)),
                                CFG.implicit_weight * recency_weight(e["at"], cutoff), e["at"])
                        for e in seen if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
            listed = {e["mal_id"]: recency_weight(e["at"], cutoff) for e in seen}
            for name, sc in keys(pop, stacker, store, scored, implicit, listed, pool).items():
                r50 = recall(sc, pool, targets, 50)
                if r50 is None:
                    continue
                res[name]["r50"].append(r50)
                res[name]["r100"].append(recall(sc, pool, targets, 100))
                s = np.where(np.isnan(sc), -np.inf, sc)
                tops[name].append({pool[i] for i in np.argsort(-s)[:20]})
                pops[name].extend(pop_rank(sc, pool))
        if res:
            summarise(f"sampled users with {budget} list entries "
                      f"({len(res['personal (no novelty)']['r50'])} users)", res, tops, pops)

    # reference profile with its production calibration
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    cal = one("SELECT metrics->'calibration' c FROM model_run WHERE user_id=%s ORDER BY id DESC LIMIT 1",
              (uid,))["c"] or {}
    slope, icpt = float(cal.get("slope", 1.0)), float(cal.get("intercept", 0.0))
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    res = defaultdict(lambda: {"r50": [], "r100": []})
    pops = defaultdict(list)
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
        implicit = [Example(r["mal_id"], float(min(max(m_ + CFG.implicit_offsets[r["status"]], 1), 10)),
                            CFG.implicit_weight * recency_weight(r["at"], cutoff), r["at"])
                    for r in hist if r["score"] == 0 and r["status"] in CFG.implicit_offsets]
        listed = {r["mal_id"]: recency_weight(r["at"], cutoff) for r in hist}
        for name, sc in keys(pop, stacker, store, scored, implicit, listed, pool, slope, icpt).items():
            r50 = recall(sc, pool, targets, 50)
            if r50 is not None:
                res[name]["r50"].append(r50)
                res[name]["r100"].append(recall(sc, pool, targets, 100))
                pops[name].extend(pop_rank(sc, pool))
    summarise(f"reference profile, 8 temporal splits (calibration slope {slope:.2f})", res, {}, pops)
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
