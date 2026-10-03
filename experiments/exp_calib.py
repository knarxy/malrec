"""Genre balance: today's overlap penalty vs calibrated re-ranking (Steck 2018).

Both lists as deployed (Safe Bets: no novelty, long-memory mix; Discover:
novelty, minus Safe Bets' 60). The 50-item list is then built from the top
300 candidates by
  penalty    today's rule: key - 0.07 x (mean count of the item's genres
             already in the list)          (diversity_weight 0.07 / 0.10)
  calib l    Steck's greedy selection: maximise
             (1 - l) * sum of min-max relevance - l * KL(p || q~)
             p = the user's genre mix (recency-weighted list), q~ = the list's
             mix smoothed with p (alpha 0.01)
Reported: recall@50 of liked future titles (plain and IPS), KL(p || q) of the
list, and distinct genres in it.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from experiments.exp_signals import CFG, keyed, listed_weights
from malrec.config import settings
from malrec.db import one, query
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, split_user
from malrec.recsys.items import ItemStore

TOP, N, ALPHA = 300, 50, 0.01
LAMBDAS = (0.3, 0.5, 0.7)


def genre_mix(ids, weights, genres) -> dict[str, float]:
    mix: Counter = Counter()
    for m, w in zip(ids, weights):
        gs = genres.get(m) or []
        for g in gs:
            mix[g] += w / len(gs)
    tot = sum(mix.values()) or 1.0
    return {g: v / tot for g, v in mix.items()}


def kl(p, q_counts, alpha=ALPHA):
    tot = sum(q_counts.values()) or 1.0
    return sum(pv * math.log(pv / ((1 - alpha) * q_counts.get(g, 0.0) / tot + alpha * pv))
               for g, pv in p.items() if pv > 0)


def penalty_list(key, cand, genres, dw):
    counts: Counter = Counter()
    picked = []
    order = sorted(cand, key=lambda i: -key[i])
    for i in order:
        gs = genres.get(i) or []
        over = (sum(counts[g] for g in gs) / len(gs)) if gs else 0.0
        picked.append((key[i] - dw * over, i))
        for g in gs:
            counts[g] += 1
        if len(picked) >= N * 3:
            break
    return [i for _, i in sorted(picked, reverse=True)[:N]]


def calib_list(key, cand, genres, p, lam):
    """Steck's greedy selection, vectorised over the remaining candidates."""
    gl = sorted({g for i in cand for g in (genres.get(i) or [])} | set(p))
    gi = {g: k for k, g in enumerate(gl)}
    Gm = np.zeros((len(cand), len(gl)))
    for r, i in enumerate(cand):
        gs = genres.get(i) or []
        for g in gs:
            Gm[r, gi[g]] = 1.0 / len(gs)
    pv = np.array([p.get(g, 0.0) for g in gl])
    mask = pv > 0
    ks = np.array([key[i] for i in cand])
    rel = (ks - ks.min()) / (ks.max() - ks.min() + 1e-9)
    counts = np.zeros(len(gl))
    left = np.ones(len(cand), bool)
    picked, total = [], 0.0
    for _ in range(min(N, len(cand))):
        idx = np.nonzero(left)[0]
        C2 = counts + Gm[idx]
        tot = C2.sum(1, keepdims=True)
        q = np.divide(C2, tot, out=np.zeros_like(C2), where=tot > 0)
        qt = (1 - ALPHA) * q + ALPHA * pv
        klv = (pv[mask] * np.log(pv[mask] / qt[:, mask])).sum(1)
        v = (1 - lam) * (total + rel[idx]) - lam * klv
        b = idx[int(np.argmax(v))]
        picked.append(cand[b]); left[b] = False
        total += rel[b]; counts += Gm[b]
    return picked


def score(lst, targets, prop, p, genres):
    hit = set(lst) & targets
    w = {t: 1 / prop.get(t, 1e-3) for t in targets}
    counts: Counter = Counter()
    for i in lst:
        gs = genres.get(i) or []
        for g in gs:
            counts[g] += 1.0 / len(gs)
    return (len(hit) / len(targets), sum(w[t] for t in hit) / sum(w.values()),
            kl(p, counts, 0.0001), len(counts))


def run_user(pop, stacker, store, scored, implicit, listed_fn, pool, items, cutoff, targets,
             prop, genres, base=(), cal=(), size_n=None):
    listed = listed_fn()
    p = genre_mix(list(listed), list(listed.values()), genres)
    nw = CFG.novelty_weight
    CFG.novelty_weight = 0.0
    safe, _ = keyed(pop, stacker, store, scored, implicit, listed_fn, pool, items,
                    base + ("mix:0.5",), cutoff, *cal, size_n=size_n)
    CFG.novelty_weight = nw
    disc, _ = keyed(pop, stacker, store, scored, implicit, listed_fn, pool, items, base, cutoff,
                    *cal, size_n=size_n)
    out = {}
    s = np.where(np.isnan(safe), -np.inf, safe)
    safe_ids = [pool[i] for i in np.argsort(-s)[:TOP]]
    d = np.where(np.isnan(disc), -np.inf, disc).copy()
    d[np.argsort(-s)[:60]] = -np.inf
    disc_ids = [pool[i] for i in np.argsort(-d)[:TOP]]
    for tab, ids, keyarr, dw in (("safe", safe_ids, safe, 0.10), ("disc", disc_ids, disc, 0.07)):
        key = {pool[i]: keyarr[i] for i in range(len(pool))}
        out[f"{tab} penalty"] = score(penalty_list(key, ids, genres, dw), targets, prop, p, genres)
        for lam in LAMBDAS:
            out[f"{tab} calib {lam}"] = score(calib_list(key, ids, genres, p, lam), targets, prop,
                                              p, genres)
    return out


def show(title, M):
    print(f"\n== {title} ==")
    print(f"{'':<16}{'recall@50':>10}{'IPS':>7}{'KL':>7}{'genres':>8}")
    for k, v in M.items():
        a = np.array(v)
        print(f"{k:<16}{a[:, 0].mean():10.3f}{a[:, 1].mean():7.3f}{a[:, 2].mean():7.3f}"
              f"{a[:, 3].mean():8.1f}")


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
    pop.fit_ease(np.array(oa), np.array(oi))
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: [(e["mal_id"], float(e["score"]), e["at"])
                                    for e in entries[u] if e["score"] > 0] for u in B},
                          store, seed=seed)
    genres = {m: (r.get("mal_genres") or []) for m, r in store.rows.items()}
    lists = defaultdict(set)
    for u_, m_ in zip(oa, oi):
        lists[m_].add(u_)
    prop = {m: (len(s) + 1) / (len(A) + 2) for m, s in lists.items()}
    pool_all = pool_ids(store)
    for budget in (25, 150, 400):
        M = defaultdict(list)
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
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            res = run_user(pop, stacker, store, scored, [],
                           lambda s_=seen, c=cutoff: listed_weights(s_, c, 0.0), pool,
                           [m for m, _, _ in test_s], cutoff, targets, prop, genres)
            for k, v in res.items():
                M[k].append(v)
        show(f"sampled users with {budget} list entries ({len(M['safe penalty'])} users)", M)

    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    M = defaultdict(list)
    for h in (25, 30, 35, 40, 45, 50, 55, 60):
        test = rated[-h:]
        cutoff = test[0]["at"]
        targets = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        test_ids = {r["mal_id"] for r in test}
        pool = [m for m in pool_all if m not in {r["mal_id"] for r in le} - test_ids]
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        scored = [Example(r["mal_id"], float(r["score"]), 1.0, r["at"])
                  for r in hist if r["score"] > 0]
        res = run_user(pop, stacker, store, scored, [],
                       lambda h_=hist, c=cutoff: listed_weights(h_, c, 0.0), pool,
                       [r["mal_id"] for r in test], cutoff, targets, prop, genres,
                       base=("hl:0.35",), size_n=len(rated))
        for k, v in res.items():
            M[k].append(v)
    show("reference profile, 8 temporal splits", M)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
