"""Why did a title the user went on to like miss their Safe Bets top 50?

Error analysis on the benchmark (experiments/bench.py: same snapshot, users
and production build). For every liked future title (>= max(own mean, 7))
outside the top 50, one reason:

  filter: <why>      not eligible for Safe Bets at all - too few MAL ratings
                     (< 2000), not a main format, a side story or recap,
                     NSFW, not aired, unknown to the catalogue
  rerank             high enough on the order key, pushed out by the genre
                     balance / franchise collapse of rank.rerank
  prediction         the predicted score is what falls short most, compared
                     with the title at place 50 of the order key
  relevance          the relevance bonus falls short most (not "in the user's
                     world" by co-occurrence / EASE)
  popularity         the popularity prior (thin lists) costs it most
  risk               a risk penalty costs it most (disagreement, unfamiliar
                     genre, acclaim, long series)

The terms come from rank.order_terms - the function production orders by -
so the reasons describe the shipped key, not a re-implementation. Also
reported: how far down the misses sit, and the reference profile's misses by
title.

  python -m experiments.bench_misses [--users 100]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np

from experiments import bench
from malrec import rank
from malrec.recsys.hybrid import split_user
from malrec.surfaces import SPECS

CAUSES = ("prediction", "relevance", "popularity", "risk", "memory")


def filter_reason(m: int, store, derived) -> str | None:
    r = store.rows.get(m)
    if r is None:
        return "filter: not in catalogue"
    if (r.get("mal_num_scoring_users") or 0) < SPECS["safe_bets"].min_scorers:
        return "filter: < 2000 MAL ratings"
    if r.get("format_class") != "main":
        return "filter: not a main format"
    if m in derived:
        return "filter: side story / recap"
    if r.get("nsfw") not in (None, "white"):
        return "filter: NSFW"
    if r.get("status") == "not_yet_aired":
        return "filter: not aired"
    return None


def diagnose(scorer, rows, li, familiar, store, derived, liked: set[int]):
    """(reasons for each missed liked title, key position of each) for one user."""
    _, top = bench.ranked(scorer, rows, li, familiar, store)
    top50 = {c.mal_id for c in top[:50]}
    real = rank.familiar_genres
    rank.familiar_genres = lambda _uid: familiar
    try:
        t = rank.order_terms(scorer, rows, pop_prior=bench.CFG.safe_bets_popularity)
    finally:
        rank.familiar_genres = real
    ids = [r["mal_id"] for r in rows]
    order = [k for k in np.argsort(-t.key) if t.ok[k]]
    pos = {ids[k]: p for p, k in enumerate(order)}
    ref = order[49] if len(order) > 49 else order[-1]
    out = []
    for m in liked - top50:
        why = filter_reason(m, store, derived)
        if why is None and m not in pos:
            why = "filter: model cannot score it"
        if why is not None:
            out.append((m, why, None))
            continue
        p = pos[m]
        if p < 50:
            out.append((m, "rerank", p))
            continue
        k = ids.index(m)
        gaps = {"prediction": t.shown[k] - t.shown[ref], "relevance": t.rel[k] - t.rel[ref],
                "popularity": t.pop[k] - t.pop[ref], "risk": -(t.risk[k] - t.risk[ref]),
                "memory": t.mix[k] - t.mix[ref]}
        out.append((m, min(gaps, key=lambda c: gaps[c]), p))
    return out


def main(max_eval: int):
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store, derived = static["store"], static["derived"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = bench.population(entries, store, A, B)
    pool_all = bench.eligible(store, derived)
    print(f"population model {how} [{time.time() - t0:.0f}s]", flush=True)

    def genres(ids):
        return {g for m in ids for g in ((store.rows.get(m) or {}).get("mal_genres") or [])}

    def report(name, misses, n_liked, n_users, titles=False):
        c = Counter(why for _, why, _ in misses)
        depth = [p for _, why, p in misses if p is not None and why != "rerank"]
        unit = "splits" if name.startswith("reference") else "users"
        print(f"\n== {name}: {n_users} {unit}, {n_liked} liked future titles, "
              f"{len(misses)} outside the top 50 ({len(misses) / max(n_liked, 1):.0%})")
        for why, k in c.most_common():
            print(f"   {why:<32}{k:6d}  {k / len(misses):6.1%}")
        if depth:
            q = np.percentile(depth, [25, 50, 75])
            print(f"   order-key place of the scored misses: quartiles {q[0]:.0f} / {q[1]:.0f}"
                  f" / {q[2]:.0f}; within 51-100: {np.mean([d < 100 for d in depth]):.0%}")
        if titles:
            for m, why, p in sorted(misses, key=lambda x: (x[1], x[2] or 0)):
                print(f"     {why:<30} #{p if p is not None else '-':<6} "
                      f"{(store.rows.get(m) or {}).get('title', m)}")

    for budget in bench.BUDGETS:
        urng = np.random.default_rng(bench.SEED + budget)
        misses, n_liked, n = [], 0, 0
        for u in C:
            if n >= max_eval:
                break
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, urng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in urng.permutation(len(history))[:budget]]
            sc, im, li = bench.examples(seen, cutoff)
            if len(sc) < 5:
                continue
            listed = {e["mal_id"] for e in es} - test_ids
            liked = {m for m, s, _ in test_s if s >= max(mean, 7.0)} - listed
            scorer = bench.make_scorer(pop, stacker, store, sc, im, li, (1.0, 0.0))
            rows = [bench.row_of(store, m) for m in pool_all if m not in listed]
            misses += diagnose(scorer, rows, li, genres(e["mal_id"] for e in seen), store,
                               derived, liked)
            n_liked += len(liked)
            n += 1
        report(f"sampled {budget}", misses, n_liked, n)
        print(f"   [{time.time() - t0:.0f}s]", flush=True)

    ref = static["ref"]
    le = ref["le"]
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    misses, liked_all = [], set()
    for h in bench.REF_SPLITS:
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        sc, im, li = bench.examples(hist, cutoff)
        listed = {r["mal_id"] for r in le} - test_ids
        liked = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        scorer = bench.make_scorer(pop, stacker, store, sc, im, li, ref["cal"])
        rows = [bench.row_of(store, m) for m in pool_all if m not in listed]
        misses += diagnose(scorer, rows, li, genres(r["mal_id"] for r in hist), store,
                           derived, liked)
        liked_all |= liked
    # the same title misses in several splits; count it once, at its best place
    best: dict[int, tuple] = {}
    for m, why, p in misses:
        if m not in best or (p or 1e9) < (best[m][2] or 1e9):
            best[m] = (m, why, p)
    report("reference profile (8 splits, each title once)", list(best.values()),
           len(liked_all), 8, titles=True)
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100)
    a = ap.parse_args()
    main(a.users)
