"""Sequels: does the rating of the previous season predict the next one?

Pairs (prequel p, direct sequel s, both main format) rated by held-out sampled
users. For each pair the user's history is everything they rated *before*
rating s (so no later season leaks in), and it must contain p.

  model     the deployed hybrid (size-weighted) fitted on that history
  anchor    r_p + delta(p, s): the user's prequel score plus how raters of
            both typically move from p to s (mean of r_s - r_p over
            population-fit users, shrunk toward the global mean delta with
            k = 20 pseudo-pairs)
  blend w   w * anchor + (1 - w) * model

Reported: RMSE, bias and rank correlation over the pairs, and the same for
sequels whose population delta rests on fewer than 20 pairs (thin data).
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries
from malrec.db import query
from malrec.eval import spearman
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight
from malrec.recsys.items import ItemStore

SHRINK = 20.0
WEIGHTS = (0.0, 0.25, 0.5, 0.65, 0.8, 1.0)


def main(max_pairs: int, seed: int):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pairs = [(r["src"], r["dst"]) for r in query("""
        SELECT r.src, r.dst FROM relation r
          JOIN anime a ON a.mal_id = r.src JOIN anime b ON b.mal_id = r.dst
         WHERE r.relation_type = 'sequel'
           AND format_class(a.media_type) = 'main' AND format_class(b.media_type) = 'main'""")]
    seq_of = defaultdict(set)
    for p, s in pairs:
        seq_of[s].add(p)
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
    # population deltas from the fit users only
    deltas = defaultdict(list)
    for u in A:
        sc = {e["mal_id"]: float(e["score"]) for e in entries[u] if e["score"] > 0}
        for p, s in pairs:
            if p in sc and s in sc:
                deltas[(p, s)].append(sc[s] - sc[p])
    all_d = [d for v in deltas.values() for d in v]
    g = float(np.mean(all_d))
    print(f"{len(pairs)} sequel pairs, {len(all_d)} population pair-ratings, global delta {g:+.2f}"
          f" [{time.time() - t0:.0f}s]", flush=True)

    def delta(p, s):
        v = deltas.get((p, s), [])
        return (sum(v) + SHRINK * g) / (len(v) + SHRINK), len(v)

    rows = []
    for u in C:
        es = [e for e in entries[u] if e["score"] > 0]
        sc = {e["mal_id"]: e for e in es}
        for s in [m for m in sc if m in seq_of]:
            ps = [p for p in seq_of[s] if p in sc and sc[p]["at"] < sc[s]["at"]]
            if not ps:
                continue
            p = ps[0]
            cutoff = sc[s]["at"]
            hist = [e for e in es if e["at"] < cutoff and e["mal_id"] != s]
            if len(hist) < 10:
                continue
            ex = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff), e["at"])
                  for e in hist]
            listed = {e["mal_id"]: recency_weight(e["at"], cutoff) for e in hist}
            um = fit_user(pop, stacker, store, ex, [], mode="sized", listed=listed)
            pm = float(um.predict([s])[0])
            if np.isnan(pm):
                continue
            d, n = delta(p, s)
            rows.append((float(sc[s]["score"]), pm, float(sc[p]["score"]) + d, n, len(hist)))
            if len(rows) >= max_pairs:
                break
        if len(rows) >= max_pairs:
            break
    R = np.array(rows)
    t, pm, pa, n = R[:, 0], R[:, 1], np.clip(R[:, 2], 1, 10), R[:, 3]
    print(f"\n{len(R)} held-out sequel ratings [{time.time() - t0:.0f}s]")
    print(f"{'':<14}{'RMSE':>7}{'bias':>7}{'rho':>7}   thin-data RMSE ({int((n < 20).sum())} pairs)")
    for w in WEIGHTS:
        p = w * pa + (1 - w) * pm
        thin = n < 20
        print(f"{'model' if w == 0 else 'anchor' if w == 1 else f'blend {w}':<14}"
              f"{np.sqrt(np.mean((p - t) ** 2)):7.3f}{np.mean(p - t):+7.2f}{spearman(p, t):7.3f}"
              f"   {np.sqrt(np.mean((p[thin] - t[thin]) ** 2)) if thin.any() else float('nan'):7.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_pairs, a.seed)
