"""Is it the amount of data or where it comes from?

After the expansion to ~20k public lists the population model got clearly
better for sampled users and worse for the reference profile on every split.
Most new lists were discovered by `sync cf-expand` (source `mal_expand`, the
boards round robin), the rest by the original forum discovery. The
benchmark's population model fitted on three subsets of its training users,
everything else unchanged (same held-out users, same production code path):

  old     only users from the original sources
  random  as many users as `old`, drawn from all of them (size control)
  all     every training user

old ~ random < all: the amount matters, the source does not.
old > random: the newly discovered lists pull the model somewhere else.

With --stacker the population model is fitted on all users and the stacker
(which learns how much each part of it counts) on a subset of the stacker
users instead: `old` (original sources only), `random` (as many, from all)
or `all` - plus the original-sources-only model for comparison.

With --as-model-11 the population is the one stored model 11 was fitted on
(every list fetched before it, roles by the production hash) - its sampled
numbers leak (eval users are in it), only the reference profile counts.

With --spread, random 85 % subsets of model 11's users and of all users,
three each: how much the reference profile's rho moves when only the
membership changes - the noise any single-profile gate has to live with.

  python -m experiments.exp_source [--users 100] [--stacker | --as-model-11 | --spread]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import bench
from malrec.db import query


def main(max_eval: int, stacker: bool, as11: bool = False, spread: bool = False):
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B = users[:nA], users[nA:nA + nB]
    source = {r["id"]: r["source"] for r in query("SELECT id, source FROM cf_user")}
    old = [u for u in A if source.get(u) != "mal_expand"]
    rand = list(np.random.default_rng(1).choice(A, len(old), replace=False))
    b_old = [u for u in B if source.get(u) != "mal_expand"]
    b_rand = list(np.random.default_rng(2).choice(B, len(b_old), replace=False))
    if as11 or spread:
        from malrec.recsys.hybrid import unit_hash
        cut = query("SELECT created_at FROM global_model WHERE id=11")[0]["created_at"]
        before = {r["id"] for r in query("SELECT id FROM cf_user WHERE fetched_at <= %s", (cut,))}
        prod = [u for u in entries.users_with_scored(5) if u in before]
        arms = {"model11-users": ([u for u in prod if unit_hash("role", 0, u) >= 0.15],
                                  [u for u in prod if unit_hash("role", 0, u) < 0.15]),
                "old/old": (old, b_old)}
        if spread:
            arms = {}
            for k in range(3):
                r = np.random.default_rng(100 + k)
                p_, s_ = (
                    [u for u in prod if unit_hash("role", 0, u) >= 0.15],
                    [u for u in prod if unit_hash("role", 0, u) < 0.15])
                arms[f"m11 85% #{k}"] = (sorted(r.choice(p_, int(len(p_) * .85), replace=False)),
                                          sorted(r.choice(s_, int(len(s_) * .85), replace=False)))
            for k in range(3):
                r = np.random.default_rng(200 + k)
                arms[f"all 85% #{k}"] = (sorted(r.choice(A, int(len(A) * .85), replace=False)),
                                          sorted(r.choice(B, int(len(B) * .85), replace=False)))
    elif stacker:
        arms = {"old/old": (old, b_old), "all/old": (A, b_old), "all/random": (A, b_rand),
                "all/all": (A, B)}
    else:
        arms = {"old": (old, B), "random": (rand, B), "all": (A, B)}
    print(f"{len(A)} training users, {len(old)} from the original sources; stacker users "
          f"{len(B)}, {len(b_old)} original [{time.time() - t0:.0f}s]", flush=True)
    rows = []
    for name, (sub, bsub) in arms.items():
        real = bench.population
        fitted = real(entries, store, sub, bsub, cache=False)
        bench.population = lambda *a, f=fitted, **k: f      # run() reuses this fit
        try:
            out = bench.run(max_eval)
        finally:
            bench.population = real
        line = {"arm": name, "users": len(sub)}
        for g, gv in out["groups"].items():
            for m in ("rho", "r50", "hit10"):
                vals = [v[m] for v in gv["users"].values() if m in v]
                line[f"{g}/{m}"] = float(np.mean(vals)) if vals else float("nan")
        rows.append(line)
        print(f"  {name} ({len(sub)} users) done [{time.time() - t0:.0f}s]", flush=True)

    groups = list(out["groups"])
    print(f"\n{'pop/stacker' if stacker else 'arm':>13}{'users':>7}" + "".join(f"{g:>34}" for g in groups))
    print(" " * 18 + "".join(f"{'rho':>12}{'r50':>11}{'hit10':>11}" for _ in groups))
    for r in rows:
        print(f"{r['arm']:>13}{r['users']:7d}" + "".join(
            f"{r[f'{g}/rho']:12.4f}{r[f'{g}/r50']:11.4f}{r[f'{g}/hit10']:11.4f}" for g in groups))
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--stacker", action="store_true")
    ap.add_argument("--as-model-11", action="store_true")
    ap.add_argument("--spread", action="store_true")
    a = ap.parse_args()
    main(a.users, a.stacker, a.as_model_11, a.spread)
