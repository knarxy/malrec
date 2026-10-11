"""Population model settings for the expanded sample.

CFParams were chosen on ~4,000 public lists. After the expansion to ~15,000
the sampled users gained and the reference profile lost (stored model 11
-> 14: rho 0.788 -> 0.755 on the benchmark's reference splits) - the
pattern of a niche taste diluted by many mainstream lists, which more
capacity (factors, neighbours) or different shrinkage may undo. Each arm
changes one setting from the default, fits the benchmark's population model
on all its training users and runs the benchmark; fit time is printed too,
since a bigger model also costs every user's training.

  python -m experiments.exp_params [--users 100] [--arms "mf_rank=64;knn_k=120"]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import bench
from malrec.recsys.cf import CFParams

ARMS = "default;mf_rank=64;mf_rank=96;knn_k=120;sim_shrink=50;min_item_raters=30;bias_shrink=25"


def parse(arm: str) -> dict:
    if arm == "default":
        return {}
    out = {}
    for kv in arm.split(","):
        k, v = kv.split("=")
        out[k] = type(getattr(CFParams(), k))(float(v))
    return out


def main(max_eval: int, arms: list[str]):
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B = users[:nA], users[nA:nA + nB]
    rows = []
    real_params, real_pop = bench.CFParams, bench.population
    for arm in arms:
        over = parse(arm)
        bench.CFParams = lambda o=over: CFParams(**o)
        t = time.time()
        try:
            fitted = real_pop(entries, store, A, B, cache=False)
        finally:
            bench.CFParams = real_params
        fit_s = time.time() - t
        bench.population = lambda *a, f=fitted, **k: f      # run() reuses this fit
        try:
            out = bench.run(max_eval)
        finally:
            bench.population = real_pop
        line = {"arm": arm, "fit_s": fit_s}
        for g, gv in out["groups"].items():
            for m in ("rho", "r50", "hit10"):
                vals = [v[m] for v in gv["users"].values() if m in v]
                line[f"{g}/{m}"] = float(np.mean(vals)) if vals else float("nan")
        rows.append(line)
        print(f"  {arm} done, fit {fit_s:.0f}s [{time.time() - t0:.0f}s]", flush=True)

    groups = list(out["groups"])
    print(f"\n{'arm':>20}{'fit s':>7}" + "".join(f"{g:>34}" for g in groups))
    print(" " * 27 + "".join(f"{'rho':>12}{'r50':>11}{'hit10':>11}" for _ in groups))
    for r in rows:
        print(f"{r['arm']:>20}{r['fit_s']:7.0f}" + "".join(
            f"{r[f'{g}/rho']:12.4f}{r[f'{g}/r50']:11.4f}{r[f'{g}/hit10']:11.4f}" for g in groups))
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--arms", default=ARMS)
    a = ap.parse_args()
    main(a.users, a.arms.split(";"))
