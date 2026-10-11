"""Stored production population models on the benchmark's reference profile.

The gate and the benchmark disagreed after the expansion: the gate saw the
reference profile lose 0.022 rho on every split (model 11 -> 14), the
benchmark's own fits showed no loss from more data. This puts the stored
models themselves through the benchmark's reference evaluation, so the
question "is it the models or the protocol" has one answer. Only the
reference profile counts here: it is in no population (app users are
excluded by name), while the benchmark's sampled users may be in a stored
model's training data, so their numbers are printed but leak.

  python -m experiments.exp_models 11 14 [--users 10]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import bench
from malrec.recsys.service import load_global


def main(ids: list[int], max_eval: int):
    t0 = time.time()
    rows = []
    for gid in ids:
        gm = load_global(gid)
        real = bench.population
        bench.population = lambda *a, g=gm, **k: ((g.pop, g.stacker), f"stored {gid}")
        try:
            out = bench.run(max_eval)
        finally:
            bench.population = real
        ref = out["groups"]["reference"]["users"]
        rows.append((gid, gm.meta.get("users"), ref))
        print(f"  model {gid} done [{time.time() - t0:.0f}s]", flush=True)

    splits = list(rows[0][2])
    print(f"\n{'model':>6}{'users':>7}{'rho':>9}{'r50':>8}{'hit10':>8}   rho per split")
    for gid, n, ref in rows:
        rho = [ref[s]["rho"] for s in splits]
        print(f"{gid:6d}{n:7d}{np.mean(rho):9.4f}{np.mean([ref[s]['r50'] for s in splits]):8.4f}"
              f"{np.mean([ref[s]['hit10'] for s in splits]):8.4f}   "
              + " ".join(f"{x:.3f}" for x in rho))
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", type=int, nargs="+")
    ap.add_argument("--users", type=int, default=10)
    a = ap.parse_args()
    main(a.ids, a.users)
