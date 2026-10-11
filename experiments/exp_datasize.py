"""How much does the population model gain from more public lists?

The learning curve before a long fetch: the benchmark's population model
(experiments/bench.py) fitted on 25 / 50 / 75 / 100 % of its training users,
everything else unchanged - same held-out users, same Safe Bets build through
the production code, same metrics. If the curve is still rising at 100 %,
more lists pay; where it flattens says how far.

  python -m experiments.exp_datasize [--fracs 0.25,0.5,0.75,1]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import bench


def main(fracs: list[float]):
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B = users[:nA], users[nA:nA + nB]
    print(f"{len(A)} population users at 100 % [{time.time() - t0:.0f}s]", flush=True)
    rows = []
    for f in fracs:
        sub = A[:int(len(A) * f)]
        real = bench.population
        fitted = real(entries, store, sub, B, cache=False)
        bench.population = lambda *a, f=fitted, **k: f      # run() reuses this fit
        try:
            out = bench.run(100)
        finally:
            bench.population = real
        line = {"frac": f, "users": len(sub)}
        for g, gv in out["groups"].items():
            for m in ("rho", "r50", "hit10"):
                vals = [v[m] for v in gv["users"].values() if m in v]
                line[f"{g}/{m}"] = float(np.mean(vals)) if vals else float("nan")
            line[f"{g}/overlap"] = gv.get("overlap")
        rows.append(line)
        print(f"  {f:.0%} ({len(sub)} users) done [{time.time() - t0:.0f}s]", flush=True)

    groups = [g for g in out["groups"]]
    print(f"\n{'share':>6}{'users':>7}" + "".join(f"{g:>34}" for g in groups))
    print(" " * 13 + "".join(f"{'rho':>12}{'r50':>11}{'hit10':>11}" for _ in groups))
    for r in rows:
        print(f"{r['frac']:6.0%}{r['users']:7d}" + "".join(
            f"{r[f'{g}/rho']:12.4f}{r[f'{g}/r50']:11.4f}{r[f'{g}/hit10']:11.4f}" for g in groups))
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.25,0.5,0.75,1")
    a = ap.parse_args()
    main([float(x) for x in a.fracs.split(",")])
