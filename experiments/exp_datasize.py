"""How much does the population model gain from more public lists?

The learning curve before a long fetch: the benchmark's population model
(experiments/bench.py) fitted on a share of its training users, everything
else unchanged - same held-out users, same builds through the production
code, the live tables frozen as the benchmark's (bench --freeze). Run with
the learned orders off (their rankers belong to the full model): each
share's metrics are paired with the full model's on the same users. If they
still rise toward 100 %, more lists pay; where they flatten says how far.

  python -m experiments.exp_datasize [--fracs 0.5,0.75,1] [--max-eval 300]
"""
from __future__ import annotations

import argparse
import pickle
import time

import numpy as np

from experiments import bench

METRICS = ("rho", "r50", "hit10", "disc_r50", "disc_hit10", "gems_r50", "gems_hit10", "r50old",
           "disc_r50old", "gems_r50old")


def main(fracs: list[float], max_eval: int):
    t0 = time.time()
    if bench.FROZEN.exists():
        bench.LIVE = pickle.loads(bench.FROZEN.read_bytes())
    entries, static = bench.load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    n_a, n_b = int(len(users) * 0.72), int(len(users) * 0.13)
    a_users, b_users = users[:n_a], users[n_a:n_a + n_b]
    print(f"{len(a_users)} population users at 100 % [{time.time() - t0:.0f}s]", flush=True)
    runs = {}
    for f in fracs:
        sub = a_users[:int(len(a_users) * f)]
        real = bench.population
        fitted = real(entries, store, sub, b_users, cache=False)
        bench.population = lambda *a, f=fitted, **k: f      # run() reuses this fit
        try:
            runs[f] = bench.run(max_eval)
        finally:
            bench.population = real
        print(f"  {f:.0%} ({len(sub)} users) done [{time.time() - t0:.0f}s]", flush=True)

    full = runs[max(fracs)]
    for g in full["groups"]:
        print(f"\n== {g}: mean, and change vs {max(fracs):.0%} (paired, +-2 SE)")
        print(f"   {'share':>6}" + "".join(f"{m:>26}" for m in METRICS))
        for f, out in sorted(runs.items()):
            line = f"   {f:6.0%}"
            for m in METRICS:
                un, uf = out["groups"][g]["users"], full["groups"][g]["users"]
                ks = [k for k in un if m in un[k] and m in uf.get(k, {})]
                if not ks:
                    line += f"{'':>26}"
                    continue
                d = np.array([un[k][m] - uf[k][m] for k in ks])
                se = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                line += f"{np.mean([un[k][m] for k in ks]):9.4f}{d.mean():+9.4f}({se:.4f})"
            print(line)
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.5,0.75,1")
    ap.add_argument("--max-eval", type=int, default=300)
    a = ap.parse_args()
    main([float(x) for x in a.fracs.split(",")], a.max_eval)
