"""Population x personal weight x learned order, on the benchmark.

After the expansion the population model got better for sampled users and
worse for the reference profile, whose taste the many new lists pull toward
the mainstream (exp_source.py, exp_refitems.py). Two counterweights:
  - the personal side: the blend's handover (blend_size_n0), its cap
    (blend_lam_max) and the list size from which the share is chosen on the
    user's own newest ratings (chosen_blend_min), all set on ~4,000 lists
  - the learned Safe Bets order (order_ranker_weight, recsys/ltr.py)
Each arm is "population:settings:weight":
  population  all   the benchmark's (every training user)
              m11   the users stored model 11 was fitted on (lists fetched
                    before it; its sampled numbers leak - eval users are in
                    it - only the reference profile counts)
  settings    key=value,... overrides of config (empty: as configured)
  weight      order_ranker_weight (0 = the hand-tuned key)
  features    optional: "base" (the first ranker's 19), "taste" (with the
              user's long-series taste - the rankers shipped until 2026-10-13,
              with the release year) or "age" (release age and the user's
              age taste instead of the year, recsys/ltr.py; the default)
Every metric is compared per user with the first arm (paired, +/- beyond
two standard errors).

  python -m experiments.exp_blend "all::0" "all:blend_lam_max=0.7:0" ...
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import bench
from malrec.config import settings
from malrec.db import query
from malrec.recsys import ltr
from malrec.recsys.hybrid import unit_hash

METRICS = (("rho", True), ("r50", True), ("ips50", True), ("hit10", True), ("bad10", False),
           ("agegap", False), ("new20", False),
           ("disc_r50", True), ("disc_ips50", True), ("disc_hit10", True),
           ("disc_agegap", False), ("disc_new20", False),
           ("gems_r50", True), ("gems_ips50", True), ("gems_hit10", True),
           ("gems_agegap", False), ("gems_new20", False))


def parse(arm: str) -> tuple[str, dict, float, str]:
    pop, over, w, *fs = arm.split(":")
    cfg = settings()
    out = {}
    for kv in filter(None, over.split(",")):
        k, v = kv.split("=")
        cur = getattr(cfg, k)
        out[k] = type(cur)(float(v)) if isinstance(cur, (int, float)) else type(cur)(v)
    return pop, out, float(w), (fs[0] if fs else "age")


def main(arms: list[str], max_eval: int):
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B = users[:nA], users[nA:nA + nB]
    subsets = {"all": (A, B)}
    if any(a.startswith("m11:") for a in arms):
        cut = query("SELECT created_at FROM global_model WHERE id=11")[0]["created_at"]
        before = {r["id"] for r in query("SELECT id FROM cf_user WHERE fetched_at <= %s", (cut,))}
        prod = [u for u in entries.users_with_scored(5) if u in before]
        subsets["m11"] = ([u for u in prod if unit_hash("role", 0, u) >= 0.15],
                          [u for u in prod if unit_hash("role", 0, u) < 0.15])
    fits: dict[str, tuple] = {}
    cfg = settings()
    real_pop, real_ranker = bench.population, bench.order_ranker
    results = []
    for arm in arms:
        pname, over, w, fset = parse(arm)
        if pname not in fits:       # settings here do not touch the population fit
            pa, pb = subsets[pname]
            fits[pname] = real_pop(entries, store, pa, pb, cache=(pname == "all"))
        fitted = fits[pname]
        saved = {k: getattr(cfg, k) for k in [*over, "order_ranker_weight"]}
        for k, v in {**over, "order_ranker_weight": w}.items():
            setattr(cfg, k, v)
        bench.population = lambda *a, f=fitted, **k: f
        real_features = ltr.FEATURES
        ltr.FEATURES = {"base": ltr.BASE_FEATURES, "taste": ltr.BASE_FEATURES + ltr.TASTE_FEATURES
                        }.get(fset, real_features)
        if pname != "all":
            # the ranker learns from this population's own stacker users
            sb = subsets[pname][1]
            bench.order_ranker = lambda e, s, p, st, _B, pool, sb=sb: (
                ltr.train(p, st, s, pool, {u: e[u] for u in sb}), "trained")
        try:
            out = bench.run(max_eval)
        finally:
            bench.population, bench.order_ranker = real_pop, real_ranker
            ltr.FEATURES = real_features
            for k, v in saved.items():
                setattr(cfg, k, v)
        results.append((arm, out))
        print(f"  {arm} done [{time.time() - t0:.0f}s]", flush=True)
        table(results)                  # after every arm, so a late failure loses nothing
    print(f"[{time.time() - t0:.0f}s]")


def table(results):
    base = results[0][1]
    for g in base["groups"]:
        print(f"\n== {g}")
        print(f"   {'arm':<44}" + "".join(f"{m:>17}" for m, _ in METRICS) + f"{'pop#':>7}")
        for arm, out in results:
            gu, bu = out["groups"][g]["users"], base["groups"][g]["users"]
            cells = []
            for m, higher in METRICS:
                pairs = [(gu[u][m], bu[u][m]) for u in gu if u in bu and m in gu[u] and m in bu[u]]
                now = float(np.mean([a for a, _ in pairs])) if pairs else float("nan")
                if out is base or len(pairs) < 3:
                    cells.append(f"{now:17.4f}")
                    continue
                d = np.array([a - b for a, b in pairs])
                se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
                flag = ""
                if abs(d.mean()) > 2 * se and se > 0:
                    flag = "+" if (d.mean() > 0) == higher else "-"
                cells.append(f"{now:9.4f}{d.mean():+7.4f}{flag:1}")
            pop = out["groups"][g].get("pop")
            print(f"   {arm:<44}" + "".join(cells) + f"{pop if pop is not None else float('nan'):7.0f}")
    print("\n(+/- : better/worse than the first arm by more than two standard errors, paired)",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--users", type=int, default=100)
    a = ap.parse_args()
    main(a.arms, a.users)
