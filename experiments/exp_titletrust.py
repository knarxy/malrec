"""The personal share per title instead of per user.

A user's prediction is lam * personal + (1 - lam) * stack with one lam for
every title. On long lists the lam best in hindsight beats every rule for
choosing it by +.008 rho (exp_lamcap, 2026-10-25), so a single number per user
may be what limits it. Here lam_i = clip(lam_u * m(bucket of x_i), 0, 1): the
user's lam as served (long lists: chosen on five folds by correlation, floor
0.3, ceiling 0.5; shorter: the size rule), times a multiplier for the title's
tercile of a feature x:

  pop   log MAL scorers: the population layer (kNN, MF) is weakest on titles
        few people rated, the content ridge does not care
  ev    the user's co-rating evidence for the title (FoldIn.evidence): how
        much their rated titles say about it through other people's ratings

The terciles' bounds and the three multipliers (grid MULTS) are fitted on
half of the users (by id), by mean rho on their future ratings; the other
half is scored with them, paired against lam_u as served. Benchmark snapshot
and population model, held-out users, the newest 20 % of each user's
ratings as the future.

  docker compose --profile lab run --rm -T --user root lab python -u -m experiments.exp_titletrust
"""
from __future__ import annotations

import itertools
import math
import sys
import time

import numpy as np

from experiments import bench
from malrec.eval import spearman
from malrec.recsys import hybrid, ltr
from malrec.recsys.hybrid import fit_user, size_weight, split_user

GROUPS = (("150-374", 150, 374), ("375+", 375, 10 ** 9))
MAX_USERS = 500                 # per group
MULTS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
FEATURES = ("pop", "ev")


def served_lam(pop, stacker, store, sc, im) -> float:
    if len(sc) >= 300:
        lam = hybrid._choose_lambda(pop, stacker, store, sc, im, "personal", None, 40.0, 40,
                                    False, "folds", "corr")
        return min(max(lam, 0.3), 0.5)
    return min(size_weight(len(sc), 150.0, 15.0), 0.5)


def rho_of(lam: np.ndarray, u: dict) -> float:
    return float(spearman(lam * u["a"] + (1 - lam) * u["b"], u["y"]))


def main() -> int:
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    bench.GROUPS = bench.groupings(store)
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    n_a, n_b = int(len(users) * 0.72), int(len(users) * 0.13)
    (pop, stacker), how = bench.population(entries, store, users[:n_a], users[n_a:n_a + n_b])
    held = users[n_a + n_b:]
    print(f"population model {how} [{time.time() - t0:.0f}s]", flush=True)
    urng = np.random.default_rng(bench.SEED + 2)
    data: dict[str, list[dict]] = {}
    for name, lo, hi in GROUPS:
        rows = []
        for u in (u for u in held if lo <= entries._n_scored.get(u, 0) <= hi * 1.25):
            if len(rows) >= MAX_USERS:
                break
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, urng)
            test_ids = {m for m, _, _ in test_s}
            hist = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            sc, im, li = ltr.history_examples(hist, cutoff)
            if not lo * 0.8 <= len(sc) <= hi:
                continue
            lam_u = served_lam(pop, stacker, store, sc, im)
            inner = fit_user(pop, stacker, store, sc, im, mode="personal", listed=li)
            stk = fit_user(pop, stacker, store, sc, im, mode="stack", listed=li)
            items = [m for m, _, _ in test_s]
            y = np.array([s for _, s, _ in test_s])
            a, b = inner.predict(items), stk.predict(items)
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() < 10 or y[ok].std() == 0:
                continue
            ids = [m for m, k in zip(items, ok) if k]
            rows.append({
                "u": u, "lam": lam_u, "a": a[ok], "b": b[ok], "y": y[ok],
                "pop": np.array([math.log10(max((store.rows.get(m) or {})
                                                .get("mal_num_scoring_users") or 1, 1))
                                 for m in ids]),
                "ev": np.asarray(stk.fold.evidence(ids), dtype=float)})
            if len(rows) % 100 == 0:
                print(f"  {name}: {len(rows)} users [{time.time() - t0:.0f}s]", flush=True)
        data[name] = rows
        print(f"{name}: {len(rows)} users [{time.time() - t0:.0f}s]", flush=True)

    for name, rows in data.items():
        train = [r for r in rows if r["u"] % 2 == 0]
        test = [r for r in rows if r["u"] % 2 == 1]
        base = np.array([rho_of(np.full(len(r["y"]), r["lam"]), r) for r in test])
        best_fix = max((float(np.mean([rho_of(np.full(len(r["y"]), lam), r) for r in train])),
                        lam) for lam in np.round(np.linspace(0, 1, 11), 2))
        print(f"\n== {name}: fitted on {len(train)} users, scored on {len(test)};"
              f" served lam rho {base.mean():.4f} (mean lam {np.mean([r['lam'] for r in test]):.2f})")
        for f in FEATURES:
            allx = np.concatenate([r[f] for r in train])
            cuts = np.quantile(allx, [1 / 3, 2 / 3])
            best = (-1.0, (1.0, 1.0, 1.0))
            for m in itertools.product(MULTS, repeat=3):
                mm = np.array(m)
                v = float(np.mean([rho_of(np.clip(r["lam"] * mm[np.searchsorted(cuts, r[f])], 0, 1), r)
                                   for r in train]))
                if v > best[0]:
                    best = (v, m)
            mm = np.array(best[1])
            now = np.array([rho_of(np.clip(r["lam"] * mm[np.searchsorted(cuts, r[f])], 0, 1), r)
                            for r in test])
            d = now - base
            se = 2 * d.std(ddof=1) / np.sqrt(len(d))
            print(f"   {f:<4} terciles at {np.round(cuts, 2).tolist()}, multipliers {best[1]}:"
                  f" rho {now.mean():.4f} ({d.mean():+.4f} +-{se:.4f})")
        oracle = np.array([max(rho_of(np.full(len(r["y"]), lam), r)
                               for lam in np.linspace(0, 1, 11)) for r in test])
        d = oracle - base
        print(f"   per-user lam best in hindsight: {oracle.mean():.4f} ({d.mean():+.4f});"
              f" best fixed lam on the fitting half: {best_fix[1]:g}")
    print(f"\n[{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
