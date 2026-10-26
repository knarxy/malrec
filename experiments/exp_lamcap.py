"""The ceiling on a long list's chosen personal share (blend_lam_max 0.5).

From 300 ratings the share is chosen on the user's own ratings
(hybrid._choose_lambda, five folds by correlation) and then held to
[blend_lam_min, blend_lam_max] = [0.3, 0.5]. On the long lists of
exp_blendcrit (2026-10-25) 41-52 % of the users from 750 ratings on sat at
the ceiling, and a fixed 0.5 was the best fixed share at 1500+ - the search
stopped at 0.5. This lifts it: the same held-out users (benchmark snapshot,
its population model, the newest 20 % of each user's ratings as the
future), the shipped choice held to [0.3, cap] for each cap, fixed shares up
to 1, and the share best in hindsight. rho / rmse as in exp_blendcrit (the
rmse after the calibration line fitted on the newest fifth before the
cutoff).

  docker compose --profile lab run --rm -T --user root lab python -u -m experiments.exp_lamcap
"""
from __future__ import annotations

import sys
import time

import numpy as np

from experiments import bench
from experiments.exp_blendcrit import BUCKETS, MAX_USERS, MIN_SCORED, score
from malrec.recsys import hybrid, ltr
from malrec.recsys.hybrid import fit_user, split_user

CAPS = (0.5, 0.6, 0.7, 0.8, 1.0)
GRID = np.round(np.linspace(0, 1, 11), 2)


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
    long_ = [u for u in users[n_a + n_b:] if entries._n_scored.get(u, 0) >= MIN_SCORED]
    print(f"{len(long_)} held-out users with >= {MIN_SCORED} ratings, population model {how}"
          f" [{time.time() - t0:.0f}s]", flush=True)
    urng = np.random.default_rng(bench.SEED + 1)
    rows = []
    for u in long_[:MAX_USERS]:
        es = entries[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        _, test_s, cutoff = split_user(scored_all, urng)
        test_ids = {m for m, _, _ in test_s}
        hist = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
        sc, im, li = ltr.history_examples(hist, cutoff)
        if len(sc) < 300:
            continue
        chosen = hybrid._choose_lambda(pop, stacker, store, sc, im, "personal", None, 40.0, 40,
                                       False, "folds", "corr")
        # the calibration line: both sides fitted on the older 80 %, on the newest fifth
        (tr, imp, va), = hybrid._blend_splits(sc, im, "newest")
        va_items = [e.mal_id for e in va]
        va_a = fit_user(pop, stacker, store, tr, imp, mode="personal").predict(va_items)
        va_b = fit_user(pop, stacker, store, tr, imp, mode="stack").predict(va_items)
        va_truth = np.array([e.score for e in va])
        inner = fit_user(pop, stacker, store, sc, im, mode="personal", listed=li)
        stk = fit_user(pop, stacker, store, sc, im, mode="stack", listed=li)
        items = [m for m, _, _ in test_s]
        truth = np.array([s for _, s, _ in test_s])
        a, b = inner.predict(items), stk.predict(items)
        ok = ~(np.isnan(a) | np.isnan(b))
        vok = ~(np.isnan(va_a) | np.isnan(va_b))
        if ok.sum() < 10 or truth[ok].std() == 0:
            continue
        future, newest = (a[ok], b[ok], truth[ok]), (va_a[vok], va_b[vok], va_truth[vok])
        r: dict = {"n": len(sc), "chosen": chosen}
        for g in GRID:
            r[f"fix{g}"] = score(float(g), future, newest)
        for cap in CAPS:
            lam = min(max(chosen, 0.3), cap)
            r[f"cap{cap}"] = score(lam, future, newest)
            r[f"lam cap{cap}"] = lam
        r["oracle"] = max((r[f"fix{g}"] for g in GRID), key=lambda x: x[0])
        rows.append(r)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} users [{time.time() - t0:.0f}s]", flush=True)

    arms = [f"cap{c}" for c in CAPS] + [f"fix{g}" for g in GRID] + ["oracle"]
    base = "cap0.5"
    names = [f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" for lo, hi in BUCKETS] + ["all"]
    for k, name in enumerate(names):
        sel = [r for r in rows if k == len(BUCKETS) or BUCKETS[k][0] <= r["n"] <= BUCKETS[k][1]]
        if not sel:
            continue
        print(f"\n== ratings {name} ({len(sel)} users); change vs {base} (today), +-2 SE paired")
        print(f"   {'arm':<10}{'rho':>8}{'change':>9}{'±2se':>7}{'rmse':>8}{'change':>9}{'±2se':>7}"
              f"{'mean lam':>10}")
        for arm in arms:
            out = f"   {arm:<10}"
            for j in (0, 1):
                v = np.array([r[arm][j] for r in sel])
                d = v - np.array([r[base][j] for r in sel])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                out += f"{v.mean():8.4f}{d.mean():+9.4f}{2 * se:7.4f}"
            lam = [r[f"lam {arm}"] for r in sel if f"lam {arm}" in r]
            out += f"{np.mean(lam):10.2f}" if lam else ""
            print(out)
        ch = np.array([r["chosen"] for r in sel])
        print("   chosen (before the bounds): " + " ".join(
            f"{g:g}:{np.mean(np.isclose(ch, g)):.2f}" for g in GRID))
    print(f"\n[{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
