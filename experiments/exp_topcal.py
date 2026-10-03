"""Do displayed scores overshoot at the top of the list?

The per-size calibration (service.fit_size_calibration) is fitted on a user's
future ratings in general. A list, though, shows only the titles with the
highest predictions, and those are selected partly on noise - the winner's
curse. Here, for held-out users given `n` ratings:

  * the whole eligible pool is scored and calibrated as production does;
  * "top of list" = test items whose shown score clears the user's K-th
    best pool prediction (K = 60 or 200);
  * bias = mean(shown - actual) on those items, and RMSE.

Candidate fix: fit the per-size line on top-of-list pairs only (from the
stacker users, never the evaluated ones). Being one line it stays monotonic,
so rankings do not change - only the number shown.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore
from malrec.recsys.service import calibration_from

SIZES = (5, 10, 25, 60, 150)


def pairs_for(pop, stacker, store, users, pool_all, n, rng, k):
    """(raw prediction, truth, is_top) for each test item of each user."""
    out = []
    for es in users:
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        if len(scored_all) < n + 8:
            continue
        train, test, cutoff = split_user(scored_all, rng)
        if len(train) < n:
            continue
        tr = [train[i] for i in rng.permutation(len(train))[:n]]
        ex = [Example(m, s, recency_weight(a, cutoff), a) for m, s, a in tr]
        um = fit_user(pop, stacker, store, ex, [], mode="sized")
        own = {e["mal_id"] for e in es}
        test_ids = [m for m, _, _ in test]
        pool = [m for m in pool_all if m not in own]
        pp = um.predict(pool)
        pp = pp[~np.isnan(pp)]
        if len(pp) < k:
            continue
        thr = np.sort(pp)[-k]
        p = um.predict(test_ids)
        for pi, (_, s, _) in zip(p, test):
            if not np.isnan(pi):
                out.append((float(pi), s, pi >= thr))
    return out


def main(max_users: int, seed: int):
    rng = np.random.default_rng(seed)
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
    store = ItemStore.load()
    rat = {u: [(e["mal_id"], float(e["score"]), e["at"]) for e in entries[u] if e["score"] > 0]
           for u in B}
    stacker = fit_stacker(pop, rat, store, seed=seed)
    pool_all = pool_ids(store)
    Bu = [entries[u] for u in B[:max_users]]
    Cu = [entries[u] for u in C[:max_users]]

    for k in (60, 200):
        print(f"\n== top of list = above the user's {k}th best pool prediction ==")
        print(f"{'n':>4} {'top pairs':>9} {'bias general':>13} {'bias top-fit':>13} "
              f"{'RMSE gen':>9} {'RMSE top':>9} {'all-items RMSE gen/top':>23}")
        for n in SIZES:
            fit_pairs = pairs_for(pop, stacker, store, Bu, pool_all, n, rng, k)
            ev = pairs_for(pop, stacker, store, Cu, pool_all, n, rng, k)
            if not fit_pairs or not ev:
                continue
            gen = calibration_from([p for p, _, _ in fit_pairs], [t for _, t, _ in fit_pairs])
            topf = [(p, t) for p, t, top in fit_pairs if top]
            tc = calibration_from([p for p, _ in topf], [t for _, t in topf])
            res = defaultdict(list)
            for p, t, top in ev:
                g = min(max(gen["slope"] * p + gen["intercept"], 1), 10)
                h = min(max(tc["slope"] * p + tc["intercept"], 1), 10)
                res["all_g"].append((g - t) ** 2); res["all_h"].append((h - t) ** 2)
                if top:
                    res["bg"].append(g - t); res["bh"].append(h - t)
            if not res["bg"]:
                continue
            print(f"{n:>4} {len(res['bg']):>9} {np.mean(res['bg']):>+13.2f} "
                  f"{np.mean(res['bh']):>+13.2f} {np.sqrt(np.mean(np.square(res['bg']))):>9.2f} "
                  f"{np.sqrt(np.mean(np.square(res['bh']))):>9.2f} "
                  f"{np.sqrt(np.mean(res['all_g'])):>11.2f} / {np.sqrt(np.mean(res['all_h'])):.2f}"
                  f"   (top line: {tc['slope']:.2f}x{tc['intercept']:+.2f}, {len(topf)} fit pairs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    main(a.max_users, a.seed)
