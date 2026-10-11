"""What should "Not for me" do beyond hiding the one title?

Simulated on the benchmark (experiments/bench.py: frozen snapshot, held-out
users at 25 / 150 / 400 list entries, Safe Bets through the production code).
Each user is shown their list; of its top 20, every title they did not go
on to like is dismissed with probability P (seeded) - someone skimming a
list and turning down what does not appeal. Then the list is rebuilt with
the dismissals taken into account, and the *rebuilt* list is scored on the
user's future ratings (dismissed titles excluded from every arm alike).

Arms (all hide the dismissed titles themselves):
  HIDE          today: only the titles go
  NEG o/w       plus each dismissal as an implicit rating `o` under the
                user's mean with weight `w` (service.dismissal_examples), in
                the personal model only
  NEG+F o/w     the same, also placed in the population fold-in
  FRAN          plus the rest of a dismissed title's franchise leaves the pool
  NEG+F+FRAN    both, at COMBINED

  python -m experiments.exp_notforme [--p 0.3]
"""
# build() runs inside the loop iteration that defines it, so its loop variables
# are the current ones
# ruff: noqa: B023
from __future__ import annotations

import argparse
import datetime as dt
import time
from collections import defaultdict

import numpy as np

from experiments import bench
from malrec.config import settings
from malrec.recsys.hybrid import split_user
from malrec.recsys.service import dismissal_examples

# (offset, weight, also in the population fold-in)
# round 1 (personal model only): -1/0.8 and -2/0.4 helped recall at 400
# entries, -2/0.8 and stronger cost rank correlation; round 2 is milder
NEG = ((-1.0, 0.4, False), (-1.0, 0.8, False), (-2.0, 0.4, False),
       (-1.0, 0.4, True), (-1.0, 0.8, True), (-2.0, 0.4, True), (-2.0, 0.8, True))
COMBINED = (-1.0, 0.4, True)


def main(p: float, max_eval: int):
    t0 = time.time()
    cfg = settings()
    entries, static = bench.load_snapshot()
    store, derived = static["store"], static["derived"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = bench.population(entries, store, A, B)
    pool_all = bench.eligible(store, derived)
    fran = {m: (store.rows[m].get("franchise_id") or m) for m in pool_all}
    print(f"population model {how}; P={p} [{time.time() - t0:.0f}s]", flush=True)
    def name(o, w, f):
        return f"NEG{'+F' if f else ''} {o:g}/{w:g}"
    arms = ["HIDE", *(name(*n) for n in NEG), "FRAN", "NEG+F+FRAN"]

    def genres(ids):
        return {g for m in ids for g in ((store.rows.get(m) or {}).get("mal_genres") or [])}

    for budget in bench.BUDGETS:
        M = defaultdict(list)
        n_dis = []
        urng = np.random.default_rng(bench.SEED + budget)
        crng = np.random.default_rng(bench.SEED + 7 * budget)
        for u in C:
            if len(M["HIDE"]) >= max_eval:
                break
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, urng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in urng.permutation(len(history))[:budget]]
            sc, im, li = bench.examples(seen, cutoff)
            if len(sc) < 5:
                continue
            listed = {e["mal_id"] for e in es} - test_ids
            liked = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            disliked = {m for m, s, _ in test_s if s < mean - 1}
            disliked |= {e["mal_id"] for e in es if e["status"] == "dropped" and e["at"] >= cutoff}
            fam = genres(e["mal_id"] for e in seen)
            items = [m for m, _, _ in test_s]
            truth = np.array([s for _, s, _ in test_s])
            rows = [bench.row_of(store, m) for m in pool_all if m not in listed]
            _, first = bench.safe_bets(pop, stacker, store, sc, im, li, rows, (1.0, 0.0), fam)
            dis = [m for m in first[:20] if m not in liked and crng.random() < p]
            if not dis:
                continue
            n_dis.append(len(dis))
            gone_f = {fran[m] for m in dis if m in fran} - {fran.get(m) for m in listed}
            hide = set(dis)
            rows_hide = [r for r in rows if r["mal_id"] not in hide]
            rows_fran = [r for r in rows_hide if fran.get(r["mal_id"]) not in gone_f]
            sm = float(np.average([x.score for x in sc], weights=[x.weight for x in sc]))
            at = {m: cutoff - dt.timedelta(days=1) for m in dis}

            def build(rows_, neg):
                if neg is None:
                    extra = []
                else:
                    (cfg.not_interested_offset, cfg.not_interested_weight,
                     cfg.not_interested_foldin) = neg
                    extra = dismissal_examples(at, sm, cutoff, listed=listed)
                scorer, top = bench.safe_bets(pop, stacker, store, sc, im + extra, li, rows_,
                                              (1.0, 0.0), fam)
                return bench.user_metrics(scorer, top, items, truth, liked - hide,
                                          disliked - hide)

            M["HIDE"].append(build(rows_hide, None))
            for n in NEG:
                M[name(*n)].append(build(rows_hide, n))
            M["FRAN"].append(build(rows_fran, None))
            M["NEG+F+FRAN"].append(build(rows_fran, COMBINED))
        print(f"\n== sampled {budget} ({len(M['HIDE'])} users, {np.mean(n_dis):.1f} dismissals each)"
              f" [{time.time() - t0:.0f}s]")
        print(f"   {'arm':<14}" + "".join(f"{k:>18}" for k in ("rho", "r50", "hit10", "bad10")))
        for arm in arms:
            cells = []
            for k, higher in (("rho", True), ("r50", True), ("hit10", True), ("bad10", False)):
                pairs = [(a[k], b[k]) for a, b in zip(M[arm], M["HIDE"]) if k in a and k in b]
                now = np.mean([a for a, _ in pairs]) if pairs else float("nan")
                if arm == "HIDE" or not pairs:
                    cells.append(f"{now:18.4f}")
                    continue
                d = np.array([a - b for a, b in pairs])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
                good = d.mean() > 0 if higher else d.mean() < 0
                mark = ("+" if good else "-") if abs(d.mean()) > 2 * se else " "
                cells.append(f"{now:10.4f}{d.mean():+7.4f}{mark}")
            print(f"   {arm:<14}{''.join(cells)}", flush=True)
    print("   (+/- : better/worse than HIDE by more than two standard errors, paired)")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--users", type=int, default=100)
    a = ap.parse_args()
    main(a.p, a.users)
