"""Like-minded lists: does a user-neighbour signal add to the population model?

The population layer is item-based (co-rating kNN, ALS, co-occurrence, EASE)
and pulls every list toward what most lists like - after the fourfold
expansion the reference profile lost most on mainstream titles it rated low
(exp_refitems.py). A user-based signal reads the other way: find the lists
that rate like this one (bias-free deviations on shared titles, Pearson with
significance shrinkage) and take what *they* thought of a title. It only
becomes possible with many lists: at 20,000 a taste far from the mainstream
still has a few hundred close neighbours.

Per user, from the same training entries the user model sees:
  d_j       their rating - their mean - the title's population bias
  sim(v)    sum w d_j r_vj / sqrt(sum w d_j^2 . sum w r_vj^2) over shared
            titles, times n/(n + SIM_SHRINK)   (r_v = neighbour's deviations)
  dev(i)    sum sim(v) r_vi / (sum |sim(v)| over v who rated i + DEV_SHRINK),
            over the K most similar lists
The prediction becomes  model + weight * dev  - rank correlation, recall and
the Safe Bets top 10 through the benchmark's production path (bench.run),
paired with weight 0. The neighbours are the population model's fit users
(the benchmark's 72 %); evaluated users are never among them.

  python -m experiments.exp_uknn                  weights 0 0.25 0.5 1
  python -m experiments.exp_uknn --weights 0 0.5 --k 300
  python -m experiments.exp_uknn --residual       is anything left to add?

--residual (weight 0 only) records per user the rank correlation of dev
with the future ratings (dev_rho) and with what the model got wrong on them
(res_rho: future rating - shown score). res_rho near zero means the
neighbours know nothing the population model and the personal side do not.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.sparse as sp

from experiments import bench
from malrec.recsys.scorer import HybridScorer

SIM_SHRINK = 10.0
DEV_SHRINK = 5.0
METRICS = (("rho", True), ("r50", True), ("hit10", True), ("bad10", False),
           ("disc_r50", True), ("gems_r50", True))
RESIDUAL = (("rho", True), ("dev_rho", True), ("res_rho", True), ("res_rho_supp", True))


class Neighbours:
    """The fit users' bias-free deviations, users x titles."""

    def __init__(self, entries, pop, A):
        u, m, sc = entries.columns(A)
        keep = (sc > 0) & np.isin(m, np.fromiter(pop.index, dtype=np.int64))
        u, m, sc = u[keep], m[keep], sc[keep].astype(np.float64)
        uid, row = np.unique(u, return_inverse=True)
        col = np.array([pop.index[x] for x in m.tolist()])
        mean = np.bincount(row, sc) / np.bincount(row)
        dev = sc - mean[row] - pop.bias[col]
        shape = (len(uid), len(pop.index))
        self.R = sp.csr_matrix((dev, (row, col)), shape=shape)
        self.C = self.R.tocsc()
        self.pop = pop

    def of(self, ratings: dict[int, float], weights: dict[int, float], k: int):
        """(neighbour rows, their similarities) for one user."""
        ids = [x for x in ratings if x in self.pop.index]
        if len(ids) < 5:
            return None
        J = np.array([self.pop.index[x] for x in ids])
        w = np.array([weights.get(x, 1.0) for x in ids])
        r = np.array([ratings[x] for x in ids])
        mu = float((w * r).sum() / w.sum())
        d = r - mu - self.pop.bias[J]
        S = self.C[:, J]                                   # fit users x the user's titles
        B = (S != 0).astype(np.float64)
        num = S @ (w * d)
        a = B @ (w * d * d)
        b = S.multiply(S) @ w
        n = np.asarray(B.sum(axis=1)).ravel()
        sim = num / np.sqrt(np.maximum(a * b, 1e-12)) * n / (n + SIM_SHRINK)
        top = np.argpartition(-sim, min(k, len(sim) - 1))[:k]
        top = top[sim[top] > 0]
        return top, sim[top]

    def dev(self, nb, mal_ids: list[int]) -> np.ndarray:
        out = np.zeros(len(mal_ids))
        if nb is None or not len(nb[0]):
            return out
        top, sim = nb
        idx = np.array([self.pop.index.get(x, -1) for x in mal_ids])
        known = idx >= 0
        if not known.any():
            return out
        Rt = self.R[top][:, idx[known]]
        num = Rt.T @ sim
        den = (Rt != 0).astype(np.float64).T @ sim
        out[known] = num / (den + DEV_SHRINK)
        return out


class UknnScorer(HybridScorer):
    """The production scorer plus weight x the neighbours' deviation."""
    nbrs: Neighbours
    nb: tuple | None
    weight: float

    def _plus(self, ids, base):
        return base + self.weight * self.nbrs.dev(self.nb, list(ids))

    def predict(self, rows):
        return self._plus([r["mal_id"] for r in rows], super().predict(rows))

    def predict_ids(self, ids):
        return self._plus(ids, super().predict_ids(ids))


def residual_metrics(real):
    """bench.user_metrics plus how dev relates to the future ratings and to
    the shown score's errors (all of them, and where >= 20 neighbours rated)."""
    from malrec.eval import spearman

    def um(scorer, top, items, truth, liked, disliked):
        out = real(scorer, top, items, truth, liked, disliked)
        pred = scorer.calibrate(HybridScorer.predict_ids(scorer, items))
        dev = scorer.nbrs.dev(scorer.nb, list(items))
        ok = ~np.isnan(pred) & (dev != 0)
        if ok.sum() >= 8 and dev[ok].std() > 0:
            out["dev_rho"] = float(spearman(dev[ok], truth[ok]))
            out["res_rho"] = float(spearman(dev[ok], (truth - pred)[ok]))
        if scorer.nb is not None and len(scorer.nb[0]):
            idx = np.array([scorer.nbrs.pop.index.get(x, -1) for x in items])
            top_rows = scorer.nbrs.R[scorer.nb[0]]
            n = np.zeros(len(items))
            kn = idx >= 0
            n[kn] = np.asarray((top_rows[:, idx[kn]] != 0).sum(axis=0)).ravel()
            sup = ok & (n >= 20)
            if sup.sum() >= 8 and dev[sup].std() > 0:
                out["res_rho_supp"] = float(spearman(dev[sup], (truth - pred)[sup]))
        return out
    return um


def main(weights: list[float], k: int, max_eval: int, residual: bool = False):
    t0 = time.time()
    global METRICS
    if residual:
        weights, METRICS = [0.0], RESIDUAL
        bench.user_metrics = residual_metrics(bench.user_metrics)
    entries, static = bench.load_snapshot()
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    A = users[:int(len(users) * 0.72)]
    real_make = bench.make_scorer
    if not (bench.DATA / f"pop-{bench.code_hash()}.pkl").exists():
        raise SystemExit("run the benchmark first: the population model is not cached")
    (pop, _), _ = bench.population(entries, static["store"], A, [])
    nbrs = Neighbours(entries, pop, A)
    print(f"neighbours: {nbrs.R.shape[0]} lists, {nbrs.R.nnz:,} ratings [{time.time() - t0:.0f}s]",
          flush=True)

    results = []
    for wt in weights:
        def make(pop, stacker, store, sc, im, li, calibration, wt=wt):
            s = real_make(pop, stacker, store, sc, im, li, calibration, UknnScorer)
            s.nbrs, s.weight = nbrs, wt
            s.nb = nbrs.of({e.mal_id: e.score for e in sc}, {e.mal_id: e.weight for e in sc}, k)
            return s
        bench.make_scorer = make
        try:
            out = bench.run(max_eval)
        finally:
            bench.make_scorer = real_make
        results.append((wt, out))
        print(f"  weight {wt:g} done [{time.time() - t0:.0f}s]", flush=True)

    base = results[0][1]
    for g in base["groups"]:
        print(f"\n== {g}")
        print(f"   {'weight':<8}" + "".join(f"{m:>17}" for m, _ in METRICS) + f"{'pop#':>7}")
        for wt, out in results:
            gu, bu = out["groups"][g]["users"], base["groups"][g]["users"]
            cells = []
            for m, higher in METRICS:
                pairs = [(gu[u][m], bu[u][m]) for u in gu if u in bu and m in gu[u] and m in bu[u]]
                now = float(np.mean([x for x, _ in pairs])) if pairs else float("nan")
                if out is base or len(pairs) < 3:
                    cells.append(f"{now:17.4f}")
                    continue
                d = np.array([x - y for x, y in pairs])
                se = float(d.std(ddof=1) / np.sqrt(len(d)))
                flag = ("+" if (d.mean() > 0) == higher else "-") if se > 0 and abs(d.mean()) > 2 * se else ""
                cells.append(f"{now:9.4f}{d.mean():+7.4f}{flag:1}")
            pop = out["groups"][g].get("pop")
            print(f"   {wt:<8g}" + "".join(cells) + f"{pop if pop is not None else float('nan'):7.0f}")
    print(f"\n(k={k}; +/- : better/worse than weight {weights[0]:g} by more than two standard "
          f"errors, paired)\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--residual", action="store_true")
    a = ap.parse_args()
    main(a.weights, a.k, a.users, a.residual)
