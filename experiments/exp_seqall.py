"""Sequels everywhere: the Up Next anchor in every prediction.

Up Next predicts a sequel half by the model, half by "your score for the
previous season plus how raters of both typically move" (exp_sequel.py,
2026-09-27: RMSE 1.187 -> 1.052 on held-out sequels). Everywhere else - the
score on a title's page, the holdouts, the gate - the model alone decides,
and the reference profile's largest errors after the expansion were sequels
it rated far below the population (Tate no Yuusha S2/S3, Tensei Kenja S2;
exp_refitems.py).

Through the benchmark's production path (bench.run): for every title with a
direct main-format prequel the user model saw rated,
  anchor = mean over such prequels p of (r_p + delta(p, s))
  delta  = mean r_s - r_p over the benchmark's fit users who rated both,
           shrunk toward the global mean delta by SEQUEL_SHRINK pseudo-pairs
  shown  = (1 - w) * model + w * anchor        (on the shown scale)
Per group: every metric paired with the first arm, plus RMSE and rank
correlation pooled over the future ratings of such sequels (seq_rmse,
seq_rho).

How much of the prequel score carries over (the anchor's mode):
  flat    as Up Next: r_p + delta, all of it (2026-10-10: sampled sequels
          RMSE 1.13 -> 1.03 at 400 entries, but the reference profile's
          sequel order turned upside down, rho .70 -> -.32 - its 10s and 7s
          for a first season are followed by ratings near its mean)
  global  mu_u + beta (r_p - mu_u) + c(p, s): beta the fit users' carry-over
          (within each pair), c the pair's shift on top, shrunk like delta
  user    as global with the user's own beta from their seen pairs, shrunk
          toward the population's by BETA_SHRINK pseudo-pairs

  python -m experiments.exp_seqall                 flat:0 flat:0.35 global:0.35 ...
  python -m experiments.exp_seqall flat:0 user:0.5
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import scipy.sparse as sp

from experiments import bench
from malrec.db import query
from malrec.eval import spearman
from malrec.recsys.scorer import HybridScorer
from malrec.surfaces import SEQUEL_SHRINK

BETA_SHRINK = 5.0      # pseudo-pairs at the population's carry-over
METRICS = (("rho", True), ("rmse", False), ("r50", True), ("hit10", True), ("bad10", False))

PAIRS_SQL = """
    SELECT r.src AS p, r.dst AS s FROM relation r
      JOIN anime a ON a.mal_id = r.src JOIN anime b ON b.mal_id = r.dst
     WHERE r.relation_type = 'sequel'
       AND format_class(a.media_type) = 'main' AND format_class(b.media_type) = 'main'"""


class Carry:
    """What the fit users' prequel/sequel pairs say. For each pair (p, s)
    and user v who rated both, x = r_p - mu_v and y = r_s - mu_v:
      delta(p, s)   shrunk mean of y - x               (the flat anchor)
      beta          within-pair slope of y on x        (how much carries over)
      c(p, s)       shrunk mean of y - beta x          (the pair's shift)
      var_x         mean within-pair x^2, the unit of BETA_SHRINK"""

    def __init__(self, entries, A, pairs):
        u, m, sc = entries.columns(A)
        keep = sc > 0
        u, m, sc = u[keep], m[keep], sc[keep].astype(np.float64)
        uid, row = np.unique(u, return_inverse=True)
        mu = np.bincount(row, sc) / np.bincount(row)
        items = {x: k for k, x in enumerate(np.unique(m).tolist())}
        col = np.array([items[x] for x in m.tolist()])
        M = sp.csc_matrix((sc, (row, col)), shape=(len(uid), len(items)))
        xy = {}
        for p, s in pairs:
            if p not in items or s not in items:
                continue
            a, b = M[:, items[p]], M[:, items[s]]
            ra, rb = a.toarray().ravel(), b.toarray().ravel()
            v = np.flatnonzero((ra > 0) & (rb > 0))
            if len(v):
                xy[(p, s)] = (ra[v] - mu[v], rb[v] - mu[v])
        xs = np.concatenate([x for x, _ in xy.values()])
        ys = np.concatenate([y for _, y in xy.values()])
        self.delta_g = float((ys - xs).mean())
        self.delta = {k: (float((y - x).sum()) + SEQUEL_SHRINK * self.delta_g) / (len(x) + SEQUEL_SHRINK)
                      for k, (x, y) in xy.items()}
        wx = np.concatenate([x - x.mean() for x, _ in xy.values()])
        wy = np.concatenate([y - y.mean() for _, y in xy.values()])
        self.beta = float((wx * wy).sum() / (wx * wx).sum())
        self.var_x = float((wx * wx).mean())
        self.c_g = float((ys - self.beta * xs).mean())
        self.c = {k: (float((y - self.beta * x).sum()) + SEQUEL_SHRINK * self.c_g) / (len(x) + SEQUEL_SHRINK)
                  for k, (x, y) in xy.items()}
        self.pairs = len(xs)

    def user_beta(self, ratings: dict[int, float], mu: float, sequels_of) -> float:
        """The user's own carry-over from their seen pairs, shrunk toward beta."""
        num = den = 0.0
        for p, rp in ratings.items():
            for s in sequels_of.get(p, ()):
                if s in ratings:
                    x, y = rp - mu, ratings[s] - mu - self.c.get((p, s), self.c_g)
                    num, den = num + x * y, den + x * x
        k = BETA_SHRINK * self.var_x
        return (num + k * self.beta) / (den + k)


class SequelScorer(HybridScorer):
    """The production scorer with sequels pulled toward the anchor."""
    anchors: dict[int, float]
    weight: float

    def _mix(self, ids, raw):
        raw = np.asarray(raw, dtype=float)
        if not self.anchors or self.weight == 0:
            return raw
        a, b = self.calibration
        out = raw.copy()
        for k, m in enumerate(ids):
            an = self.anchors.get(m)
            if an is not None and not np.isnan(out[k]):
                shown = a * out[k] + b
                out[k] = ((1 - self.weight) * shown + self.weight * an - b) / a
        return out

    def predict(self, rows):
        return self._mix([r["mal_id"] for r in rows], super().predict(rows))

    def predict_ids(self, ids):
        return self._mix(list(ids), super().predict_ids(ids))


def main(arms: list[str], max_eval: int):
    t0 = time.time()
    entries, _ = bench.load_snapshot()
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    A = users[:int(len(users) * 0.72)]
    pairs = [(r["p"], r["s"]) for r in query(PAIRS_SQL)]
    carry = Carry(entries, A, pairs)
    sequels_of = defaultdict(list)
    for p, s in pairs:
        sequels_of[p].append(s)
    print(f"{len(pairs)} sequel pairs, {len(carry.delta)} rated by fit users ({carry.pairs:,} ratings);"
          f" global delta {carry.delta_g:+.2f}, carry-over beta {carry.beta:.2f}"
          f" [{time.time() - t0:.0f}s]", flush=True)
    betas: dict[str, list[float]] = defaultdict(list)

    def anchors_for(ratings: dict[int, float], mode: str) -> dict[int, float]:
        mu = float(np.mean(list(ratings.values()))) if ratings else 7.0
        beta = (carry.beta if mode == "global"
                else carry.user_beta(ratings, mu, sequels_of) if mode == "user" else 1.0)
        betas[mode].append(beta)
        acc = defaultdict(list)
        for p, r in ratings.items():
            for s in sequels_of.get(p, ()):
                if mode == "flat":
                    acc[s].append(r + carry.delta.get((p, s), carry.delta_g))
                else:
                    acc[s].append(mu + beta * (r - mu) + carry.c.get((p, s), carry.c_g))
        return {s: float(np.mean(v)) for s, v in acc.items()}

    real_make, real_um = bench.make_scorer, bench.user_metrics

    def um(scorer, top, items, truth, liked, disliked):
        out = real_um(scorer, top, items, truth, liked, disliked)
        k = [i for i, m in enumerate(items) if m in scorer.anchors]
        if k:
            shown = scorer.calibrate(scorer.predict_ids([items[i] for i in k]))
            out["seq"] = list(zip(shown.tolist(), truth[k].tolist()))
        return out

    results = []
    for arm in arms:
        mode, w = arm.split(":")

        def make(pop, stacker, store, sc, im, li, calibration, wt=float(w), mode=mode):
            s = real_make(pop, stacker, store, sc, im, li, calibration, SequelScorer)
            s.weight = wt
            s.anchors = anchors_for({e.mal_id: e.score for e in sc}, mode)
            return s
        bench.make_scorer, bench.user_metrics = make, um
        try:
            out = bench.run(max_eval)
        finally:
            bench.make_scorer, bench.user_metrics = real_make, real_um
        results.append((arm, out))
        print(f"  {arm} done [{time.time() - t0:.0f}s]", flush=True)

    base = results[0][1]
    for g in base["groups"]:
        print(f"\n== {g}")
        print(f"   {'arm':<12}" + "".join(f"{m:>17}" for m, _ in METRICS)
              + f"{'sequels':>9}{'seq_rmse':>10}{'seq_rho':>9}")
        for arm, out in results:
            gu, bu = out["groups"][g]["users"], base["groups"][g]["users"]
            cells = []
            for m, higher in METRICS:
                pairs_ = [(gu[u][m], bu[u][m]) for u in gu if u in bu and m in gu[u] and m in bu[u]]
                now = float(np.mean([x for x, _ in pairs_])) if pairs_ else float("nan")
                if out is base or len(pairs_) < 3:
                    cells.append(f"{now:17.4f}")
                    continue
                d = np.array([x - y for x, y in pairs_])
                se = float(d.std(ddof=1) / np.sqrt(len(d)))
                flag = ("+" if (d.mean() > 0) == higher else "-") if se > 0 and abs(d.mean()) > 2 * se else ""
                cells.append(f"{now:9.4f}{d.mean():+7.4f}{flag:1}")
            seq = [x for u in gu.values() for x in u.get("seq", [])]
            if len(seq) >= 5:
                p, t = np.array(seq).T
                tail = f"{len(seq):9d}{np.sqrt(np.mean((p - t) ** 2)):10.3f}{spearman(p, t):9.3f}"
            else:
                tail = f"{len(seq):9d}"
            print(f"   {arm:<12}" + "".join(cells) + tail)
    for mode, b in betas.items():
        q = np.percentile(b, [10, 50, 90])
        print(f"carry-over in mode {mode}: 10/50/90 % of users {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f}")
    print(f"\n(+/- : better/worse than {arms[0]} by more than two standard errors, "
          f"paired; seq_*: pooled over future ratings of sequels with a seen prequel)"
          f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", default=["flat:0", "flat:0.35", "global:0.35",
                                                 "global:0.5", "user:0.35", "user:0.5"])
    ap.add_argument("--users", type=int, default=100)
    a = ap.parse_args()
    main(a.arms, a.users)
