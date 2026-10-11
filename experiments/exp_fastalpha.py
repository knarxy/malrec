"""Can the ridge-penalty search be made fast without changing its answer?

choose_alpha() runs K-fold CV over every alpha with a full sklearn Ridge fit
each time - 89 % of a large account's training. Ridge with an intercept and
sample weights solves (Xc' W Xc + a I) b = Xc' W yc on weighted-centred data,
so one eigendecomposition of Xc' W Xc per fold gives the solution for every
alpha at once. Same maths, so the chosen alpha must be identical.

The inputs are the real ones: train_hybrid(persist=False) runs for each app
user with choose_alpha wrapped to record (X, y, w, folds) at every call.
Reported: calls, chosen alpha old/new (must match), largest CV-error
difference, and time old/new.
"""
from __future__ import annotations

import time

import numpy as np

from malrec import model
from malrec.db import query

ORIG = model.choose_alpha


def fast_choose_alpha(X, y, w, alphas=model.RIDGE_ALPHAS, folds: int = 10) -> tuple[float, list]:
    from sklearn.model_selection import KFold
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X, dtype=float)
    errs = np.zeros((len(alphas), 0)).tolist()
    per_alpha = [[] for _ in alphas]
    for tr, te in KFold(min(folds, len(y)), shuffle=True, random_state=0).split(X):
        Xt, yt, wt = X[tr], y[tr], w[tr]
        sw = wt / wt.sum()
        xm, ym = sw @ Xt, sw @ yt
        Xc, yc = Xt - xm, yt - ym
        s = np.sqrt(wt)
        A, bvec = Xc * s[:, None], yc * s
        G = A.T @ A
        evals, V = np.linalg.eigh(G)
        proj = V.T @ (A.T @ bvec)
        Xte = (X[te] - xm) @ V
        for k, a in enumerate(alphas):
            beta_v = proj / (evals + a)
            pred = Xte @ beta_v + ym
            r = pred - y[te]
            per_alpha[k].append(float(np.average(r ** 2, weights=w[te])))
    mean = [float(np.mean(e)) for e in per_alpha]
    best = alphas[int(np.argmin(mean))]
    return best, mean


def slow_errors(X, y, w, alphas=model.RIDGE_ALPHAS, folds: int = 10) -> list:
    from sklearn.model_selection import KFold
    out = []
    for a in alphas:
        errs = []
        for tr, te in KFold(min(folds, len(y)), shuffle=True, random_state=0).split(X):
            m = model._fit_ridge(X[tr], y[tr], w[tr], a)
            r = m.predict(X[te]) - y[te]
            errs.append(float(np.average(r ** 2, weights=w[te])))
        out.append(float(np.mean(errs)))
    return out


def main():
    users = query("SELECT u.id, u.mal_username FROM app_user u WHERE u.status='approved' ORDER BY id")
    print(f"{'user':<20}{'calls':>6}{'same':>6}{'max|dErr|':>12}{'old s':>8}{'new s':>8}  shape")
    for u in users:
        calls = []

        def rec(X, y, w, alphas=model.RIDGE_ALPHAS, folds=10):
            calls.append((X, y, w, folds))
            return ORIG(X, y, w, alphas=alphas, folds=folds)
        model.choose_alpha = rec
        try:
            model.train_hybrid(u["id"], persist=False)
        except Exception as e:                      # noqa: BLE001
            print(f"{u['mal_username']:<20} skipped: {e}")
            continue
        finally:
            model.choose_alpha = ORIG
        same, worst, t_old, t_new = 0, 0.0, 0.0, 0.0
        for X, y, w, folds in calls:
            t0 = time.perf_counter()
            a_old = ORIG(X, y, w, folds=folds)
            t_old += time.perf_counter() - t0
            t0 = time.perf_counter()
            a_new, e_new = fast_choose_alpha(X, y, w, folds=folds)
            t_new += time.perf_counter() - t0
            e_old = slow_errors(X, y, w, folds=folds)
            worst = max(worst, max(abs(p - q) for p, q in zip(e_old, e_new)))
            same += a_old == a_new
        shape = f"{calls[-1][0].shape}" if calls else "-"
        print(f"{u['mal_username']:<20}{len(calls):>6}{same:>6}{worst:>12.2e}{t_old:>8.2f}{t_new:>8.2f}  {shape}",
              flush=True)


if __name__ == "__main__":
    main()
