"""Experiment 3: train a ranker instead of a score regressor.

Squared error on a 1-10 score is not the task. What matters is the order of
the top of the list. Two alternatives are tried against the ridge baseline:

  * a linear pairwise model (RankNet/RankSVM style): logistic regression over
    feature *differences*, which keeps the low variance of a linear model
    while optimising ordering directly. This is the one likely to work at
    ~150 training rows.
  * LambdaMART via LightGBM, which optimises nDCG directly but needs far more
    data than this.
"""
from __future__ import annotations

import numpy as np
from exp_implicit import HOLDOUTS, IMPLICIT, MU, ROWS, SCORED, recency, show
from sklearn.linear_model import LogisticRegression, Ridge

from malrec.eval import ndcg_at, spearman
from malrec.features import build_vocabulary, vectorise

# the configuration kept from experiment 2
OFFSETS = {"dropped": -1.5, "on_hold": -0.5, "completed": 0.0}
IMPLICIT_W = 0.5


def training_rows(cutoff, train_s):
    rows, ys, ws = [], [], []
    for r in train_s:
        row = ROWS.get(r["mal_id"])
        if row is None:
            continue
        rows.append(row); ys.append(float(r["score"])); ws.append(recency(r["at"], cutoff))
    for r in IMPLICIT:
        off = OFFSETS.get(r["status"])
        if off is None or r["at"] >= cutoff:
            continue
        row = ROWS.get(r["mal_id"])
        if row is None:
            continue
        rows.append(row); ys.append(float(np.clip(MU + off, 1, 10)))
        ws.append(IMPLICIT_W * recency(r["at"], cutoff))
    return rows, np.array(ys), np.array(ws)


def fit_ridge(X, y, w, **kw):
    est = Ridge(alpha=kw.get("alpha", 30.0)).fit(X, y, sample_weight=w)
    return lambda Z: est.predict(Z)


def fit_pairwise(X, y, w, C=0.05, max_pairs=20000, seed=0):
    """Logistic regression on feature differences.

    For every pair with different labels, (x_i - x_j) is a positive example
    and its negation a negative one, weighted by how far apart the two scores
    are and by how confident we are in both rows. The learned weight vector is
    a linear scorer, so inference is identical in cost to ridge.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = [(i, j) for i in range(n) for j in range(n) if y[i] > y[j]]
    if not idx:
        return fit_ridge(X, y, w)
    if len(idx) > max_pairs:
        pick = rng.choice(len(idx), max_pairs, replace=False)
        idx = [idx[k] for k in pick]

    i_arr = np.fromiter((a for a, _ in idx), int)
    j_arr = np.fromiter((b for _, b in idx), int)
    D = X[i_arr] - X[j_arr]
    pw = np.abs(y[i_arr] - y[j_arr]) * np.sqrt(w[i_arr] * w[j_arr])

    # symmetric so the model cannot learn an intercept shortcut
    Xp = np.vstack([D, -D])
    yp = np.concatenate([np.ones(len(D)), np.zeros(len(D))])
    wp = np.concatenate([pw, pw])

    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=2000, solver="lbfgs")
    clf.fit(Xp, yp, sample_weight=wp)
    coef = clf.coef_[0]
    return lambda Z: Z @ coef


def fit_lambdamart(X, y, w, **kw):
    import lightgbm as lgb
    # LambdaMART needs graded relevance, not a continuous target
    grades = np.clip(np.round(y).astype(int) - 5, 0, 5)
    est = lgb.LGBMRanker(objective="lambdarank", n_estimators=200, learning_rate=0.05,
                         num_leaves=7, min_child_samples=8, verbose=-1,
                         label_gain=list(range(64)))
    est.fit(X, grades, group=[len(y)], sample_weight=w)
    return lambda Z: est.predict(Z)


def evaluate(fitter, holdouts=HOLDOUTS, **kw) -> dict:
    rhos, nds, rmses = [], [], []
    for h in holdouts:
        train_s, test_s = SCORED[:-h], SCORED[-h:]
        cutoff = test_s[0]["at"]
        rows, ys, ws = training_rows(cutoff, train_s)
        te_rows = [ROWS[r["mal_id"]] for r in test_s if r["mal_id"] in ROWS]
        yte = np.array([float(r["score"]) for r in test_s if r["mal_id"] in ROWS])
        if len(rows) < 20 or len(te_rows) < 5:
            continue
        vocab = build_vocabulary(rows)
        Xtr, Xte = vectorise(rows, vocab), vectorise(te_rows, vocab)
        predict = fitter(Xtr, ys, ws, **kw)
        p = np.asarray(predict(Xte), dtype=float)
        rhos.append(spearman(p, yte))
        nds.append(ndcg_at(p, list(yte), 10))
        # a pairwise scorer has no natural scale, so RMSE is only meaningful
        # after the same linear calibration the product applies
        a, b = np.polyfit(p, yte, 1) if p.std() > 1e-9 else (1.0, 0.0)
        rmses.append(float(np.sqrt(np.mean((a * p + b - yte) ** 2))))
    return {"rho": float(np.mean(rhos)), "ndcg": float(np.mean(nds)),
            "rmse": float(np.mean(rmses)), "per": rhos}


if __name__ == "__main__":
    print(f"{'configuration':<44}{'rho':>7}{'nDCG@10':>8}{'rmse*':>8}")
    print("-" * 78)
    base = evaluate(fit_ridge)
    show("ridge regression (current)", base)
    print("\n-- linear pairwise, regularisation sweep --")
    for C in (0.005, 0.02, 0.05, 0.2, 1.0):
        show(f"pairwise logistic  C={C}", evaluate(fit_pairwise, C=C), base)
    print("\n-- LambdaMART --")
    show("lightgbm lambdarank", evaluate(fit_lambdamart), base)
    print("\n* rmse after linear calibration, so scales are comparable")
