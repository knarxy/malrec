"""Does a population layer fix small users without hurting large ones?

Protocol
--------
Sampled MAL users are split three ways, by user:
    A  (70%)  fit the population model (bias, item-kNN, matrix factorisation)
    B  (15%)  fit the stacker that blends those signals
    C  (15%)  evaluation only - never seen by anything that is fitted

For each C user the most recent 20% of scored ratings are the test set
(random 20% for users whose dates are all one bulk import). To measure how
each model copes with a *new* user, it is then given only `budget` of that
user's older ratings - 10, 25, 60, 150 - while the test set stays fixed.

Everything here runs in memory: features come from ItemStore, which is
verified to reproduce the production SQL features exactly.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import time
from collections import defaultdict

import numpy as np
from sklearn.linear_model import Ridge

from malrec.config import settings
from malrec.db import query
from malrec.eval import ndcg_at, spearman
from malrec.features import build_vocabulary, vectorise
from malrec.model import choose_alpha
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.items import ItemStore

CFG = settings()
RNG = np.random.default_rng(7)
BUDGETS = (10, 25, 60, 150)
STACK_FEATURES = ("bias", "knn", "mf", "support", "known", "mal_c", "al_c", "pop", "knn_x_support")


def recency(at: dt.datetime, cutoff: dt.datetime) -> float:
    age = max((cutoff - at).total_seconds() / 31_557_600.0, 0.0)
    return CFG.recency_floor + (1 - CFG.recency_floor) * 0.5 ** (age / CFG.recency_half_life_years)


# ------------------------------------------------------------------ data --

def load_cf(min_scored: int = 5):
    rows = query("""
        SELECT r.user_id, r.mal_id, r.score,
               coalesce(r.finished_at::timestamptz, r.updated_at) AS at
          FROM cf_rating r JOIN cf_user u ON u.id = r.user_id
         WHERE u.state = 'done' AND u.n_scored >= %s AND r.score > 0
    """, (min_scored,))
    by_user: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append((r["mal_id"], float(r["score"]),
                                      r["at"] or dt.datetime(2000, 1, 1, tzinfo=dt.UTC)))
    return by_user


def split_user(ratings: list[tuple], test_frac: float = 0.2):
    """Temporal split, unless the dates are one bulk import."""
    days = defaultdict(int)
    for _, _, at in ratings:
        days[at.date()] += 1
    bulk = max(days.values()) / len(ratings) > 0.5
    rs = list(ratings)
    if bulk:
        RNG.shuffle(rs)
    else:
        rs.sort(key=lambda x: x[2])
    k = max(3, int(round(len(rs) * test_frac)))
    train, test = rs[:-k], rs[-k:]
    cutoff = max(x[2] for x in rs) if bulk else test[0][2]
    return train, test, cutoff


# -------------------------------------------------------------- features --

def stack_matrix(sig: dict, items: list[int], store: ItemStore) -> np.ndarray:
    mal = np.array([(store.rows.get(m, {}).get("mal_mean") or 7.5) - 7.5 for m in items])
    al = np.array([((store.rows.get(m, {}).get("al_average_score") or 75) - 75) / 10 for m in items])
    pop = np.array([math.log10(max(store.rows.get(m, {}).get("mal_popularity") or 5000, 1)) - 3
                    for m in items])
    return np.column_stack([sig["bias"], sig["knn"], sig["mf"], sig["support"], sig["known"],
                            mal, al, pop, sig["knn"] * sig["support"]])


def fit_stacker(pop: PopulationModel, users: dict, store: ItemStore) -> Ridge:
    X, y = [], []
    for ratings in users.values():
        if len(ratings) < 10:
            continue
        train, test, cutoff = split_user(ratings)
        # vary how much the stacker sees, so it learns to trust signals less
        # when they come from a thin list
        n = int(RNG.choice([10, 25, 60, 150, 10_000]))
        train = [train[i] for i in RNG.permutation(len(train))[:n]]
        fi = pop.fold_in({m: s for m, s, _ in train}, {m: recency(a, cutoff) for m, _, a in train})
        items = [m for m, _, _ in test]
        X.append(stack_matrix(fi.signals(items), items, store))
        y.append(np.array([s for _, s, _ in test]) - fi.mu)
    return Ridge(alpha=5.0).fit(np.vstack(X), np.concatenate(y))


# ------------------------------------------------------------ predictors --

def personal_ridge(store, train, test_items, cutoff, offset_train=None, offset_test=None,
                   min_alpha: float = 0.0):
    """The production per-user model (content features, recency weights,
    alpha by CV). With offsets it fits the residual over another model."""
    w = {m: recency(a, cutoff) for m, _, a in train}
    ratings = {m: s for m, s, _ in train}
    taste = store.taste_vector(ratings, w)
    tr_rows = store.user_rows([m for m, _, _ in train], taste)
    te_rows = store.user_rows(test_items, taste)
    keep_tr = {r["mal_id"] for r in tr_rows}
    trn = [(m, s, a) for m, s, a in train if m in keep_tr]
    if len(trn) < 3 or len(te_rows) != len(test_items):
        return None
    vocab = build_vocabulary(tr_rows)
    Xtr, Xte = vectorise(tr_rows, vocab), vectorise(te_rows, vocab)
    y = np.array([s for _, s, _ in trn])
    ww = np.array([w[m] for m, _, _ in trn])
    if offset_train is not None:
        idx = {m: k for k, (m, _, _) in enumerate(train)}
        y = y - np.array([offset_train[idx[m]] for m, _, _ in trn])
    alpha = choose_alpha(Xtr, y, ww, folds=min(10, len(y)))
    alpha = max(alpha, min_alpha)
    pred = Ridge(alpha=alpha).fit(Xtr, y, sample_weight=ww).predict(Xte)
    return pred + (offset_test if offset_test is not None else 0.0)


def evaluate_user(pop, stacker, store, train_full, test, cutoff, budget):
    train = [train_full[i] for i in RNG.permutation(len(train_full))[:budget]]
    items = [m for m, _, _ in test]
    truth = np.array([s for _, s, _ in test])
    w = {m: recency(a, cutoff) for m, _, a in train}
    fi = pop.fold_in({m: s for m, s, _ in train}, w)
    sig = fi.signals(items)
    preds = {
        "user mean": np.full(len(items), fi.mu),
        "mean + bias": fi.mu + sig["bias"],
        "+ kNN": fi.mu + sig["bias"] + sig["knn"],
        "+ MF": fi.mu + sig["bias"] + sig["mf"],
        "stack": fi.mu + stacker.predict(stack_matrix(sig, items, store)),
    }
    cur = personal_ridge(store, train, items, cutoff)
    if cur is not None:
        preds["current (per-user ridge)"] = cur
    # hybrid: the population stack, plus a personal residual on content
    tr_items = [m for m, _, _ in train]
    tr_sig = fi.signals(tr_items)
    tr_stack = fi.mu + stacker.predict(stack_matrix(tr_sig, tr_items, store))
    hyb = personal_ridge(store, train, items, cutoff, offset_train=tr_stack,
                         offset_test=preds["stack"])
    if hyb is not None:
        preds["stack + personal residual"] = hyb
    out = {}
    for name, p in preds.items():
        out[name] = {
            "rmse": float(np.sqrt(np.mean((p - truth) ** 2))),
            "rho": spearman(p, truth) if len(truth) >= 5 and truth.std() > 0 else None,
            "ndcg": ndcg_at(p, list(truth), 10) if len(truth) >= 5 else None,
        }
    return out


# ------------------------------------------------------------------ main --

def main(max_eval: int):
    t0 = time.time()
    data = load_cf()
    users = list(data)
    RNG.shuffle(users)
    nA, nB = int(len(users) * 0.70), int(len(users) * 0.15)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    print(f"{len(users)} sampled users: {len(A)} fit / {len(B)} stack / {len(C)} evaluate")

    ua, ia, ra = [], [], []
    for u in A:
        for m, s, _ in data[u]:
            ua.append(u); ia.append(m); ra.append(s)
    pop = PopulationModel(CFParams()).fit(np.array(ua), np.array(ia), np.array(ra))
    print(f"population model fitted in {time.time() - t0:.0f}s: {len(pop.items)} items")

    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: data[u] for u in B}, store)
    print("stacker weights:", dict(zip(STACK_FEATURES, np.round(stacker.coef_, 3))))

    results: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for u in C[:max_eval]:
        train_full, test, cutoff = split_user(data[u])
        for budget in BUDGETS:
            if len(train_full) < budget:
                continue
            for name, m in evaluate_user(pop, stacker, store, train_full, test, cutoff,
                                         budget).items():
                results[budget][name].append(m)

    for budget in BUDGETS:
        rs = results[budget]
        if not rs:
            continue
        n = len(next(iter(rs.values())))
        print(f"\n=== model given {budget} ratings  ({n} held-out users) ===")
        print(f"{'predictor':<30}{'rho':>8}{'nDCG@10':>9}{'RMSE':>8}")
        for name, ms in rs.items():
            rho = np.mean([m["rho"] for m in ms if m["rho"] is not None])
            nd = np.mean([m["ndcg"] for m in ms if m["ndcg"] is not None])
            rm = np.mean([m["rmse"] for m in ms])
            print(f"{name:<30}{rho:8.3f}{nd:9.3f}{rm:8.3f}")
    print(f"\n({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=400)
    main(ap.parse_args().max_eval)
