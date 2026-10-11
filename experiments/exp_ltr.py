"""Learning to rank: can a learned order beat the hand-tuned order key?

The Safe Bets order key is shown score + relevance bonus + popularity prior
- risk penalties (rank.order_terms), with ~18 weights set by hand over two
dozen experiments, each one at a time. Here a LightGBM LambdaMART ranker
learns how to combine those terms - and a few raw signals they are built
from - across all users at once.

Not the rejected per-user ranker (exp_rank.py: ~180 rows a user). This one
learns across users: every (user, list size) is one query of up to 600
candidates (production's shortlist), and the training users are the
benchmark's stacker users - held out from the population model, so their
signals are honest.

Features per candidate: shown score, relevance z, popularity prior,
the four risk parts (long, disagree, unfamiliar, acclaim), population and
personal predictions and their gap, the personal share, list size, genre
fit with the user's profile, MAL mean, log scorers, log popularity rank,
year, episodes, novelty.
Label: 2 = rated >= max(own mean + 1.5, 9), 1 = liked (>= max(own mean, 7)),
0 = everything else (most candidates were never rated).

Evaluation is the benchmark's: the same held-out users at 25 / 150 / 400
entries and the reference profile, the list built by production's rerank
(franchise collapse, genre calibration) from the shortlist ordered by
  PROD   the shipped key
  LTR    the ranker's score (put on the key's scale)
  MIX    the mean of the two, z-scored
paired per user against PROD.

  python -m experiments.exp_ltr [--train-users 600] [--users 100]
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np

from experiments import bench
from malrec import rank
from malrec.recsys.hybrid import split_user
from malrec.surfaces import SPECS

FEATURES = ("shown", "rel_z", "pop_prior", "risk_long", "risk_disagree", "risk_unfamiliar",
            "risk_acclaim", "pred_pop", "pred_personal", "pred_gap", "personal_share",
            "log_n", "genre_fit", "mal_mean", "log_scorers", "log_pop_rank", "year",
            "episodes", "novelty")
TRAIN_BUDGETS = (10, 25, 60, 150, 400)
SHORTLIST = 600


def z(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    return (v - np.mean(v)) / sd if sd > 1e-9 else np.zeros_like(v)


def query(scorer, rows, li, familiar, store, n_scored):
    """Production's shortlist for one user and the features of each title."""
    real = rank.familiar_genres
    rank.familiar_genres = lambda _uid: familiar
    try:
        t = rank.order_terms(scorer, rows, pop_prior=bench.CFG.safe_bets_popularity)
        ids = [r["mal_id"] for r in rows]
        order = [k for k in np.argsort(-t.key) if t.ok[k]][:SHORTLIST]
        sid = [ids[k] for k in order]
        parts = scorer.risk_parts(sid, familiar)
    finally:
        rank.familiar_genres = real
    st, pe = scorer.um.predict_parts(sid) if scorer.um.inner is not None else (
        scorer.um.predict(sid), np.full(len(sid), np.nan))
    prof = bench.profile_of(li, store)
    X = np.zeros((len(sid), len(FEATURES)), dtype=np.float32)
    for j, (k, m) in enumerate(zip(order, sid)):
        r = store.rows.get(m) or {}
        gs = r.get("mal_genres") or []
        X[j] = [t.shown[k], t.z[k], t.pop[k], parts["long"][j], parts["disagree"][j],
                parts["unfamiliar"][j], parts["acclaim"][j], st[j],
                pe[j] if not math.isnan(pe[j]) else st[j],
                (pe[j] - st[j]) if not math.isnan(pe[j]) else 0.0, t.lam,
                math.log1p(n_scored), sum(prof.get(g, 0.0) for g in gs) / max(len(gs), 1),
                r.get("mal_mean") or 7.0, math.log1p(r.get("mal_num_scoring_users") or 0),
                math.log10(max(r.get("mal_popularity") or 99999, 1)),
                (r.get("season_year") or 2010), (r.get("num_episodes") or 0),
                rank.novelty(r.get("mal_popularity"))]
    return sid, t.key[order], X


def label(sid, test: dict[int, float], mean: float) -> np.ndarray:
    hi, ok = max(mean + 1.5, 9.0), max(mean, 7.0)
    return np.array([2 if test.get(m, 0) >= hi else 1 if test.get(m, 0) >= ok else 0
                     for m in sid], dtype=np.int32)


def build_list(scorer, rows, li, familiar, store, key_by_id: dict[int, float] | None):
    """Production's Safe Bets list; with `key_by_id` the shortlist is ordered
    by that key instead (on the shipped key's scale), then reranked as usual."""
    real = rank.familiar_genres
    rank.familiar_genres = lambda _uid: familiar
    try:
        pool = rank.score_candidates(None, scorer, rows, enrich=False,
                                     pop_prior=bench.CFG.safe_bets_popularity)
    finally:
        rank.familiar_genres = real
    if key_by_id is not None:
        # rerank orders by predicted + relevance + bonus: carry the new key in
        # bonus, keep the shown score as it is
        pool = [replace(c, relevance=0.0, bonus=key_by_id.get(c.mal_id, -1e9) - c.predicted)
                for c in pool]
    spec = SPECS["safe_bets"]
    top = rank.rerank(pool, spec.novelty_weight, spec.diversity_weight, bench.LIMIT,
                      considered=[], profile=bench.profile_of(li, store))
    return [c.mal_id for c in top]


def main(n_train: int, max_eval: int):
    import lightgbm as lgb
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store, derived = static["store"], static["derived"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = bench.population(entries, store, A, B)
    pool_all = bench.eligible(store, derived)
    # propensity: the share of training lists a title is on. A hit on a title
    # everyone watches says little; IPS recall weights each liked title by
    # 1 / propensity, so following popularity alone does not score.
    _, ma, _ = entries.columns(A)
    lists_with = np.bincount(ma.astype(np.int64))

    def prop(m: int) -> float:
        return ((lists_with[m] if m < len(lists_with) else 0) + 1) / (len(A) + 2)

    def with_ips(m: dict, top: list[int], liked: set[int]) -> dict:
        if len(liked) >= 3:
            w = {t: 1.0 / prop(t) for t in liked}
            m["ips50"] = sum(w[t] for t in set(top[:50]) & liked) / sum(w.values())
        return m
    print(f"population model {how}; {len(B)} stacker users for training [{time.time() - t0:.0f}s]",
          flush=True)

    def genres(ids):
        return {g for m in ids for g in ((store.rows.get(m) or {}).get("mal_genres") or [])}

    # ------------------------------------------------------------ training --
    Xs, ys, groups, qusers = [], [], [], []
    trng = np.random.default_rng(bench.SEED + 11)
    for u in B[:n_train]:
        es = entries[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        if len(scored_all) < 20:
            continue
        _, test_s, cutoff = split_user(scored_all, trng)
        mean = float(np.mean([s for _, s, _ in scored_all]))
        test_ids = {m for m, _, _ in test_s}
        history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
        listed = {e["mal_id"] for e in es} - test_ids
        rows = [bench.row_of(store, m) for m in pool_all if m not in listed]
        for budget in trng.choice(TRAIN_BUDGETS, 2, replace=False):
            if len(history) < budget:
                continue
            seen = [history[i] for i in trng.permutation(len(history))[:budget]]
            sc, im, li = bench.examples(seen, cutoff)
            if len(sc) < 5:
                continue
            scorer = bench.make_scorer(pop, stacker, store, sc, im, li, (1.0, 0.0))
            sid, _, X = query(scorer, rows, li, genres(e["mal_id"] for e in seen), store, len(sc))
            y = label(sid, {m: s for m, s, _ in test_s}, mean)
            if y.sum() == 0:
                continue                    # a query with no positive teaches nothing
            Xs.append(X); ys.append(y); groups.append(len(y)); qusers.append(u)
    X, y = np.vstack(Xs), np.concatenate(ys)
    print(f"training: {len(groups)} queries, {len(y):,} candidates, "
          f"{int((y > 0).sum()):,} positives [{time.time() - t0:.0f}s]", flush=True)
    # hold out a fifth of the users for early stopping
    users_q = sorted(set(qusers))
    val_u = set(users_q[:: 5])
    tr = [k for k, u in enumerate(qusers) if u not in val_u]
    va = [k for k, u in enumerate(qusers) if u in val_u]
    ofs = np.cumsum([0] + groups)

    def take(idx):
        return (np.vstack([X[ofs[k]:ofs[k + 1]] for k in idx]),
                np.concatenate([y[ofs[k]:ofs[k + 1]] for k in idx]), [groups[k] for k in idx])
    Xt, yt, gt = take(tr)
    Xv, yv, gv = take(va)
    ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=600, learning_rate=0.03,
                            num_leaves=15, min_child_samples=100, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=5.0, label_gain=[0, 1, 3],
                            lambdarank_truncation_level=60, verbose=-1)
    ranker.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10, 50],
               callbacks=[lgb.early_stopping(50, verbose=False)])
    imp = sorted(zip(FEATURES, ranker.booster_.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(v for _, v in imp) or 1.0
    print(f"ranker: {ranker.best_iteration_} trees; gain share: "
          + ", ".join(f"{n} {v / tot:.0%}" for n, v in imp[:8]) + f" [{time.time() - t0:.0f}s]",
          flush=True)

    # ---------------------------------------------------------- evaluation --
    arms = ("PROD", "LTR", "MIX")

    def lists(scorer, rows, li, fam, n_scored):
        sid, key, Xq = query(scorer, rows, li, fam, store, n_scored)
        s = ranker.predict(Xq)
        on_key = float(np.mean(key)) + float(np.std(key)) * z(s)
        mix = float(np.mean(key)) + float(np.std(key)) * 0.5 * (z(key) + z(s))
        return {"PROD": build_list(scorer, rows, li, fam, store, None),
                "LTR": build_list(scorer, rows, li, fam, store, dict(zip(sid, on_key))),
                "MIX": build_list(scorer, rows, li, fam, store, dict(zip(sid, mix)))}

    def report(name, M, tops):
        print(f"\n== {name} ({len(M['PROD'])})")
        print(f"   {'arm':<6}" + "".join(f"{k:>17}" for k in ("r50", "ips50", "hit10", "bad10"))
              + f"{'overlap':>9}{'pop#':>7}")
        for arm in arms:
            cells = []
            for k, higher in (("r50", True), ("ips50", True), ("hit10", True), ("bad10", False)):
                pairs = [(a[k], b[k]) for a, b in zip(M[arm], M["PROD"]) if k in a and k in b]
                now = np.mean([a for a, _ in pairs]) if pairs else float("nan")
                if arm == "PROD" or not pairs:
                    cells.append(f"{now:17.4f}")
                    continue
                d = np.array([a - b for a, b in pairs])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
                good = d.mean() > 0 if higher else d.mean() < 0
                mark = ("+" if good else "-") if abs(d.mean()) > 2 * se else " "
                cells.append(f"{now:9.4f}{d.mean():+7.4f}{mark}")
            sets = [set(t) for t in tops[arm]]
            pr = list(itertools.islice(itertools.combinations(range(len(sets)), 2), 3000))
            ov = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pr]) \
                if pr else float("nan")
            pp = np.median([store.rows[m].get("mal_popularity") or 99999
                            for t in tops[arm] for m in t])
            print(f"   {arm:<6}{''.join(cells)}{ov:9.3f}{pp:7.0f}", flush=True)

    for budget in bench.BUDGETS:
        M, tops = defaultdict(list), defaultdict(list)
        urng = np.random.default_rng(bench.SEED + budget)
        for u in C:
            if len(M["PROD"]) >= max_eval:
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
            rows = [bench.row_of(store, m) for m in pool_all if m not in listed]
            scorer = bench.make_scorer(pop, stacker, store, sc, im, li, (1.0, 0.0))
            for arm, top in lists(scorer, rows, li, genres(e["mal_id"] for e in seen),
                                  len(sc)).items():
                M[arm].append(with_ips(bench.list_metrics(top, liked, disliked, ""), top, liked))
                tops[arm].append(top[:20])
        report(f"sampled {budget}", M, tops)
        print(f"   [{time.time() - t0:.0f}s]", flush=True)

    ref = static["ref"]
    le = ref["le"]
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    M, tops = defaultdict(list), defaultdict(list)
    for h in bench.REF_SPLITS:
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        sc, im, li = bench.examples(hist, cutoff)
        listed = {r["mal_id"] for r in le} - test_ids
        rows = [bench.row_of(store, m) for m in pool_all if m not in listed]
        scorer = bench.make_scorer(pop, stacker, store, sc, im, li, ref["cal"])
        liked = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        for arm, top in lists(scorer, rows, li, genres(r["mal_id"] for r in hist),
                              len(sc)).items():
            M[arm].append(with_ips(bench.list_metrics(
                top, liked, {r["mal_id"] for r in test if r["score"] < mean - 1}, ""),
                top, liked))
            tops[arm].append(top[:20])
    report("reference (8 splits)", M, tops)
    print("   (+/- : better/worse than PROD by more than two standard errors, paired)")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-users", type=int, default=600)
    ap.add_argument("--users", type=int, default=100)
    a = ap.parse_args()
    main(a.train_users, a.users)
