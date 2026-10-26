"""Simulated rating round for new users: which titles should we ask about?

Held-out sampled users with a long history play a new user: the model knows
10 random entries of their list; the rest of their history is what they have
"seen" (their real scores answer the cards); their newest 20% of ratings are
the future the result is scored on and are never offered.

Strategies pick the next card from well-known titles not yet on the known
list:
  popular   most popular first
  smart     P(seen) x informativeness:
              P(seen)   popularity (share of lists) times how common the title
                        is on lists like theirs (co-occurrence relevance)
              informative  spread of centred ratings among people who rated it
                        (a title everyone loves tells us little)
            re-scored after every answer, as the relevance changes
  seen      P(seen) alone: popularity times co-occurrence relevance, no
            informativeness - the titles this user most likely watched
  franchise popular, at most one title per franchise (asked or known)
  genre     popular, the next card from the least-asked main genre among
            the 30 most popular left
A card for a title in their history with a score is answered with that score;
anything else counts as "haven't seen" (no information).

Reported after 5 / 10 / 15 ratings: rho on future ratings, recall@50 (plain
and IPS) of liked future titles, and cards shown per rating obtained. Data:
the benchmark's snapshot, population model and held-out users (bench.py);
the model is fitted as served (service.blend_opts), each strategy paired
with "popular" on the same users.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from experiments import bench
from experiments.exp_retrieval import pool_ids
from malrec.config import settings
from malrec.eval import spearman
from malrec.rank import relevance_bonus
from malrec.recsys.hybrid import Example, fit_user, recency_weight, split_user
from malrec.recsys.service import blend_opts

CFG = settings()
KS = (0, 5, 10, 15)
MAX_CARDS = 80
ASK_POOL = 1500          # candidate cards: the most popular titles
STRATEGIES = ("popular", "smart", "seen", "franchise", "genre")


class _Share:
    def __init__(self, lam):
        self.personal_share = lam


def z(x):
    x = np.asarray(x, dtype=float)
    sd = float(np.nanstd(x))
    return (x - np.nanmean(x)) / sd if sd > 1e-9 else np.zeros_like(x)


def evaluate(pop, stacker, store, known, cutoff, pool, test_items, truth, targets, prop):
    scored = [Example(m, s, recency_weight(a, cutoff), a) for m, s, a in known if s > 0]
    listed = {m: recency_weight(a, cutoff) for m, _, a in known}
    um = fit_user(pop, stacker, store, scored, [], mode="sized", listed=listed, **blend_opts())
    p = um.predict(test_items)
    ok = ~np.isnan(p)
    rho = spearman(p[ok], truth[ok]) if ok.sum() >= 5 and truth[ok].std() > 0 else None
    key = um.predict(pool) + relevance_bonus(_Share(um.lam), um.relevance(pool))
    s = np.where(np.isnan(key), -np.inf, key)
    top = {pool[i] for i in np.argsort(-s)[:50]}
    hit = top & targets
    w = {t: 1 / prop.get(t, 1e-3) for t in targets}
    return rho, len(hit) / len(targets), sum(w[t] for t in hit) / sum(w.values())


def run_round(strategy, pop, store, known, answers, ask, prop, info, k_max):
    """Returns the elicited (mal_id, score, at) list and cards shown per k."""
    asked, got, cards_at = set(), [], {}
    cards = 0
    while len(got) < k_max and cards < MAX_CARDS:
        cand = [m for m in ask if m not in asked and m not in {x[0] for x in known + got}]
        if not cand:
            break
        if strategy == "popular":
            m = cand[0]
        elif strategy == "franchise":
            seen_fr = {(store.rows.get(x) or {}).get("franchise_id") or x
                       for x in [y[0] for y in known + got] + list(asked)}
            m = next((c for c in cand if ((store.rows.get(c) or {}).get("franchise_id") or c)
                      not in seen_fr), cand[0])
        elif strategy == "genre":
            asked_g = defaultdict(int)
            for x in asked:
                asked_g[((store.rows.get(x) or {}).get("mal_genres") or ["-"])[0]] += 1
            m = min(cand[:30], key=lambda c: (asked_g[((store.rows.get(c) or {})
                                                        .get("mal_genres") or ["-"])[0]],
                                              cand.index(c)))
        else:
            listed = {x[0]: 1.0 for x in known + got}
            fold = pop.fold_in({x[0]: x[1] for x in known + got if x[1] > 0}, listed=listed)
            rel = fold.signals(cand)["rel"]
            seen = np.array([prop.get(c, 1e-4) for c in cand]) * np.exp(z(rel))
            score = seen if strategy == "seen" else seen * np.array([info.get(c, 0.0)
                                                                     for c in cand])
            m = cand[int(np.argmax(score))]
        asked.add(m)
        cards += 1
        if m in answers:
            got.append(answers[m])
            cards_at[len(got)] = cards
    return got, cards_at


def main(max_eval: int, seed: int):
    t0 = time.time()
    CFG.relevance_weight_personal = 0.5
    entries, static = bench.load_snapshot()
    store = static["store"]
    bench.GROUPS = bench.groupings(store)       # the live franchises, as the benchmark
    brng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    brng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = bench.population(entries, store, A, B)
    print(f"population model {how} [{time.time() - t0:.0f}s]", flush=True)
    rng = np.random.default_rng(seed)
    # popularity (share of lists) and informativeness (sd of user-centred ratings)
    ua, ia, ra = entries.columns(A)
    prop = dict(enumerate(((np.bincount(ia.astype(np.int64)) + 1.0) / (len(A) + 2)).tolist()))
    r = ra > 0
    ua, ia, ra = ua[r].astype(np.int64), ia[r].astype(np.int64), ra[r].astype(float)
    sums, cnt = np.bincount(ua, ra), np.bincount(ua)
    dev = ra - (sums / np.maximum(cnt, 1))[ua]
    n_i = np.bincount(ia)
    m1, m2 = np.bincount(ia, dev), np.bincount(ia, dev ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd_i = np.sqrt(np.maximum(m2 / n_i - (m1 / n_i) ** 2, 0.0))
    info = {int(m): float(sd_i[m]) for m in np.nonzero(n_i >= 30)[0]}
    pool_all = pool_ids(store)
    ask = sorted((m for m in pool_all if m in info), key=lambda m: -prop.get(m, 0))[:ASK_POOL]
    print(f"{len(users)} users; {len(ask)} candidate cards [{time.time() - t0:.0f}s]", flush=True)

    R: dict = defaultdict(dict)                 # (strategy, k) -> user -> metrics
    n_users = 0
    for u in C:
        if n_users >= max_eval:
            break
        es = entries[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        if len(scored_all) < 60:
            continue
        _, test_s, cutoff = split_user(scored_all, rng)
        mean = float(np.mean([s for _, s, _ in scored_all]))
        targets = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
        test_ids = {m for m, _, _ in test_s}
        history = [(e["mal_id"], float(e["score"]), e["at"]) for e in es
                   if e["mal_id"] not in test_ids and e["at"] < cutoff]
        if len(history) < 40 or len(targets) < 3:
            continue
        n_users += 1
        idx = rng.permutation(len(history))
        known = [history[i] for i in idx[:10]]
        answers = {h[0]: h for h in (history[i] for i in idx[10:]) if h[1] > 0}
        pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
        items = [m for m, _, _ in test_s]
        truth = np.array([s for _, s, _ in test_s])
        for strat in STRATEGIES:
            got, cards_at = run_round(strat, pop, store, known, answers, ask, prop, info, max(KS))
            for k in KS:
                if (k and len(got) < k) or (not k and strat != STRATEGIES[0]):
                    continue
                rho, rec, ips = evaluate(pop, stacker, store, known + got[:k], cutoff, pool,
                                         items, truth, targets, prop)
                R[(strat, k)][u] = {"rho": rho, "rec": rec, "ips": ips,
                                    "cards": cards_at[k] if k else None}
            R[(strat, "reached")][u] = {"n": 1.0 if len(got) >= max(KS) else 0.0}
    print(f"\n{n_users} users [{time.time() - t0:.0f}s]; change vs popular at the same k,"
          " paired, +-2 SE")
    print(f"{'':<16}" + "".join(f"{m:>22}" for m in ("rho", "recall@50", "IPS", "cards")))
    for k in KS:
        for strat in STRATEGIES if k else STRATEGIES[:1]:
            res, base = R[(strat, k)], R[(STRATEGIES[0], k)]
            line = f"{strat + (f' k={k}' if k else ' none'):<16}"
            for m in ("rho", "rec", "ips", "cards"):
                us = [x for x in res if res[x][m] is not None and x in base
                      and base[x][m] is not None]
                if not us:
                    line += f"{'':>22}"
                    continue
                v = np.array([res[x][m] for x in us], dtype=float)
                d = v - np.array([base[x][m] for x in us], dtype=float)
                se = 2 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                line += f"{v.mean():8.3f}{d.mean():+7.3f}({se:.3f})"
            print(line)
    for strat in STRATEGIES:
        print(f"{strat} reaches {max(KS)} ratings within {MAX_CARDS} cards for"
              f" {np.mean([v['n'] for v in R[(strat, 'reached')].values()]):.0%} of the users")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
