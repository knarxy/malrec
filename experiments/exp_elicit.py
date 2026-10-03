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
A card for a title in their history with a score is answered with that score;
anything else counts as "haven't seen" (no information).

Reported after 5 / 10 / 15 ratings: rho on future ratings, recall@50 (plain
and IPS) of liked future titles, and cards shown per rating obtained.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from malrec.config import settings
from malrec.eval import spearman
from malrec.rank import relevance_bonus
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore

CFG = settings()
KS = (0, 5, 10, 15)
MAX_CARDS = 80
ASK_POOL = 1500          # candidate cards: the most popular titles


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
    um = fit_user(pop, stacker, store, scored, [], mode="sized", listed=listed)
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
        else:
            listed = {x[0]: 1.0 for x in known + got}
            fold = pop.fold_in({x[0]: x[1] for x in known + got if x[1] > 0}, listed=listed)
            rel = fold.signals(cand)["rel"]
            seen = np.array([prop.get(c, 1e-4) for c in cand]) * np.exp(z(rel))
            score = seen * np.array([info.get(c, 0.0) for c in cand])
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
    pop.fit_ease(np.array(oa), np.array(oi))
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: [(e["mal_id"], float(e["score"]), e["at"])
                                    for e in entries[u] if e["score"] > 0] for u in B},
                          store, seed=seed)
    # popularity (share of lists) and informativeness (sd of user-centred ratings)
    lists = defaultdict(set)
    for u_, m_ in zip(oa, oi):
        lists[m_].add(u_)
    prop = {m: (len(s) + 1) / (len(A) + 2) for m, s in lists.items()}
    umean = defaultdict(list)
    for u_, s_ in zip(ua, ra):
        umean[u_].append(s_)
    umean = {u_: float(np.mean(v)) for u_, v in umean.items()}
    dev = defaultdict(list)
    for u_, m_, s_ in zip(ua, ia, ra):
        dev[m_].append(s_ - umean[u_])
    info = {m: float(np.std(v)) for m, v in dev.items() if len(v) >= 30}
    pool_all = pool_ids(store)
    ask = sorted((m for m in pool_all if m in info), key=lambda m: -prop.get(m, 0))[:ASK_POOL]
    print(f"{len(users)} users; {len(ask)} candidate cards [{time.time() - t0:.0f}s]", flush=True)

    R = defaultdict(lambda: defaultdict(list))
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
        for strat in ("popular", "smart"):
            got, cards_at = run_round(strat, pop, store, known, answers, ask, prop, info, max(KS))
            for k in KS:
                if k and len(got) < k:
                    continue
                rho, rec, ips = evaluate(pop, stacker, store, known + got[:k], cutoff, pool,
                                         items, truth, targets, prop)
                key = f"{strat} k={k}" if k else "no round"
                if rho is not None:
                    R[key]["rho"].append(rho)
                R[key]["rec"].append(rec); R[key]["ips"].append(ips)
                if k:
                    R[key]["cards"].append(cards_at[k])
            R[f"{strat} reached 15"]["n"].append(1.0 if len(got) >= 15 else 0.0)
    print(f"\n{n_users} users [{time.time() - t0:.0f}s]")
    print(f"{'':<18}{'rho':>7}{'recall@50':>11}{'IPS':>7}{'cards':>7}{'users':>7}")
    for name, v in R.items():
        if "reached" in name:
            print(f"{name:<18}{'':>32}{np.mean(v['n']):7.0%}")
            continue
        cards = f"{np.mean(v['cards']):7.1f}" if v["cards"] else f"{'':>7}"
        print(f"{name:<18}{np.mean(v['rho']):7.3f}{np.mean(v['rec']):11.3f}"
              f"{np.mean(v['ips']):7.3f}{cards}{len(v['rec']):7d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
