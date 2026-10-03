"""Retrieval: does the model surface what people actually go on to love?

Rating prediction on watched items (exp_arch.py) asks "given they watch it,
how much will they like it?". It cannot see the failure that motivated this
experiment: ranking the *catalogue* by that conditional rating hands every
newcomer the same devotee classics, because only devotees watch them.

Protocol, per held-out sampled user:
  * the newest 20% of their scored entries are the future; the ones they
    liked (>= max(their mean, 7)) are the targets
  * the model sees a random `budget` of their older list entries (any
    status - that is what a new user's list looks like)
  * the whole eligible catalogue is ranked, minus everything else on their
    list; recall@50 / @100 = share of liked future titles in the top 50 / 100
  * overlap = mean Jaccard between different users' top-20 (lower = the
    lists are actually personal)

The same is done for reference profile over its 8 temporal splits.
Novelty is off here: it trades recall for discovery on purpose and would
blur the comparison. Its effect is reported separately.
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import defaultdict

import numpy as np

from malrec.config import settings
from malrec.db import one, query
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore

CFG = settings()
BUDGETS = (10, 25, 60, 400)


def load_entries():
    rows = query("""
        SELECT r.user_id, r.mal_id, r.score, r.status,
               coalesce(r.finished_at::timestamptz, r.updated_at) AS at
          FROM cf_rating r JOIN cf_user u ON u.id = r.user_id
         WHERE u.state = 'done' AND r.updated_at IS NOT NULL""")
    by_user: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_user[r["user_id"]].append(r)
    return by_user


def pool_ids(store: ItemStore) -> list[int]:
    return [m for m, r in store.rows.items()
            if (r.get("mal_num_scoring_users") or 0) >= CFG.min_scoring_users
            and r.get("format_class") == "main" and (r.get("nsfw") in (None, "white"))
            and r.get("status") != "not_yet_aired"]


def zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.nanstd(x))
    return (x - np.nanmean(x)) / sd if sd > 1e-9 else np.zeros_like(x)


def rankers(pop, stacker, store, scored, implicit, listed, pool, betas):
    """Scores over the pool for each strategy (higher = recommend first)."""
    out = {}
    personal = fit_user(pop, stacker, store, scored, implicit, mode="personal")
    out["current (personal)"] = personal.predict(pool)
    stack = fit_user(pop, stacker, store, scored, implicit, mode="stack")
    # relevance needs every list entry, not just the rated ones
    stack.fold = pop.fold_in({e.mal_id: e.score for e in scored},
                             {e.mal_id: e.weight for e in scored}, listed=listed)
    sp = stack.predict(pool)
    rel = zscore(stack.fold.signals(pool)["rel"])
    out["stack"] = sp
    for b in betas:
        out[f"stack + {b:g}*relevance"] = sp + b * rel
    sized = fit_user(pop, stacker, store, scored, implicit, mode="sized")
    sized.fold = stack.fold
    lam = sized.lam
    zp = sized.predict(pool)
    for b in betas:
        out[f"sized + {b:g}*(1-lam)*rel"] = zp + b * (1 - lam) * rel
    out["_lam"] = lam
    return out


def recall(scores: np.ndarray, pool: list[int], targets: set[int], k: int) -> float | None:
    hits = [m for m in targets if m in set(pool)]
    if not hits:
        return None
    s = np.where(np.isnan(scores), -np.inf, scores)
    top = {pool[i] for i in np.argsort(-s)[:k]}
    return len(top & set(hits)) / len(hits)


def main(max_eval: int, seed: int, betas):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    entries = load_entries()
    users = sorted(u for u, es in entries.items() if sum(e["score"] > 0 for e in es) >= 15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.70), int(len(users) * 0.15)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]

    ua, ia, ra = [], [], []
    oa, oi = [], []
    for u in A:
        for e in entries[u]:
            oa.append(u); oi.append(e["mal_id"])
            if e["score"] > 0:
                ua.append(u); ia.append(e["mal_id"]); ra.append(float(e["score"]))
    pop = PopulationModel(CFParams()).fit(np.array(ua), np.array(ia), np.array(ra))
    pop.fit_occurrence(np.array(oa), np.array(oi))
    store = ItemStore.load()
    stack_data = {u: [(e["mal_id"], float(e["score"]), e["at"]) for e in entries[u] if e["score"] > 0]
                  for u in B}
    stacker = fit_stacker(pop, stack_data, store, seed=seed)
    pool_all = pool_ids(store)
    print(f"{len(users)} users ({len(A)}/{len(B)}/{len(C)}), pool {len(pool_all)} titles "
          f"[{time.time() - t0:.0f}s]")

    res: dict = defaultdict(lambda: defaultdict(lambda: {"r50": [], "r100": []}))
    tops: dict = defaultdict(lambda: defaultdict(list))
    for u in C[:max_eval]:
        es = entries[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        _, test_s, cutoff = split_user(scored_all, rng)
        mean = float(np.mean([s for _, s, _ in scored_all]))
        targets = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
        if len(targets) < 3:
            continue
        test_ids = {m for m, _, _ in test_s}
        history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
        known = {e["mal_id"] for e in es} - test_ids
        pool = [m for m in pool_all if m not in known]
        for budget in BUDGETS:
            if len(history) < budget:
                continue
            seen = [history[i] for i in rng.permutation(len(history))[:budget]]
            scored = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff),
                              e["at"]) for e in seen if e["score"] > 0]
            if len(scored) < 3:
                continue
            listed = {e["mal_id"]: recency_weight(e["at"], cutoff) for e in seen}
            m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
            implicit = [Example(e["mal_id"], float(min(max(m_ + CFG.implicit_offsets[e["status"]],
                                                            1), 10)),
                                CFG.implicit_weight * recency_weight(e["at"], cutoff), e["at"])
                        for e in seen if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
            for name, sc in rankers(pop, stacker, store, scored, implicit, listed, pool,
                                    betas).items():
                if name == "_lam":
                    continue
                r50, r100 = recall(sc, pool, targets, 50), recall(sc, pool, targets, 100)
                if r50 is not None:
                    res[budget][name]["r50"].append(r50)
                    res[budget][name]["r100"].append(r100)
                s = np.where(np.isnan(sc), -np.inf, sc)
                tops[budget][name].append({pool[i] for i in np.argsort(-s)[:20]})

    for budget in BUDGETS:
        rs = res[budget]
        if not rs:
            continue
        n = len(rs["current (personal)"]["r50"])
        print(f"\n== sampled users who have {budget} list entries ({n} users) ==")
        print(f"{'ranking':<34}{'recall@50':>10}{'recall@100':>11}{'overlap':>9}")
        for name, m in rs.items():
            sets = tops[budget][name]
            pairs = list(itertools.islice(itertools.combinations(range(len(sets)), 2), 3000))
            jac = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs]) \
                if pairs else float("nan")
            print(f"{name:<34}{np.mean(m['r50']):10.3f}{np.mean(m['r100']):11.3f}{jac:9.3f}")

    # ------------------------------------------------------ reference profile --
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    gate: dict = defaultdict(lambda: {"r50": [], "r100": []})
    for h in (25, 30, 35, 40, 45, 50, 55, 60):
        test = rated[-h:]
        cutoff = test[0]["at"]
        targets = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        test_ids = {r["mal_id"] for r in test}
        known = {r["mal_id"] for r in le} - test_ids
        pool = [m for m in pool_all if m not in known]
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        scored = [Example(r["mal_id"], float(r["score"]), recency_weight(r["at"], cutoff), r["at"])
                  for r in hist if r["score"] > 0]
        m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
        implicit = [Example(r["mal_id"], float(min(max(m_ + CFG.implicit_offsets[r["status"]], 1), 10)),
                            CFG.implicit_weight * recency_weight(r["at"], cutoff), r["at"])
                    for r in hist if r["score"] == 0 and r["status"] in CFG.implicit_offsets]
        listed = {r["mal_id"]: recency_weight(r["at"], cutoff) for r in hist}
        for name, sc in rankers(pop, stacker, store, scored, implicit, listed, pool, betas).items():
            if name == "_lam":
                continue
            r50 = recall(sc, pool, targets, 50)
            if r50 is not None:
                gate[name]["r50"].append(r50)
                gate[name]["r100"].append(recall(sc, pool, targets, 100))
    print("\n== reference profile: its future liked titles in his top-K (8 temporal splits) ==")
    print(f"{'ranking':<34}{'recall@50':>10}{'recall@100':>11}")
    for name, m in gate.items():
        print(f"{name:<34}{np.mean(m['r50']):10.3f}{np.mean(m['r100']):11.3f}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--betas", default="0.5,1,2")
    a = ap.parse_args()
    main(a.max_eval, a.seed, [float(b) for b in a.betas.split(",")])
