"""Which architecture? Compared on both populations that matter.

  1. sampled MAL users held out from everything fitted, bucketed by how many
     ratings the model is given (10 / 25 / 60 / 150) - the "new user" question
  2. the reference profile's 8 temporal splits - the "no losses" gate

Every mode goes through recsys.hybrid.fit_user, i.e. the code that would ship.
Implicit rows (dropped / on-hold / finished-unrated) are included for both
populations, dated, and only if they predate the cutoff.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from experiments.exp_population import fit_stacker, load_cf, split_user
from malrec.config import settings
from malrec.db import one, query
from malrec.eval import _rated_in_time_order, ndcg_at, spearman
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_user
from malrec.recsys.items import ItemStore

CFG = settings()
RNG = np.random.default_rng(11)
BUDGETS = (10, 25, 60, 150)
H = (25, 30, 35, 40, 45, 50, 55, 60)

VARIANTS = {
    "personal (current)": {"mode": "personal"},
    "stack": {"mode": "stack"},
    "hierarchical": {"mode": "hierarchical"},
    "augmented": {"mode": "augmented"},
    "blend(personal)": {"mode": "blend", "blend_of": "personal"},
    "blend(augmented)": {"mode": "blend", "blend_of": "augmented"},
}


def recency(at, cutoff) -> float:
    age = max((cutoff - at).total_seconds() / 31_557_600.0, 0.0)
    return CFG.recency_floor + (1 - CFG.recency_floor) * 0.5 ** (age / CFG.recency_half_life_years)


def load_cf_implicit(users: list[int]) -> dict[int, list[tuple]]:
    rows = query("""SELECT user_id, mal_id, status,
                           coalesce(finished_at::timestamptz, updated_at) AS at
                      FROM cf_rating WHERE score = 0 AND user_id = ANY(%s)
                       AND status = ANY(%s)""", (users, list(CFG.implicit_offsets)))
    out: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        if r["at"] is not None:
            out[r["user_id"]].append((r["mal_id"], r["status"], r["at"]))
    return out


def implicit_examples(imp, cutoff, mean) -> list[Example]:
    return [Example(m, float(min(max(mean + CFG.implicit_offsets[s], 1), 10)),
                    CFG.implicit_weight * recency(at, cutoff), at)
            for m, s, at in imp if at < cutoff]


def score_variants(pop, stacker, store, scored, imp, items):
    out = {}
    for name, opts in VARIANTS.items():
        um = fit_user(pop, stacker, store, scored, imp, **opts)
        out[name] = um.predict(items)
    return out


def metrics(pred, truth):
    ok = ~np.isnan(pred)
    p, t = pred[ok], truth[ok]
    if len(t) < 5:
        return None
    return {"rho": spearman(p, t) if t.std() > 0 else None,
            "ndcg": ndcg_at(p, list(t), 10),
            "rmse": float(np.sqrt(np.mean((p - t) ** 2)))}


def run(max_eval: int, seed: int):
    t0 = time.time()
    data = load_cf()
    users = sorted(data)
    np.random.default_rng(seed).shuffle(users)
    nA, nB = int(len(users) * 0.70), int(len(users) * 0.15)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    ua, ia, ra = [], [], []
    for u in A:
        for m, s, _ in data[u]:
            ua.append(u); ia.append(m); ra.append(s)
    pop = PopulationModel(CFParams()).fit(np.array(ua), np.array(ia), np.array(ra))
    store = ItemStore.load()
    stacker = fit_stacker(pop, {u: data[u] for u in B}, store)
    print(f"{len(users)} sampled users ({len(A)} fit / {len(B)} stack / {len(C)} eval); "
          f"{len(pop.items)} items in the population model  [{time.time() - t0:.0f}s]")

    # ---------------------------------------------------- sampled users --
    evalu = C[:max_eval]
    imps = load_cf_implicit(evalu)
    res: dict = defaultdict(lambda: defaultdict(list))
    for u in evalu:
        train_full, test, cutoff = split_user(data[u])
        items = [m for m, _, _ in test]
        truth = np.array([s for _, s, _ in test])
        for budget in BUDGETS:
            if len(train_full) < budget:
                continue
            tr = [train_full[i] for i in RNG.permutation(len(train_full))[:budget]]
            scored = [Example(m, s, recency(a, cutoff), a) for m, s, a in tr]
            mean = float(np.average([e.score for e in scored], weights=[e.weight for e in scored]))
            imp = implicit_examples(imps.get(u, []), cutoff, mean)
            for name, pred in score_variants(pop, stacker, store, scored, imp, items).items():
                m = metrics(pred, truth)
                if m:
                    res[budget][name].append(m)
    for budget in BUDGETS:
        rs = res[budget]
        if not rs:
            continue
        n = len(rs["personal (current)"])
        print(f"\n== sampled users given {budget} ratings ({n} users) ==")
        print(f"{'model':<22}{'rho':>8}{'nDCG@10':>9}{'RMSE':>8}")
        for name, ms in rs.items():
            rho = np.mean([m["rho"] for m in ms if m["rho"] is not None])
            print(f"{name:<22}{rho:8.3f}{np.mean([m['ndcg'] for m in ms]):9.3f}"
                  f"{np.mean([m['rmse'] for m in ms]):8.3f}")

    # --------------------------------------------------------------- gate --
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    rated = _rated_in_time_order(uid)
    app_imp = query("""SELECT mal_id, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                         FROM list_entry WHERE user_id=%s AND score=0 AND status = ANY(%s)""",
                    (uid, list(CFG.implicit_offsets)))
    gate: dict = defaultdict(list)
    for h in H:
        train, test = rated[:-h], rated[-h:]
        cutoff = test[0]["at"]
        scored = [Example(r["mal_id"], float(r["score"]), recency(r["at"], cutoff), r["at"])
                  for r in train]
        mean = float(np.average([e.score for e in scored], weights=[e.weight for e in scored]))
        imp = implicit_examples([(r["mal_id"], r["status"], r["at"]) for r in app_imp],
                                cutoff, mean)
        items = [r["mal_id"] for r in test]
        truth = np.array([float(r["score"]) for r in test])
        for name, pred in score_variants(pop, stacker, store, scored, imp, items).items():
            gate[name].append(metrics(pred, truth))
    base = [m["rho"] for m in gate["personal (current)"]]
    print("\n== reference profile gate (8 temporal splits) ==")
    print(f"{'model':<22}{'rho':>8}{'nDCG@10':>9}{'RMSE':>8}   wins vs current")
    for name, ms in gate.items():
        rho = [m["rho"] for m in ms]
        wins = "" if name == "personal (current)" else \
            f"   {sum(a > b for a, b in zip(rho, base))}/{len(base)}"
        print(f"{name:<22}{np.mean(rho):8.4f}{np.mean([m['ndcg'] for m in ms]):9.4f}"
              f"{np.mean([m['rmse'] for m in ms]):8.4f}{wins}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.max_eval, a.seed)
