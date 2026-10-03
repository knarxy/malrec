"""Where the personal model should take over, and what that does to the lists.

The blend is  lam * personal + (1 - lam) * population  with lam logistic in the
number of ratings (50/50 at n0). The same lam also sets the ranking dials:
relevance is pulled in by (1 - lam), the Safe Bets popularity prior by
0.5 * (1 - lam). So moving the handover changes the order twice - once
through better predictions, once through the dials, which were tuned against
the old curve. Configurations:

  BASE     n0=80, no cap; long lists (>= chosen_blend_min) choose lam on
           their own newest ratings; dials follow lam (production today)
  H150     n0=150, otherwise as BASE
  H150D    n0=150; the dials follow list size on the old curve (n0=80)
           instead of lam ("decoupled")
  H150CD   as H150D, and lam never above 0.5 (cap), chosen lam included
  H150CDS  as H150CD, but long lists use the size curve too (no chosen lam)

Each user's personal, population and long-memory models are fitted once; a
configuration only changes how they are mixed, exactly as fit_user("sized")
and rank._score_hybrid / _mix_bonus mix them. Genre calibration and
franchise collapse are left out (same for every configuration).

Metrics: rank correlation and RMSE on newest ratings; Safe Bets recall@50 of
liked future titles, plain and IPS-weighted; median popularity rank of the
top 20; how many of BASE's top 20 survive.
"""
from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from malrec.config import settings
from malrec.db import one, query
from malrec.eval import spearman
from malrec.rank import relevance_bonus
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import (
    Example,
    GlobalModel,
    _choose_lambda,
    fit_stacker,
    fit_user,
    recency_weight,
    size_weight,
    split_user,
)
from malrec.recsys.items import ItemStore
from malrec.recsys.service import calibration_from, user_examples, user_listed

CFG = settings()
BUDGETS = (25, 60, 150, 250, 400)
CHOSEN_MIN = 300
MIX = 0.5            # safe_bets_memory_mix on dev
POP_PRIOR = 0.5      # safe_bets_popularity
CONFIGS = {          # name: (n0, cap, decoupled, chosen)
    "BASE": (80.0, 1.0, False, True),
    "H150": (150.0, 1.0, False, True),
    "H150D": (150.0, 1.0, True, True),
    "H150CD": (150.0, 0.5, True, True),
    "H150CDS": (150.0, 0.5, True, False),
}


class _Share:
    def __init__(self, lam):
        self.personal_share = lam


def zs(x):
    sd = float(np.nanstd(x))
    return (x - np.nanmean(x)) / sd if sd > 1e-9 else np.zeros_like(x)


class Parts:
    """One user's fitted pieces: personal and population predictions over the
    pool and the test items, relevance, and the chosen blend weight."""

    def __init__(self, G, store, scored, implicit, listed, ids, n, chosen):
        per = fit_user(G.pop, G.stacker, store, scored, implicit, mode="personal",
                       listed=listed)
        st = fit_user(G.pop, G.stacker, store, scored, implicit, mode="stack", listed=listed)
        self.P = per.predict(ids)
        self.S = st.predict(ids)
        self.P = np.where(np.isnan(self.P), self.S, self.P)
        self.n = n
        self.chosen = (_choose_lambda(G.pop, G.stacker, store, scored, implicit, "personal",
                                      None, 40.0, 40, False) if chosen else None)
        self._st = st

    def relevance(self, pool):
        return self._st.relevance(pool)

    def lam(self, n0, cap, use_chosen):
        if use_chosen and self.chosen is not None:
            return min(self.chosen, cap)
        return min(size_weight(self.n, n0, 15.0), cap)


def safe_key(short, long_, k_pool, z_s, z_l, popz, cfg, cal):
    """rank._score_hybrid + _mix_bonus + the popularity prior, for one config."""
    n0, cap, dec, ch = cfg
    a, b = cal
    lam_s, lam_l = short.lam(n0, cap, ch), long_.lam(n0, cap, ch)
    share_s = size_weight(short.n, 80.0, 15.0) if dec else lam_s
    share_l = size_weight(long_.n, 80.0, 15.0) if dec else lam_l
    raw = lam_s * short.P[k_pool] + (1 - lam_s) * short.S[k_pool]
    raw2 = lam_l * long_.P[k_pool] + (1 - lam_l) * long_.S[k_pool]
    key = np.clip(a * raw + b, 1, 10) + relevance_bonus(_Share(share_s), z_s)
    key2 = np.clip(a * raw2 + b, 1, 10) + relevance_bonus(_Share(share_l), z_l)
    ok = np.isfinite(key) & np.isfinite(key2)
    m1, s1 = key[ok].mean(), key[ok].std() + 1e-9
    m2, s2 = key2[ok].mean(), key2[ok].std() + 1e-9
    mixed = m1 + s1 * ((1 - MIX) * (key - m1) / s1 + MIX * (key2 - m2) / s2)
    bonus = np.where(ok, mixed - key, 0.0)
    w = POP_PRIOR * (1.0 - share_s)
    if w > 1e-3:
        bonus = bonus + w * float(key[ok].std()) * popz
    return key + bonus, lam_s


def score_user(short, long_, ids, items, truth, pool, targets, prop, popul, cal,
               M, base_top):
    # `ids` is pool + the test items not already in it, without duplicates:
    # predict() fills one slot per mal_id, so a repeated id would stay NaN
    at = {m: k for k, m in enumerate(ids)}
    k_pool = np.array([at[m] for m in pool])
    k_items = np.array([at[m] for m in items])
    z_s, z_l = short.relevance(pool), long_.relevance(pool)
    popz = zs(np.array([-math.log10(popul.get(m, 99999)) for m in pool]))
    tops = {}
    for name, cfg in CONFIGS.items():
        key, lam = safe_key(short, long_, k_pool, z_s, z_l, popz, cfg, cal)
        pred = lam * short.P[k_items] + (1 - lam) * short.S[k_items]
        p = np.clip(cal[0] * pred + cal[1], 1, 10)
        m = M[name]
        m["lam"].append(lam)
        if truth.std() > 0:
            m["rho"].append(spearman(p, truth))
            m["rmse"].append(float(np.sqrt(np.mean((p - truth) ** 2))))
        s = np.where(np.isfinite(key), key, -np.inf)
        order = np.argsort(-s)
        top50 = {pool[i] for i in order[:50]}
        top20 = [pool[i] for i in order[:20]]
        tops[name] = set(top20)
        m["pop20"].extend(popul.get(x, 99999) for x in top20)
        if targets:
            hit = top50 & targets
            w = {t: 1.0 / prop.get(t, 1e-3) for t in targets}
            m["r50"].append(len(hit) / len(targets))
            m["ips"].append(sum(w[t] for t in hit) / sum(w.values()))
    for name in CONFIGS:
        M[name]["keep20"].append(len(tops[name] & tops["BASE"]))
    if base_top is not None:
        base_top.append(tops)


def report(title, M):
    print(f"\n== {title} ==", flush=True)
    print(f"{'':<9}{'lam':>6}{'rho':>8}{'RMSE':>7}{'rec@50':>8}{'IPS':>7}{'pop20':>7}"
          f"{'keep20':>8}")
    for name in CONFIGS:
        m = M[name]
        f = lambda k, w=7, d=3, m=m: f"{np.mean(m[k]):{w}.{d}f}" if m[k] else f"{'-':>{w}}"
        print(f"{name:<9}{f('lam', 6, 2)}{f('rho', 8, 4)}{f('rmse')}{f('r50', 8)}{f('ips')}"
              f"{np.median(m['pop20']) if m['pop20'] else 0:7.0f}{f('keep20', 8, 1)}")


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
    G = GlobalModel(pop, stacker, {})
    lists = defaultdict(set)
    for u_, m_ in zip(oa, oi):
        lists[m_].add(u_)
    prop = {m: (len(s) + 1) / (len(A) + 2) for m, s in lists.items()}
    pool_all = pool_ids(store)
    popul = {m: float(r.get("mal_popularity") or 99999) for m, r in store.rows.items()}
    print(f"{len(users)} users ({len(A)} fit / {len(B)} stack / {len(C)} eval) "
          f"[{time.time() - t0:.0f}s]", flush=True)

    def examples(seen, cutoff, hl):
        def rw(at):
            return 1.0 if hl is None else recency_weight(at, cutoff)
        scored = [Example(e["mal_id"], float(e["score"]), rw(e["at"]), e["at"])
                  for e in seen if e["score"] > 0]
        if len(scored) < 3:
            return scored, [], {}
        m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
        implicit = [Example(e["mal_id"], float(min(max(m_ + CFG.implicit_offsets[e["status"]],
                                                        1), 10)),
                            CFG.implicit_weight * rw(e["at"]), e["at"])
                    for e in seen if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
        return scored, implicit, {e["mal_id"]: rw(e["at"]) for e in seen}

    for budget in BUDGETS:
        M = defaultdict(lambda: defaultdict(list))
        for u in C[:max_eval]:
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, rng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            targets = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in rng.permutation(len(history))[:budget]]
            sc, im, li = examples(seen, cutoff, 0.5)
            if len(sc) < 5:
                continue
            scL, imL, liL = examples(seen, cutoff, None)
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            items = [m for m, _, _ in test_s]
            truth = np.array([s for _, s, _ in test_s])
            ids = pool + [m for m in items if m not in set(pool)]
            chosen = len(sc) >= CHOSEN_MIN
            short = Parts(G, store, sc, im, li, ids, len(sc), chosen)
            long_ = Parts(G, store, scL, imL, liL, ids, len(sc), chosen)
            score_user(short, long_, ids, items, truth, pool,
                       targets if len(targets) >= 3 else set(), prop, popul, (1.0, 0.0), M, None)
        report(f"sampled users, {budget} list entries ({len(M['BASE']['lam'])} users) "
               f"[{time.time() - t0:.0f}s]", M)

    # ------------------------------------------------------------ dennis --
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    le = query("""SELECT mal_id, score, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    n_real = len(rated)
    mean = float(np.mean([r["score"] for r in rated]))
    splits = (25, 30, 35, 40, 45, 50, 55, 60)
    fitted = {}
    for h in splits:
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        pool = [m for m in pool_all if m not in {r["mal_id"] for r in le} - test_ids]
        items = [r["mal_id"] for r in test]
        ids = pool + [m for m in items if m not in set(pool)]
        parts = []
        for hl in (0.35, float("inf")):
            sc, im = user_examples(uid, cutoff, hl)
            li = user_listed(uid, cutoff, hl)
            parts.append(Parts(G, store, sc, im, li, ids, n_real, n_real >= CHOSEN_MIN))
        fitted[h] = (parts, pool, items, test, ids)
    # display calibration per config, fitted as production does (holdouts 30/40/50)
    cals = {}
    for name, (n0, cap, _, ch) in CONFIGS.items():
        pr, tr = [], []
        for h in (30, 40, 50):
            (short, _), pool, items, test, ids = fitted[h]
            lam = short.lam(n0, cap, ch)
            at = {m: j for j, m in enumerate(ids)}
            k = np.array([at[m] for m in items])
            pr += list(lam * short.P[k] + (1 - lam) * short.S[k])
            tr += [float(r["score"]) for r in test]
        c = calibration_from(pr, tr)
        cals[name] = (c["slope"], c["intercept"])
    print(f"\nreference profile calibration per config: {cals}")
    M = defaultdict(lambda: defaultdict(list))
    tops = []
    for h in splits:
        (short, long_), pool, items, test, ids = fitted[h]
        targets = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        truth = np.array([float(r["score"]) for r in test])
        # calibration differs per config, so score each config with its own
        for name in CONFIGS:
            sub = defaultdict(lambda: defaultdict(list))
            score_user(short, long_, ids, items, truth, pool, targets, prop, popul,
                       cals[name], sub, None)
            for k, v in sub[name].items():
                if k != "keep20":
                    M[name][k].extend(v)
        # list overlap needs every config's key at once (calibration as served today)
        sub = defaultdict(lambda: defaultdict(list))
        score_user(short, long_, ids, items, truth, pool, targets, prop, popul,
                   cals["BASE"], sub, tops)
    for name in CONFIGS:
        M[name]["keep20"] = [len(t[name] & t["BASE"]) for t in tops]
    report(f"reference profile, 8 temporal splits at its real size ({n_real} ratings)", M)
    for name in CONFIGS:
        print(f"  {name:<9} rho per split: {[round(x, 3) for x in M[name]['rho']]}")
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.max_eval, a.seed)
