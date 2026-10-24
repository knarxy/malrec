"""Audit 2026-10-23: measurements behind the Prio 1 findings, on the benchmark.

Same snapshot, population model, rankers, user split and per-user protocol as
experiments/bench.py; every variant is scored on the same users and compared
paired (+/- beyond two standard errors). Nothing here changes production code:
the variants are monkeypatches inside this process.

  pool    the benchmark (and the rankers' training) leave titles in the
          Safe Bets / Discover / Hidden Gems pool that production never shows
          there: sequels whose main-format prequels are not completed
          (eligible_candidates -> prereqs_satisfied) and the continuations
          Up Next shows (surfaces.continuation_ids). Variants: bench pool vs
          production pool, each with the learned orders as configured and
          with the hand key only, plus Discover / Hidden Gems ranker weights
          0.75 and 1 on the production pool (RESEARCH.md open question B3).
  calib   the rankers are trained, and sampled users benchmarked, with the
          identity calibration (1, 0); production applies each user's own
          line or the population line for their list size
          (service.fit_size_calibration). Variants: identity, the size line as
          production fits it today (fit_user defaults: handover 80, no cap),
          and the size line fitted with the served blend (blend_opts).
  tagcos  the personal ridge's tag_cosine column: training rows are scored
          against a taste vector that contains the row's own rating, the
          candidates never are. Variant: leave-one-out tag_cosine for the
          training rows.

  selfleak
          app users whose own public list is also in the population sample
          (cf_user 'done' under their name hash - fullfetch excludes app users
          only when a list is fetched, not when someone signs up later). The
          population model is refitted in memory without those lists (same
          code and roles as service.fit_global, nothing stored) and each app
          user's temporal holdout, display calibration and served personal
          share are compared between the active model and the clean refit;
          users not in the sample show the refit noise.

  python -m experiments.audit_lab pool     [--users 100] [--out FILE]
  python -m experiments.audit_lab calib    [--users 100] [--cal-users 1000]
  python -m experiments.audit_lab tagcos   [--users 100]
  python -m experiments.audit_lab selfleak [--max-rated 1500]
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from experiments import bench
from malrec.config import settings
from malrec.db import query
from malrec.recsys import hybrid, ltr, service
from malrec.recsys.hybrid import split_user
from malrec.recsys.items import ItemStore
from malrec.surfaces import SPECS

CFG = settings()
WEIGHT_KEYS = ("order_ranker_weight", "discover_ranker_weight", "gems_ranker_weight")
SURFACE_OF = {v: k for k, v in ltr.SURFACES.items()}


# ------------------------------------------------------------------ setup --

@dataclass
class Ctx:
    entries: object
    store: ItemStore
    pop: object
    stacker: object
    A: list
    B: list
    C: list
    pool_all: list
    gems_all: list
    rankers: dict
    chain: dict = field(default_factory=dict)       # mal_id -> main-format prequel chain
    sequels: dict = field(default_factory=dict)     # prequel -> direct sequels
    tables: dict = field(default_factory=dict)      # size-calibration tables by name


def setup(only: str | None = None) -> Ctx:
    """The benchmark's data, population model and rankers; `only` trains
    (or loads) just that tab's ranker and returns - to fill the cache in
    parallel processes."""
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store, derived = static["store"], static["derived"]
    bench.GROUPS = bench.groupings(store)
    bench.SNAP_DAY = dt.datetime.fromisoformat(static["created"]).date()
    bench.popularity_columns(store, bench.SNAP_DAY)
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = bench.population(entries, store, A, B)
    pool_all = bench.eligible(store, derived)
    gems_all = bench.eligible(store, derived, SPECS["hidden_gems"].min_scorers)
    _, ma, _ = entries.columns(A)
    bench.PROP = (np.bincount(ma.astype(np.int64)) + 1.0) / (len(A) + 2.0)
    bench.RANKERS.clear()
    pools = {"safe_bets": (pool_all, None), "discover": (pool_all, pool_all),
             "hidden_gems": ([m for m in gems_all if bench.gem(store.rows[m])], None)}
    for tab, setting in ltr.SURFACES.items():
        if getattr(CFG, setting) > 0 and only in (None, tab):
            bench.RANKERS[tab], rhow = bench.order_ranker(entries, store, pop, stacker, B,
                                                          pools[tab][0], tab, pools[tab][1])
            how += f", {tab} order {rhow}"
    print(f"setup: {len(users)} users ({len(A)}/{len(B)}/{len(C)}), pool {len(pool_all)},"
          f" {how} [{time.time() - t0:.0f}s]", flush=True)
    print("settings: " + ", ".join(f"{k}={getattr(CFG, k)}" for k in (
        *WEIGHT_KEYS, "safe_bets_memory_mix", "age_calibration", "calibration_block",
        "age_fresh_weight", "blend_lam_min", "model_mode", "gems_popularity")), flush=True)
    return Ctx(entries, store, pop, stacker, A, B, C, pool_all, gems_all, dict(bench.RANKERS))


def load_relations(ctx: Ctx) -> None:
    """Prequel chains (as SQL prerequisites(): relation_type 'prequel',
    depth <= 12, main formats only - prereqs_satisfied of 012) and direct
    sequels (NEXT_UP_SQL), from the live relation table."""
    pre, seq = defaultdict(set), defaultdict(set)
    for r in query("SELECT src, dst, relation_type FROM relation"
                   " WHERE relation_type IN ('prequel', 'sequel')"):
        (pre if r["relation_type"] == "prequel" else seq)[r["src"]].add(r["dst"])
    main = {m for m, r in ctx.store.rows.items() if r.get("format_class") == "main"}
    for m in set(ctx.pool_all) | set(ctx.gems_all):
        seen, frontier, depth = set(), {m}, 0
        while frontier and depth < 12:
            nxt = set()
            for x in frontier:
                nxt |= pre.get(x, set())
            nxt -= seen
            seen |= nxt
            frontier, depth = nxt, depth + 1
        seen.discard(m)
        ctx.chain[m] = frozenset(seen & main)
    ctx.sequels = {p: frozenset(s & main) for p, s in seq.items()}


def prod_exclusions(ctx: Ctx, full_hist: list[dict]) -> tuple[set[int], set[int]]:
    """(unmet prerequisites, Up Next continuations) at the cutoff, from the
    whole list before it - what production keeps out of the discovery tabs."""
    completed = {e["mal_id"] for e in full_hist if e["status"] == "completed"}
    scored = {e["mal_id"]: e["score"] for e in full_hist if e["score"] > 0}
    mu = float(np.mean(list(scored.values()))) if scored else 7.0
    thr = max(round(mu) - 1, 6)
    unmet = {m for m, ch in ctx.chain.items() if ch and not ch <= completed}
    cont = {m for p in completed if scored.get(p, 0) >= thr
            for m in ctx.sequels.get(p, ()) if m not in unmet}
    return unmet, cont


# --------------------------------------------------------------- variants --

@dataclass
class Variant:
    name: str
    pool: str = "bench"               # bench | prod
    weights: dict = field(default_factory=dict)   # overrides of WEIGHT_KEYS
    cal: str = "identity"             # identity | own (ref) | skew | fixed
    loo: bool = False


@contextlib.contextmanager
def weights(over: dict):
    saved = {k: getattr(CFG, k) for k in WEIGHT_KEYS}
    saved_r = dict(bench.RANKERS)
    try:
        for k, v in over.items():
            setattr(CFG, k, v)
        for k in WEIGHT_KEYS:
            if getattr(CFG, k) <= 0:
                bench.RANKERS.pop(SURFACE_OF[k], None)
        yield
    finally:
        for k, v in saved.items():
            setattr(CFG, k, v)
        bench.RANKERS.clear()
        bench.RANKERS.update(saved_r)


# tag_cosine leave-one-out: the taste vector a fit used, by id of the vector
_TASTE: dict[int, tuple] = {}
_LOO = {"on": False}
_orig_tv = ItemStore.taste_vector
_orig_fit_ridge = hybrid._fit_ridge_part


def _tv(self, ratings, weights_):
    v = _orig_tv(self, ratings, weights_)
    if v is not None:
        _TASTE[id(v)] = (dict(ratings), dict(weights_), v)
    return v


def _fit_ridge_loo(model, rows, ex, alpha):
    t = _TASTE.get(id(model.taste)) if model.taste is not None else None
    if _LOO["on"] and t is not None:
        ratings, wts, _ = t
        st = model.store
        ids = [m for m in ratings if m in st.vec_index]
        if ids:
            w_all = np.array([wts[m] for m in ratings])
            s_all = np.array([ratings[m] for m in ratings])
            mu = float((w_all * s_all).sum() / max(w_all.sum(), 1e-9))
            T = st.vecs[[st.vec_index[m] for m in ids]].astype(np.float64)
            coef = np.array([(ratings[m] - mu) * wts[m] for m in ids])
            v = coef @ T
            pos = {m: k for k, m in enumerate(ids)}
            for r in rows:
                k = pos.get(r["mal_id"])
                if k is None:
                    continue
                vi = v - coef[k] * T[k]
                n = float(np.linalg.norm(vi))
                r["tag_cosine"] = float(T[k] @ vi / n) if n > 0 else 0.0
    return _orig_fit_ridge(model, rows, ex, alpha)


ItemStore.taste_vector = _tv
hybrid._fit_ridge_part = _fit_ridge_loo


def size_tables(ctx: Ctx, n_users: int) -> None:
    """The population size-calibration table as production fits it today and
    with the served blend, on the same stacker users."""
    users = {}
    for u in ctx.B[:n_users]:
        users[u] = [(e["mal_id"], float(e["score"]), e["at"])
                    for e in ctx.entries[u] if e["score"] > 0]
    t0 = time.time()
    ctx.tables["skew"] = service.fit_size_calibration(ctx.pop, ctx.stacker, ctx.store, users)
    print(f"size table as fitted today [{time.time() - t0:.0f}s]", flush=True)
    real = service.fit_user

    def served(pop, stacker, store, scored, implicit=None, mode="sized", **kw):
        return hybrid.fit_user(pop, stacker, store, scored, implicit,
                               mode=service.blend_mode(len(scored)), blend_of="personal",
                               **{**service.blend_opts(), **kw})
    service.fit_user = served
    try:
        t0 = time.time()
        ctx.tables["fixed"] = service.fit_size_calibration(ctx.pop, ctx.stacker, ctx.store, users)
        print(f"size table with the served blend [{time.time() - t0:.0f}s]", flush=True)
    finally:
        service.fit_user = real
    for name in ("skew", "fixed"):
        print(f"  {name}: " + "; ".join(
            f"n={r['n']} a={r['slope']:.3f} b={r['intercept']:.3f} rmse {r.get('rmse_raw')}"
            f"->{r.get('rmse_calibrated')} top a={r.get('top_slope', float('nan')):.3f}"
            f" b={r.get('top_intercept', float('nan')):.3f}" for r in ctx.tables[name]),
            flush=True)


def calibration_for(ctx: Ctx, v: Variant, n: int, ref_cal) -> tuple[float, float]:
    if v.cal == "identity":
        return ref_cal if ref_cal is not None else (1.0, 0.0)
    pc = service.size_calibration(ctx.tables[v.cal], n)
    return pc if pc is not None else (1.0, 0.0)


# ---------------------------------------------------------------- per user --

def genres(store, ids):
    return {g for m in ids for g in ((store.rows.get(m) or {}).get("mal_genres") or [])}


def evaluate(ctx: Ctx, variants: list[Variant], case: dict, ref_cal=None) -> dict | None:
    """Every variant's Safe Bets / Discover / Hidden Gems for one user."""
    hist, cutoff = case["hist"], case["cutoff"]
    sc, im, li = bench.examples(hist, cutoff)
    if len(sc) < 5:
        return None
    store = ctx.store
    fam = genres(store, (e["mal_id"] for e in hist))
    unmet, cont = prod_exclusions(ctx, case["full_hist"]) if ctx.chain else (set(), set())
    excl = unmet | cont
    liked, disliked = case["liked"], case["disliked"]
    out: dict = {"_n": len(sc)}
    if liked:
        out["_liked_unmet"] = len(liked & unmet) / len(liked)
        out["_liked_cont"] = len(liked & cont) / len(liked)
    ums: dict[bool, object] = {}
    for v in variants:
        _LOO["on"] = v.loo
        _TASTE.clear()
        if v.loo not in ums:
            ums[v.loo] = hybrid.fit_user(ctx.pop, ctx.stacker, store, sc, im,
                                        mode=service.blend_mode(len(sc)), blend_of="personal",
                                        **service.blend_opts(), listed=li)
        um = ums[v.loo]
        cal = calibration_for(ctx, v, len(sc), ref_cal)
        scorer = bench.make_scorer(ctx.pop, ctx.stacker, store, sc, im, li, cal, um=um)
        long_model = None
        if CFG.safe_bets_memory_mix > 0:
            sc2, im2, li2 = ltr.history_examples(hist, cutoff, float("inf"))
            long_model = bench.make_scorer(ctx.pop, ctx.stacker, store, sc2, im2, li2, cal)
        drop = excl if v.pool == "prod" else set()
        rows = [bench.row_of(store, m) for m in ctx.pool_all
                if m not in case["listed"] and m not in drop]
        gems_rows = [bench.row_of(store, m) for m in ctx.gems_all
                     if m not in case["listed"] and m not in drop]
        with weights(v.weights):
            _, sb = bench.ranked(scorer, rows, li, fam, store, history=hist,
                                 long_model=long_model, cutoff=cutoff)
            top = [c.mal_id for c in sb]
            _, disc = bench.ranked(scorer, rows, li, fam, store, "discover", taken=set(top),
                                   history=hist, cutoff=cutoff)
            _, gems = bench.ranked(scorer, gems_rows, li, fam, store, "hidden_gems",
                                   history=hist, cutoff=cutoff,
                                   taken=set(top[:CFG.gems_skip_safe_bets]))
        m = bench.user_metrics(scorer, top, case["items"], case["truth"], liked, disliked)
        dids, gids = [c.mal_id for c in disc], [c.mal_id for c in gems]
        m |= bench.list_metrics(dids, liked, disliked, "disc_")
        m |= bench.list_metrics(gids, liked, disliked, "gems_")
        ages = bench.rank.age_mix(case["full_hist"], bench.year_of(store), cutoff.year)
        m |= bench.age_metrics(top, ages)
        # how much of each top 10 production would never show in that tab
        for p, ids in (("", top), ("disc_", dids), ("gems_", gids)):
            if len(ids) >= 10:
                m[f"{p}excl10"] = len(set(ids[:10]) & excl) / 10
                m[f"{p}hitx10"] = len(set(ids[:10]) & excl & liked) / 10
        out[v.name] = m
    _LOO["on"] = False
    return out


def cases(ctx: Ctx, budget: int):
    """The benchmark's sampled users at one budget, drawn exactly as bench.run."""
    urng = np.random.default_rng(bench.SEED + budget)
    for u in ctx.C:
        es = ctx.entries[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        _, test_s, cutoff = split_user(scored_all, urng)
        mean = float(np.mean([s for _, s, _ in scored_all]))
        test_ids = {m for m, _, _ in test_s}
        history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
        if len(history) < budget:
            continue
        seen = [history[i] for i in urng.permutation(len(history))[:budget]]
        disliked = {m for m, s, _ in test_s if s < mean - 1}
        disliked |= {e["mal_id"] for e in es if e["status"] == "dropped" and e["at"] >= cutoff}
        yield str(u), {"hist": seen, "cutoff": cutoff,
                       "listed": {e["mal_id"] for e in es} - test_ids,
                       "items": [m for m, _, _ in test_s],
                       "truth": np.array([s for _, s, _ in test_s]),
                       "liked": {m for m, s, _ in test_s if s >= max(mean, 7.0)},
                       "disliked": disliked, "full_hist": history}


def ref_cases():
    _, static = bench.load_snapshot()
    ref = static["ref"]
    le = ref["le"]
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    for h in bench.REF_SPLITS:
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        yield f"split {h}", {"hist": hist, "cutoff": cutoff,
                             "listed": {r["mal_id"] for r in le} - test_ids,
                             "items": [r["mal_id"] for r in test],
                             "truth": np.array([float(r["score"]) for r in test]),
                             "liked": {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)},
                             "disliked": {r["mal_id"] for r in test if r["score"] < mean - 1},
                             "full_hist": hist}, ref["cal"]


def run(ctx: Ctx, variants: list[Variant], max_eval: int, budgets=bench.BUDGETS) -> dict:
    t0 = time.time()
    res: dict = {}
    for b in budgets:
        per = {}
        for u, case in cases(ctx, b):
            if len(per) >= max_eval:
                break
            m = evaluate(ctx, variants, case)
            if m is not None:
                per[u] = m
        res[f"sampled {b}"] = per
        print(f"  sampled {b}: {len(per)} users [{time.time() - t0:.0f}s]", flush=True)
    per = {}
    for k, case, cal in ref_cases():
        m = evaluate(ctx, variants, case, ref_cal=cal)
        if m is not None:
            per[k] = m
    res["reference"] = per
    print(f"  reference: {len(per)} splits [{time.time() - t0:.0f}s]", flush=True)
    return res


# ----------------------------------------------------------------- report --

METRICS = (("rho", True), ("rmse", False), ("r50", True), ("ips50", True), ("hit10", True),
           ("bad10", False), ("agegap", False), ("disc_r50", True), ("disc_ips50", True),
           ("disc_hit10", True), ("disc_bad10", False), ("gems_r50", True), ("gems_ips50", True),
           ("gems_hit10", True), ("gems_bad10", False), ("excl10", None), ("hitx10", None),
           ("disc_excl10", None), ("disc_hitx10", None), ("gems_excl10", None),
           ("gems_hitx10", None))


def report(res: dict, pairs: list[tuple[str, str]]) -> None:
    for g, per in res.items():
        print(f"\n== {g} ({len(per)})")
        for k in ("_liked_unmet", "_liked_cont"):
            vals = [v[k] for v in per.values() if k in v]
            if vals:
                print(f"   {k:<14} mean {np.mean(vals):.4f}")
        for base, arm in pairs:
            print(f"   -- {arm}  vs  {base}")
            for m, higher in METRICS:
                pv = [(v[arm][m], v[base][m]) for v in per.values()
                      if arm in v and base in v and m in v[arm] and m in v[base]]
                if len(pv) < 2:
                    continue
                d = np.array([a - b for a, b in pv])
                se = float(d.std(ddof=1) / np.sqrt(len(d)))
                flag = ""
                if higher is not None and se > 0 and abs(d.mean()) > 2 * se:
                    flag = "+" if (d.mean() > 0) == higher else "-"
                if np.all(np.abs(d) < 1e-12):
                    flag = "="
                base_mean = float(np.mean([b for _, b in pv]))
                if g == "reference":
                    win = int((d > 1e-9).sum())
                    lose = int((d < -1e-9).sum())
                    print(f"      {m:<12}{base_mean:9.4f}{d.mean():+9.4f}  {win}+/{lose}-")
                else:
                    print(f"      {m:<12}{base_mean:9.4f}{d.mean():+9.4f} ±{2 * se:.4f} {flag}")


SELF_SQL = """
    SELECT u.id AS uid, c.id AS cid, c.state
      FROM app_user u
      JOIN cf_user c ON c.name_hash = encode(sha256(convert_to(lower(trim(u.mal_username)),
                                                               'UTF8')), 'hex')
     WHERE u.status = 'approved'"""


def selfleak(max_rated: int, out: str) -> None:
    from malrec.recsys.cf import CFParams, PopulationModel
    from malrec.recsys.hybrid import GlobalModel, fit_stacker, fit_user, unit_hash
    t0 = time.time()
    hits = query(SELF_SQL)
    leaked = {r["uid"]: r["cid"] for r in hits if r["state"] == "done"}
    print(f"app users with their list in the sample: {sorted(leaked)}", flush=True)
    active = service.active_global()
    cols = service.load_cf_columns()
    users = sorted(set(cols.scored_users().tolist()) - set(leaked.values()))
    stack_ids = [u for u in users if unit_hash("role", 0, u) < 0.15]
    pop_ids = [u for u in users if unit_hash("role", 0, u) >= 0.15]
    print(f"sample {len(cols.scored_users())} -> {len(users)} without them"
          f" (active model: {active.meta.get('users')}) [{time.time() - t0:.0f}s]", flush=True)
    in_pop = np.isin(cols.user, np.array(pop_ids))
    rated = in_pop & (cols.score > 0)
    pop = PopulationModel(CFParams()).fit(cols.user[rated].astype(np.int64),
                                          cols.mal_id[rated].astype(np.int64),
                                          cols.score[rated].astype(np.float64))
    ou, oi = cols.user[in_pop].astype(np.int64), cols.mal_id[in_pop].astype(np.int64)
    pop.fit_occurrence(ou, oi)
    if CFG.relevance_ease:
        pop.fit_ease(ou, oi)
    store = service.item_store()
    stacker = fit_stacker(pop, cols.ratings(stack_ids), store, seed=0)
    del cols
    clean = GlobalModel(pop, stacker, {})
    print(f"clean refit [{time.time() - t0:.0f}s]", flush=True)

    def served_lam(uid, gm, hl):
        scored, implicit = service.user_examples(uid, None, hl)
        um = fit_user(gm.pop, gm.stacker, store, scored, implicit,
                      mode=service.blend_mode(len(scored)), blend_of="personal",
                      **service.blend_opts(), listed=service.user_listed(uid, None, hl))
        return round(float(um.lam), 3), um.mode

    res = {}
    for r in query("SELECT id FROM app_user WHERE status='approved' ORDER BY id"):
        uid = r["id"]
        n = query("SELECT count(*) AS n FROM list_entry WHERE user_id=%s AND score>0", (uid,))[0]["n"]
        if n > max_rated:
            continue
        hl = service._half_life(uid)
        row = {"in_sample": uid in leaked, "n": n}
        for name, gm in (("active", active), ("clean", clean)):
            h = service.temporal_holdout(uid, gm=gm, half_life=hl)
            cal = service.calibration_from(h.pop("_preds", []), h.pop("_truth", []))
            lam, mode = served_lam(uid, gm, hl)
            row[name] = {"rho": h.get("spearman"), "rmse": h.get("rmse"), "slope": cal["slope"],
                         "intercept": cal["intercept"], "lam": lam, "mode": mode}
        res[str(uid)] = row
        a_, c_ = row["active"], row["clean"]
        print(f"  user {uid:>5} in_sample={row['in_sample']!s:<5} n={n:>4}  rho {a_['rho']} -> "
              f"{c_['rho']}  rmse {a_['rmse']} -> {c_['rmse']}  slope {a_['slope']} -> "
              f"{c_['slope']}  lam {a_['lam']} -> {c_['lam']} ({a_['mode']})"
              f" [{time.time() - t0:.0f}s]", flush=True)
    for flag in (True, False):
        rows = [v for v in res.values() if v["in_sample"] == flag and v["active"]["rho"] is not None
                and v["clean"]["rho"] is not None]
        if rows:
            d = np.array([v["active"]["rho"] - v["clean"]["rho"] for v in rows])
            print(f"in_sample={flag}: {len(rows)} users, mean rho active - clean {d.mean():+.4f}"
                  f" (sd {d.std():.4f})", flush=True)
    if out:
        with open(out, "w") as f:
            json.dump(res, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("pool", "calib", "tagcos", "selfleak", "ranker"))
    ap.add_argument("--tab", default="safe_bets")
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--cal-users", type=int, default=1000)
    ap.add_argument("--max-rated", type=int, default=1500)
    ap.add_argument("--budgets", default=",".join(map(str, bench.BUDGETS)))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    budgets = tuple(int(x) for x in a.budgets.split(","))
    t0 = time.time()
    if a.what == "selfleak":
        selfleak(a.max_rated, a.out)
        print(f"\n[{time.time() - t0:.0f}s]")
        return 0
    if a.what == "ranker":
        setup(only=a.tab)
        print(f"\n[{time.time() - t0:.0f}s]")
        return 0
    ctx = setup()
    load_relations(ctx)
    if a.what == "pool":
        variants = [Variant("bench/conf"), Variant("bench/hand", weights=dict.fromkeys(WEIGHT_KEYS, 0.0)),
                    Variant("prod/conf", pool="prod"),
                    Variant("prod/hand", pool="prod", weights=dict.fromkeys(WEIGHT_KEYS, 0.0)),
                    Variant("prod/w075", pool="prod", weights={"discover_ranker_weight": 0.75,
                                                               "gems_ranker_weight": 0.75}),
                    Variant("prod/w1", pool="prod", weights={"discover_ranker_weight": 1.0,
                                                             "gems_ranker_weight": 1.0})]
        pairs = [("bench/hand", "bench/conf"), ("prod/hand", "prod/conf"),
                 ("bench/conf", "prod/conf"), ("prod/conf", "prod/w075"), ("prod/conf", "prod/w1")]
    elif a.what == "calib":
        size_tables(ctx, a.cal_users)
        hand = dict.fromkeys(WEIGHT_KEYS, 0.0)
        variants = [Variant("id/conf"), Variant("fixed/conf", cal="fixed"),
                    Variant("skew/conf", cal="skew"), Variant("id/hand", weights=hand),
                    Variant("fixed/hand", cal="fixed", weights=hand),
                    Variant("prodpool/fixed/conf", pool="prod", cal="fixed"),
                    Variant("prodpool/fixed/hand", pool="prod", cal="fixed", weights=hand)]
        pairs = [("id/conf", "fixed/conf"), ("skew/conf", "fixed/conf"),
                 ("id/hand", "id/conf"), ("fixed/hand", "fixed/conf"),
                 ("prodpool/fixed/hand", "prodpool/fixed/conf")]
    else:
        variants = [Variant("base"), Variant("loo", loo=True)]
        pairs = [("base", "loo")]
    res = run(ctx, variants, a.users, budgets)
    report(res, pairs)
    if a.out:
        def clean(x):
            if isinstance(x, dict):
                return {k: clean(v) for k, v in x.items()}
            return float(x) if isinstance(x, (np.floating, np.integer)) else x
        with open(a.out, "w") as f:
            json.dump({"what": a.what, "variants": [v.name for v in variants], "res": clean(res),
                       "tables": ctx.tables}, f)
    print(f"\n[{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
