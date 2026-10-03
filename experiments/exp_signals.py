"""Do the signals from the full fetch improve anything?

Candidates (all off in production until they pass here):

  stack:  staff      how the user rated other work by the same director /
                     series writer / original creator (bias-free, leave-one-out)
          drop       log drop rate            (MAL, AniList fallback)
          polar      score-distribution spread (how divided opinion is)
          tag_cos    tag-profile similarity - the one content signal the
                     population stacker has never had
  personal ridge:    staff tokens (dir:/wri:/orig:/mus:), audience numerics,
                     staff affinity
  relevance:         dropped titles as *negative* list evidence - the offline
                     stand-in for the app's "Not for me"

Stack variants are scored where the population carries the prediction
(10-150 list entries); personal-model variants where the personal model does
(150+, and reference profile at its real size). Every variant goes through the
production fit_user / relevance_bonus / novelty_scale, as in exp_final.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np

from experiments.exp_final import _Share, z_over
from experiments.exp_retrieval import load_entries, pool_ids, recall
from malrec.config import settings
from malrec.db import one, query
from malrec.eval import ndcg_at, spearman
from malrec.rank import novelty, relevance_bonus
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import Example, fit_stacker, fit_user, recency_weight, split_user
from malrec.recsys.items import ItemStore
from malrec.recsys.scorer import novelty_scale

CFG = settings()
FLAGS = ("content_staff", "content_audience", "content_staff_aff")

# name -> (stacker extras, content flags switched on, dropped-as-negative weight)
STACK_VARIANTS = {
    "base": ((), (), 0.0),
    "stack+staff": (("staff",), (), 0.0),
    "stack+drop+polar": (("drop", "polar"), (), 0.0),
    "stack+tag_cos": (("tag_cos",), (), 0.0),
    "stack+all": (("staff", "drop", "polar", "tag_cos"), (), 0.0),
    "dropped negative": ((), (), -0.5),
}
PERSONAL_VARIANTS = {
    "base": ((), (), 0.0),
    "pers+staff tokens": ((), ("content_staff",), 0.0),
    "pers+audience": ((), ("content_audience",), 0.0),
    "pers+staff_aff": ((), ("content_staff_aff",), 0.0),
    "pers+all": ((), FLAGS, 0.0),
}


LEVEL_VARIANTS = {
    "base": ((), (), 0.0),
    "level lin3": ((), ("level:lin3",), 0.0),
    "level log": ((), ("level:log",), 0.0),
    "level koren": ((), ("level:koren",), 0.0),
}


def set_flags(on: tuple[str, ...]) -> None:
    for f in FLAGS:
        setattr(CFG, f, f in on)
    CFG.level_trend = next((f.split(":", 1)[1] for f in on if f.startswith("level:")), "")
    hl = next((f.split(":", 1)[1] for f in on if f.startswith("hl:")), None)
    CFG.recency_half_life_years = 0.5 if hl is None or hl == "none" else float(hl)
    CFG.recency_floor = 1.0 if hl == "none" else 0.05
    EASE["mode"] = next((f.split(":", 1)[1] for f in on if f.startswith("ease:")), "")
    if PROD["pop"] is not None:          # production sparse EASE on/off
        PROD["pop"].ease = PROD["ease"] if "easeprod" in on else None


def mix_weight(flags) -> float:
    return next((float(f.split(":", 1)[1]) for f in flags if f.startswith("mix:")), 0.0)


def keyed(pop, stacker, store, scored, implicit, listed_fn, pool, test_items, flags, cutoff,
          *cal, size_n=None):
    """run_one under `flags`; with a mix weight, the discovery key blends in
    the z-scored key of a no-decay model (predictions stay the tuned model's)."""
    set_flags(flags)
    key, pred = run_one(pop, stacker, store, reweight(scored, cutoff),
                        reweight(implicit, cutoff, True), listed_fn(), pool, test_items,
                        *cal, size_n=size_n)
    w = mix_weight(flags)
    if w:
        set_flags(("hl:none",) + tuple(f for f in flags if f.startswith("ease:")))
        k2, _ = run_one(pop, stacker, store, reweight(scored, cutoff),
                        reweight(implicit, cutoff, True), listed_fn(), pool, test_items,
                        *cal, size_n=size_n)
        set_flags(flags)
        z = lambda x: (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
        key = (1 - w) * z(key) + w * z(k2)
    return key, pred


# ranking mix: display/rating from the tuned model, discovery order from the
# mean of its key and a no-decay model's key (long memory finds more titles)
MIX_VARIANTS = {"base": ((), (), 0.0)}
for _hl in ("0.25", "0.35", "0.75"):
    MIX_VARIANTS[f"hl={_hl}"] = ((), (f"hl:{_hl}",), 0.0)
for _w in ("0.3", "0.5", "0.7"):
    MIX_VARIANTS[f"mix nodecay w={_w}"] = ((), (f"mix:{_w}",), 0.0)


COMBO_VARIANTS = {"base": ((), (), 0.0)}
for _hl in ("0.3", "0.35", "0.4"):
    for _w in ("0", "0.5", "0.7"):
        COMBO_VARIANTS[f"hl={_hl} mix={_w}"] = (
            (), (f"hl:{_hl}",) + ((f"mix:{_w}",) if _w != "0" else ()), 0.0)


def top20_pop(key, pool, store) -> float:
    s = np.where(np.isnan(key), -np.inf, key)
    return float(np.median([store.rows[pool[i]].get("mal_popularity") or 99999
                            for i in np.argsort(-s)[:20]]))


PROD: dict = {"pop": None, "ease": None}
PROD_VARIANTS = {"base": ((), (), 0.0), "ease prod (top-100)": ((), ("easeprod",), 0.0)}

FINAL_VARIANTS = {
    "base": ((), (), 0.0),
    "ease": ((), ("ease:mix",), 0.0),
    "hl.35": ((), ("hl:0.35",), 0.0),
    "hl.35+mix.5": ((), ("hl:0.35", "mix:0.5"), 0.0),
    "hl.35+mix.5+ease": ((), ("hl:0.35", "mix:0.5", "ease:mix"), 0.0),
    "hl.35+mix.7+ease": ((), ("hl:0.35", "mix:0.7", "ease:mix"), 0.0),
}

EASE_VARIANTS = {"base": ((), (), 0.0),
                 "ease replace": ((), ("ease:replace",), 0.0),
                 "ease mix": ((), ("ease:mix",), 0.0)}


# drift form x recency half-life: Koren found instance decay unnecessary once
# drift is modelled explicitly
GRID_VARIANTS = {"base": ((), (), 0.0)}
for _lv in ("", "koren", "log"):
    for _hl in ("0.5", "1.0", "2.0", "none"):
        if _lv or _hl != "0.5":
            GRID_VARIANTS[f"{_lv or 'nolevel'} hl={_hl}"] = (
                (), tuple(x for x in (f"level:{_lv}" if _lv else "", f"hl:{_hl}") if x), 0.0)


# --------------------------------------------------------------- EASE --
# Steck (WWW 2019): B = I - P / diag(P), P = (X'X + lambda I)^-1 on binary
# user x item list membership; a user's score for j is sum_i x_i B_ij.
EASE: dict = {"B": None, "index": {}, "mode": ""}


def fit_ease(user_ids, item_ids, min_users: int = 50, lam: float = 400.0) -> None:
    import scipy.sparse as sp
    pairs = np.unique(np.column_stack([user_ids, item_ids]), axis=0)
    items, counts = np.unique(pairs[:, 1], return_counts=True)
    keep = items[counts >= min_users]
    m = np.isin(pairs[:, 1], keep)
    u = np.unique(pairs[m, 0], return_inverse=True)[1]
    i = np.searchsorted(keep, pairs[m, 1])
    X = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(u.max() + 1, len(keep)))
    G = (X.T @ X).toarray().astype(np.float64)
    G[np.diag_indices_from(G)] += lam
    P = np.linalg.inv(G)
    B = -P / np.diag(P)
    B[np.diag_indices_from(B)] = 0.0
    EASE.update(B=B.astype(np.float32), index={int(k): n for n, k in enumerate(keep)})
    print(f"EASE: {len(keep)} items (>= {min_users} lists), lambda {lam}", flush=True)


def ease_scores(listed: dict[int, float], pool: list[int]) -> np.ndarray:
    idx = EASE["index"]
    x = np.zeros(len(idx), dtype=np.float32)
    for m, w in listed.items():
        k = idx.get(m)
        if k is not None:
            x[k] = max(w, 0.0)
    s = x @ EASE["B"]
    return np.array([s[idx[m]] if m in idx else np.nan for m in pool], dtype=float)


def run_one(pop, stacker, store, scored, implicit, listed, pool, test_items,
            slope=1.0, icpt=0.0, size_n=None):
    sz = fit_user(pop, stacker, store, scored, implicit, mode="sized", listed=listed,
                  size_n=size_n)
    shown = np.clip(slope * sz.predict(pool) + icpt, 1, 10)
    nov = np.array([novelty(store.rows[m].get("mal_popularity")) for m in pool])
    relz = sz.relevance(pool)            # production: co-occurrence (+ EASE if fitted)
    if EASE["mode"]:
        ez = ease_scores(listed, pool)
        ez = z_over(np.where(np.isnan(ez), np.nanmin(ez), ez))
        relz = ez if EASE["mode"] == "replace" else 0.5 * (relz + ez)
    base = shown + relevance_bonus(_Share(sz.lam), relz)
    top = np.argsort(-np.nan_to_num(base, nan=-1e9))[:600]
    key = base + CFG.novelty_weight * novelty_scale(shown[top], CFG.novelty_ref_sd) * nov
    return key, np.clip(slope * sz.predict(test_items) + icpt, 1, 10)


def reweight(examples, cutoff, implicit=False):
    """Recency weights under the current config (the grid varies it)."""
    k = CFG.implicit_weight if implicit else 1.0
    return [Example(e.mal_id, e.score, k * recency_weight(e.at, cutoff), e.at) for e in examples]


def listed_weights(entries, cutoff, neg: float) -> dict[int, float]:
    return {e["mal_id"]: recency_weight(e["at"], cutoff) * (neg if neg and e["status"] == "dropped"
                                                          else 1.0)
            for e in entries}


def report(title, M, names):
    print(f"\n== {title} ==")
    print(f"{'':<20}{'rho':>7}{'nDCG@10':>9}{'RMSE':>7}{'recall@50':>11}{'pop20':>7}"
          f"   rho wins vs base")
    base = M["base"]["rho"]
    for n in names:
        m = M[n]
        if not m["rho"]:
            continue
        w = "" if n == "base" else \
            f"   {sum(a > b + 1e-9 for a, b in zip(m['rho'], base))}/{len(base)} better, " \
            f"{sum(a < b - 1e-9 for a, b in zip(m['rho'], base))} worse"
        print(f"{n:<20}{np.mean(m['rho']):7.4f}{np.mean(m['ndcg']):9.4f}{np.mean(m['rmse']):7.3f}"
              f"{np.mean(m['r50']) if m['r50'] else float('nan'):11.3f}"
              f"{np.median(m['pop']) if m['pop'] else float('nan'):7.0f}{w}")


def main(max_eval: int, seed: int, part: str = "all"):
    t0 = time.time()
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
    if part in ("ease", "final"):
        fit_ease(np.array(oa), np.array(oi))
    if part == "prod":
        pop.fit_ease(np.array(oa), np.array(oi))
        PROD.update(pop=pop, ease=pop.ease)
        pop.ease = None
    if part in ("ease", "combo", "final", "prod"):
        CFG.relevance_weight_personal = 0.5          # as deployed
    store = ItemStore.load()
    stack_users = {u: [(e["mal_id"], float(e["score"]), e["at"])
                       for e in entries[u] if e["score"] > 0] for u in B}
    stackers = {}
    for extras, _, _ in (PERSONAL_VARIANTS if part in ("personal", "level", "grid", "mix", "ease", "combo", "final", "prod")
                         else {**STACK_VARIANTS, **PERSONAL_VARIANTS}).values():
        if extras not in stackers:
            stackers[extras] = fit_stacker(pop, stack_users, store, seed=seed, extras=extras)
            print(f"stacker {extras or '(base)'}: "
                  f"{np.round(stackers[extras].coef_, 3).tolist()}")
    pool_all = pool_ids(store)
    print(f"{len(users)} users ({len(A)} fit / {len(B)} stack / {len(C)} eval) "
          f"[{time.time() - t0:.0f}s]")

    # ------------------------------------------------------ sampled users --
    plan = ((10, STACK_VARIANTS), (25, STACK_VARIANTS), (60, STACK_VARIANTS),
            (150, {**STACK_VARIANTS, **PERSONAL_VARIANTS}), (400, PERSONAL_VARIANTS))
    if part == "personal":
        plan = ((150, PERSONAL_VARIANTS), (400, PERSONAL_VARIANTS))
    if part == "level":
        plan = ((60, LEVEL_VARIANTS), (150, LEVEL_VARIANTS), (400, LEVEL_VARIANTS))
    if part == "grid":
        plan = ((150, GRID_VARIANTS), (400, GRID_VARIANTS))
    if part == "prod":
        plan = tuple((b, PROD_VARIANTS) for b in (10, 25, 60, 150, 400))
    if part == "final":
        plan = tuple((b, FINAL_VARIANTS) for b in (10, 25, 60, 150, 400))
    if part == "combo":
        plan = ((25, COMBO_VARIANTS), (150, COMBO_VARIANTS), (400, COMBO_VARIANTS))
    if part == "ease":
        plan = ((10, EASE_VARIANTS), (25, EASE_VARIANTS), (60, EASE_VARIANTS),
                (150, EASE_VARIANTS))
    if part == "mix":
        plan = ((150, MIX_VARIANTS), (400, MIX_VARIANTS))
    for budget, variants in plan:
        M = defaultdict(lambda: defaultdict(list))
        n_users = 0
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
            scored = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff),
                              e["at"]) for e in seen if e["score"] > 0]
            if len(scored) < 3:
                continue
            m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
            implicit = [Example(e["mal_id"], float(min(max(m_ + CFG.implicit_offsets[e["status"]],
                                                            1), 10)),
                                CFG.implicit_weight * recency_weight(e["at"], cutoff), e["at"])
                        for e in seen if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
            pool = [m for m in pool_all if m not in {e["mal_id"] for e in es} - test_ids]
            test_items = [m for m, _, _ in test_s]
            truth = np.array([s for _, s, _ in test_s])
            n_users += 1
            for name, (extras, flags, neg) in variants.items():
                key, pred = keyed(pop, stackers[extras], store, scored, implicit,
                                  lambda s=seen, c=cutoff, n=neg: listed_weights(s, c, n), pool, test_items,
                                  flags, cutoff)
                ok = ~np.isnan(pred)
                if ok.sum() >= 5 and truth[ok].std() > 0:
                    M[name]["rho"].append(spearman(pred[ok], truth[ok]))
                    M[name]["ndcg"].append(ndcg_at(pred[ok], list(truth[ok]), 10))
                    M[name]["rmse"].append(float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2))))
                M[name]["pop"].append(top20_pop(key, pool, store))
                if len(targets) >= 3:
                    r = recall(key, pool, targets, 50)
                    if r is not None:
                        M[name]["r50"].append(r)
            set_flags(())
        report(f"sampled users with {budget} list entries ({n_users} users) "
               f"[{time.time() - t0:.0f}s]", M, list(variants))

    # --------------------------------------------------------- reference profile --
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    cal = one("SELECT metrics->'calibration' c FROM model_run WHERE user_id=%s"
              " AND metrics ? 'calibration' ORDER BY id DESC LIMIT 1", (uid,))
    cal = (cal or {}).get("c") or {}
    slope, icpt = float(cal.get("slope", 1.0)), float(cal.get("intercept", 0.0))
    le = query("""SELECT mal_id, score, status, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    variants = ({**STACK_VARIANTS, **PERSONAL_VARIANTS} if part == "all" else
                LEVEL_VARIANTS if part == "level" else GRID_VARIANTS if part == "grid" else
                MIX_VARIANTS if part == "mix" else EASE_VARIANTS if part == "ease" else
                COMBO_VARIANTS if part == "combo" else FINAL_VARIANTS if part == "final" else
                PROD_VARIANTS if part == "prod" else
                {**PERSONAL_VARIANTS, "dropped negative": STACK_VARIANTS["dropped negative"]})
    G = defaultdict(lambda: defaultdict(list))
    for h in (25, 30, 35, 40, 45, 50, 55, 60):
        test = rated[-h:]
        cutoff = test[0]["at"]
        targets = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        test_ids = {r["mal_id"] for r in test}
        pool = [m for m in pool_all if m not in {r["mal_id"] for r in le} - test_ids]
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        scored = [Example(r["mal_id"], float(r["score"]), recency_weight(r["at"], cutoff),
                          r["at"]) for r in hist if r["score"] > 0]
        m_ = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
        implicit = [Example(r["mal_id"],
                            float(min(max(m_ + CFG.implicit_offsets[r["status"]], 1), 10)),
                            CFG.implicit_weight * recency_weight(r["at"], cutoff), r["at"])
                    for r in hist if r["score"] == 0 and r["status"] in CFG.implicit_offsets]
        items = [r["mal_id"] for r in test]
        truth = np.array([float(r["score"]) for r in test])
        # the pre-population production model, for the no-regression gate
        today = np.clip(slope * fit_user(pop, stackers[()], store, scored, implicit,
                                         mode="personal").predict(items) + icpt, 1, 10)
        G["TODAY"]["rho"].append(spearman(today, truth))
        G["TODAY"]["ndcg"].append(ndcg_at(today, list(truth), 10))
        G["TODAY"]["rmse"].append(float(np.sqrt(np.mean((today - truth) ** 2))))
        for name, (extras, flags, neg) in variants.items():
            key, pred = keyed(pop, stackers[extras], store, scored, implicit,
                              lambda h=hist, c=cutoff, n=neg: listed_weights(h, c, n), pool, items,
                              flags, cutoff, slope, icpt, size_n=len(rated))
            G[name]["rho"].append(spearman(pred, truth))
            G[name]["ndcg"].append(ndcg_at(pred, list(truth), 10))
            G[name]["rmse"].append(float(np.sqrt(np.mean((pred - truth) ** 2))))
            G[name]["r50"].append(recall(key, pool, targets, 50) or 0.0)
            G[name]["pop"].append(top20_pop(key, pool, store))
        set_flags(())
    report(f"reference profile, 8 temporal splits at its real size [{time.time() - t0:.0f}s]",
           G, list(variants))
    for name in ("TODAY", *variants):
        print(f"  {name:<20} per split: {np.round(G[name]['rho'], 4).tolist()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=150)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--part", default="all", choices=("all", "personal", "level", "grid", "mix", "ease", "combo", "final", "prod"))
    a = ap.parse_args()
    main(a.max_eval, a.seed, a.part)
