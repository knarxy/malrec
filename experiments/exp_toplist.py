"""What sits at the top of the lists - and fixes for what should not.

Rank correlation and recall@50 are measured on titles people went on to
watch, which they chose; neither checks whether the single top pick is
plausible for someone who has never touched its genre. After the handover
moved to 150 the reference profile's Safe Bets #1 was Hajime no Ippo (2000,
75 episodes, boxing): population 9.07, personal 7.91, and no rated title of
his linked to it. The population layer rates devotee titles highly because
mostly devotees watch them.

Every configuration runs on the shipped blend (handover 150, cap 0.5, dials
on the n0=80 curve, long-memory mix, popularity prior). Side stories and
recaps are left out of the pools, as the discovery tabs now do.

  BASE       as shipped on 2026-10-03
  DIS k      order key minus k * t * max(pop - personal, 0), in display
             points; t = personal share / cap, the trust in the personal
             model (so thin lists, where it is weak, are left alone).
             Display unchanged.
  DISABS k   the same with |pop - personal|: any disagreement is risk
  EVID e0    the prediction itself: the population's weight for an item
             shrinks with the user's own evidence for it,
             lam_i = lam + (1 - lam) * t * e0 / (e0 + evidence_i)
  UNFAM p    order key minus p * t for a title with a genre absent from the
             user's list (the owner's red flag, as a rule)
  ACC k      (round 7) order key minus k * t * the "acclaim" part of the
             personal model's lift above the user's rated average - the
             consensus columns (MAL/AniList score, popularity, favourites,
             drop rate, polarity). Cowboy Bebop's personal 8.22 was +0.79
             acclaim, +0.04 content and had no co-rating link. Variants:
             ACCEV  the penalty fades with co-rating evidence, e0/(e0+ev)
             ACCC   the content part of the lift (genres, tags, studio, era,
                    affinity) may offset it as well

Metrics, on the Safe Bets key and on a Hidden Gems key (novelty 2.2, pool of
titles past popularity rank 1200 with >= 300 scorers):
  rho / RMSE      on newest ratings (changes only for EVID)
  r50 / ips       recall@50 of liked future titles, plain and IPS
  hit10 / bad10   share of the top 10 the user later rated >= their mean and
                  >= 7 / more than a point below their mean or dropped
  noev / unf / long / old
                  share of the top 10 with no evidence from the user's own
                  ratings / an unfamiliar genre / 50+ episodes / before 2005
"""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter, defaultdict

import numpy as np

from experiments.exp_retrieval import load_entries, pool_ids
from malrec.config import settings
from malrec.db import one, query
from malrec.eval import spearman
from malrec.rank import CONSENSUS_FEATURES, novelty, relevance_bonus
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import (
    Example,
    GlobalModel,
    fit_stacker,
    fit_user,
    recency_weight,
    size_weight,
    split_user,
)
from malrec.recsys.items import ItemStore
from malrec.recsys.service import calibration_from, user_examples, user_listed

CFG = settings()
BUDGETS = (25, 60, 150, 400)
N0, CAP, DIAL_N0 = 150.0, 0.5, 80.0
MIX, POP_PRIOR, GEMS_NOVELTY = 0.5, 0.5, 2.2
NOEV = 0.05                  # evidence below this counts as "none"
NON_GENRE = {"Shounen", "Seinen", "Shoujo", "Josei", "Kids"}
# MAL's genres proper (round 9); everything else in mal_genres is a theme
# (Medical, Showbiz, Childcare, ...) or a demographic. "Award Winning" is
# listed as a genre by MAL but says nothing about the content.
GENRES = {"Action", "Adventure", "Avant Garde", "Boys Love", "Comedy", "Drama", "Ecchi",
          "Erotica", "Fantasy", "Girls Love", "Gourmet", "Hentai", "Horror", "Mystery",
          "Romance", "Sci-Fi", "Slice of Life", "Sports", "Supernatural", "Suspense"}


def unfamiliar(pool, store, seen, genre_only: bool) -> np.ndarray:
    """1 for a title with a category absent from the user's list: any genre
    or theme (as shipped), or genres proper only."""
    def counts(g):
        return g in GENRES if genre_only else g not in NON_GENRE
    return np.array([float(any(counts(g) and seen[g] == 0
                               for g in (store.rows[m].get("mal_genres") or [])))
                     for m in pool])

# name: penalties (all scaled by trust, i.e. personal share / cap)
#   dis    k * max(pop - personal, 0) in display points (order only)
#   disabs k * |pop - personal|
#   evid   e0 in the evidence-weighted blend (changes the prediction)
#   unfam  points off for a genre absent from the user's list
#   long   points off for 50+ episodes
ROUND1 = {
    "BASE": {},
    "DIS 0.5": {"dis": 0.5}, "DIS 1": {"dis": 1.0}, "DIS 2": {"dis": 2.0},
    "DISABS 1": {"disabs": 1.0},
    "EVID 0.3": {"evid": 0.3}, "EVID 1": {"evid": 1.0},
    "UNFAM 0.3": {"unfam": 0.3},
}
ROUND2 = {
    "BASE": {},
    "U.3": {"unfam": 0.3}, "U.6": {"unfam": 0.6},
    "D.25": {"dis": 0.25}, "D.5": {"dis": 0.5},
    "U.3 D.25": {"unfam": 0.3, "dis": 0.25}, "U.3 D.5": {"unfam": 0.3, "dis": 0.5},
    "U.6 D.5": {"unfam": 0.6, "dis": 0.5},
    "U.3 L.3": {"unfam": 0.3, "long": 0.3}, "U.3 D.5 L.3": {"unfam": 0.3, "dis": 0.5, "long": 0.3},
}
# confirmation on another user split (--seed 4)
ROUND3 = {
    "BASE": {},
    "U.3 L.3": {"unfam": 0.3, "long": 0.3},
    "U.3 D.25 L.3": {"unfam": 0.3, "dis": 0.25, "long": 0.3},
    "U.3 L.5": {"unfam": 0.3, "long": 0.5},
    "U.6 L.3": {"unfam": 0.6, "long": 0.3},
    "U.3 D.25 L.5": {"unfam": 0.3, "dis": 0.25, "long": 0.5},
}
# thin lists: a length rule that also counts long-running ongoing series
# (MAL gives them 0 episodes), with and without the trust scaling
ROUND4 = {
    "BASE": {},
    "L.3": {"long": 0.3}, "LX.3": {"longx": 0.3}, "LXALL.3": {"longxall": 0.3},
    "LXALL.6": {"longxall": 0.6},
}
# Safe Bets guards on top of what shipped (2026-10-03 night): a title the
# current model predicts below the user's own mean got into Safe Bets on the
# long-memory adjustment alone (Mahouka, 6.90 vs mean 7.51, memory +0.65)
SHIPPED = {"unfam": 0.3, "dis": 0.25, "longxall": 0.3}
ROUND5 = {
    "SHIPPED": SHIPPED,
    "FLOOR mean": {**SHIPPED, "floor": 0.0},
    "FLOOR mean-.25": {**SHIPPED, "floor": -0.25},
    "MEMCAP .3": {**SHIPPED, "memcap": 0.3},
    "MEMCAP .15": {**SHIPPED, "memcap": 0.15},
}
ROUND6 = {k: ROUND5[k] for k in ("SHIPPED", "MEMCAP .3", "MEMCAP .15")}   # with --seed 4
# evidence check (2026-10-03): is a top pick backed by more than acclaim?
PROD = {**SHIPPED, "memcap": 0.15}
ROUND7 = {
    "PROD": PROD,
    "ACC .15": {**PROD, "acc": 0.15}, "ACC .3": {**PROD, "acc": 0.3},
    "ACCEV .3": {**PROD, "acc": 0.3, "acce": 0.1}, "ACCEV .6": {**PROD, "acc": 0.6, "acce": 0.1},
    "ACCC .3": {**PROD, "acc": 0.3, "acce": 0.1, "accc": True},
    "ACCC .6": {**PROD, "acc": 0.6, "acce": 0.1, "accc": True},
}
# round 8: how strong, and how fast the evidence fade (round 7 rose .3 -> .6)
SHIPPED7 = {**PROD, "acc": 0.6, "acce": 0.1}
ROUND8 = {
    "A.6 e.1": SHIPPED7,
    "A1 e.1": {**PROD, "acc": 1.0, "acce": 0.1}, "A1.5 e.1": {**PROD, "acc": 1.5, "acce": 0.1},
    "A.6 e.05": {**PROD, "acc": 0.6, "acce": 0.05}, "A.6 e.2": {**PROD, "acc": 0.6, "acce": 0.2},
    "A1 e.05": {**PROD, "acc": 1.0, "acce": 0.05}, "A1 e.2": {**PROD, "acc": 1.0, "acce": 0.2},
}
# round 9: the unfamiliar rule on genres proper instead of genres + themes
# (the reference profile's highest prediction, The Apothecary Diaries 8.08,
# sat at #16 for the theme "Medical"; [Oshi no Ko] at #58 for "Showbiz")
ROUND9 = {
    "PROD": SHIPPED7,
    "UG .3": {**SHIPPED7, "ugen": True},
    "UG .5": {**SHIPPED7, "ugen": True, "unfam": 0.5},
    "U off": {**SHIPPED7, "unfam": 0.0},
}
WATCH = {58514: "Apothecary", 52034: "Oshi no Ko", 50265: "Spy x Family", 6547: "Angel Beats"}
CONFIGS = ROUND1


def is_long(row: dict, ongoing: bool) -> bool:
    """50+ episodes; with `ongoing`, also a series still airing that began
    three or more years ago (MAL lists it with 0 episodes)."""
    if (row.get("num_episodes") or 0) >= 50:
        return True
    return bool(ongoing and not row.get("num_episodes") and row.get("status") == "currently_airing"
                and (row.get("season_year") or 2100) <= 2023)


class _Share:
    def __init__(self, lam):
        self.personal_share = lam


def zs(x):
    sd = float(np.nanstd(x))
    return (x - np.nanmean(x)) / sd if sd > 1e-9 else np.zeros_like(x)


def derived_ids() -> set[int]:
    """Side stories and recaps - left to the Side Stories tab, or out."""
    return {r["src"] for r in query("SELECT DISTINCT src FROM relation"
                                    " WHERE relation_type IN ('parent_story', 'full_story')")}


class Parts:
    def __init__(self, G, store, scored, implicit, listed, ids, n):
        per = fit_user(G.pop, G.stacker, store, scored, implicit, mode="personal",
                       listed=listed)
        st = fit_user(G.pop, G.stacker, store, scored, implicit, mode="stack", listed=listed)
        self.S = st.predict(ids)
        self.P = per.predict(ids)
        self.P = np.where(np.isnan(self.P), self.S, self.P)
        self.ev = st.fold.evidence(ids)
        self.n = n
        self.lam = min(size_weight(n, N0, 15.0), CAP)
        self.trust = self.lam / CAP
        self.mu = st.fold.mu
        self._st = st
        self.A, self.C = self._lift_parts(per, scored, ids)

    @staticmethod
    def _lift_parts(per, scored, ids):
        """The personal model's lift over the user's average rated title,
        split into its acclaim columns and everything else (content)."""
        A, C = np.zeros(len(ids)), np.zeros(len(ids))
        if per.ridge is None or per.vocab is None:
            return A, C
        rated = per.rows([x.mal_id for x in scored])
        rows = per.rows(ids)
        if not rated or not rows:
            return A, C
        coef = np.asarray(per.ridge.coef_, dtype=float)
        names = list(per.vocab.names)
        cons = np.array([n in CONSENSUS_FEATURES for n in names] + [False] * (len(coef) - len(names)))
        D = (per._features(rows) - per._features(rated).mean(0)) * coef
        pos = {m: j for j, m in enumerate(ids)}
        for r, d in zip(rows, D):
            j = pos[r["mal_id"]]
            A[j], C[j] = d[cons].sum(), d[~cons].sum()
        return A, C

    def relevance(self, ids):
        return self._st.relevance(ids)

    def pred(self, cfg, k):
        lam = np.full(len(k), self.lam)
        if "evid" in cfg:
            e0 = cfg["evid"]
            lam = lam + (1 - lam) * self.trust * e0 / (e0 + self.ev[k])
        return lam * self.P[k] + (1 - lam) * self.S[k]


def key_for(short, long_, k, z_s, z_l, popz, cfg, cal, unfam, longs, longx=None):
    a, b = cal
    share = size_weight(short.n, DIAL_N0, 15.0)
    raw = short.pred(cfg, k)
    raw2 = long_.pred(cfg, k)
    shown = np.clip(a * raw + b, 1, 10)
    key = shown + relevance_bonus(_Share(share), z_s)
    key2 = np.clip(a * raw2 + b, 1, 10) + relevance_bonus(_Share(share), z_l)
    ok = np.isfinite(key) & np.isfinite(key2)
    m1, s1 = key[ok].mean(), key[ok].std() + 1e-9
    m2, s2 = key2[ok].mean(), key2[ok].std() + 1e-9
    mixed = m1 + s1 * ((1 - MIX) * (key - m1) / s1 + MIX * (key2 - m2) / s2)
    bonus = np.where(ok, mixed - key, 0.0)
    if "memcap" in cfg:
        bonus = np.clip(bonus, -cfg["memcap"], cfg["memcap"])
    wp = POP_PRIOR * (1.0 - share)
    if wp > 1e-3:
        bonus = bonus + wp * float(key[ok].std()) * popz
    gap = short.S[k] - short.P[k]
    t = short.trust
    bonus = bonus - t * (cfg.get("dis", 0.0) * a * np.maximum(gap, 0.0)
                         + cfg.get("disabs", 0.0) * a * np.abs(gap)
                         + cfg.get("unfam", 0.0) * unfam + cfg.get("long", 0.0) * longs
                         + (cfg.get("longx", 0.0) * longx if longx is not None else 0.0))
    if longx is not None:
        bonus = bonus - cfg.get("longxall", 0.0) * longx      # no trust scaling
    if cfg.get("acc"):
        lift = short.A[k] - (np.maximum(short.C[k], 0.0) if cfg.get("accc") else 0.0)
        fade = cfg["acce"] / (cfg["acce"] + short.ev[k]) if "acce" in cfg else 1.0
        bonus = bonus - t * cfg["acc"] * a * np.maximum(lift, 0.0) * fade
    out = key + bonus
    if "floor" in cfg:
        # Safe Bets only: nothing predicted below the user's own mean
        mu = a * short.mu + b
        out = np.where(shown >= mu + cfg["floor"], out, -np.inf)
    return out, shown


def top_metrics(key, pool, k_idx, short, liked, disliked, prop, store, unfam, M, tag,
                unfam_g=None):
    s = np.where(np.isfinite(key), key, -np.inf)
    order = np.argsort(-s)
    top10 = order[:10]
    t10 = [pool[i] for i in top10]
    M[f"{tag}hit10"].append(len(set(t10) & liked) / 10)
    M[f"{tag}bad10"].append(len(set(t10) & disliked) / 10)
    M[f"{tag}noev"].append(float(np.mean(short.ev[k_idx[top10]] < NOEV)))
    M[f"{tag}unf"].append(float(np.mean(unfam[top10] > 0)))
    if unfam_g is not None:
        M[f"{tag}unfg"].append(float(np.mean(unfam_g[top10] > 0)))
    M[f"{tag}long"].append(float(np.mean([is_long(store.rows[m], True) for m in t10])))
    M[f"{tag}old"].append(float(np.mean([(store.rows[m].get("season_year") or 2100) < 2005
                                         for m in t10])))
    if liked:
        top50 = {pool[i] for i in order[:50]}
        w = {t: 1.0 / prop.get(t, 1e-3) for t in liked}
        M[f"{tag}r50"].append(len(top50 & liked) / len(liked))
        M[f"{tag}ips"].append(sum(w[t] for t in top50 & liked) / sum(w.values()))


def score_user(short, long_, ids, items, truth, pools, liked, disliked, prop, popul, store,
               seen_genres, cal, R):
    at = {m: j for j, m in enumerate(ids)}
    k_items = np.array([at[m] for m in items])
    for name, cfg in CONFIGS.items():
        M = R[name]
        p = np.clip(cal[0] * short.pred(cfg, k_items) + cal[1], 1, 10)
        if truth.std() > 0:
            M["rho"].append(spearman(p, truth))
            M["rmse"].append(float(np.sqrt(np.mean((p - truth) ** 2))))
        for tag, pool, nov_w in (("", pools[0], 0.0), ("g:", pools[1], GEMS_NOVELTY)):
            if tag:
                cfg = {x: v for x, v in cfg.items() if x != "floor"}
            k = np.array([at[m] for m in pool])
            unfam = unfamiliar(pool, store, seen_genres, False)
            unfam_g = unfamiliar(pool, store, seen_genres, True)
            popz = zs(np.array([-math.log10(popul.get(m, 99999)) for m in pool]))
            longs = np.array([float(is_long(store.rows[m], False)) for m in pool])
            longx = np.array([float(is_long(store.rows[m], True)) for m in pool])
            key, shown = key_for(short, long_, k, short.relevance(pool), long_.relevance(pool),
                                 popz, cfg, cal, unfam_g if cfg.get("ugen") else unfam,
                                 longs, longx)
            if nov_w:
                sd = float(np.nanstd(shown))
                key = key + nov_w * min(1.0, sd / CFG.novelty_ref_sd) * np.array(
                    [novelty(popul.get(m)) for m in pool])
            top_metrics(key, pool, k, short, liked & set(pool), disliked, prop, store,
                        unfam, M, tag, unfam_g)


def report(title, R):
    print(f"\n== {title} ==", flush=True)
    cols = ["rho", "rmse", "r50", "ips", "hit10", "bad10", "noev", "unf", "unfg", "long", "old"]
    print(f"{'Safe Bets':<13}" + "".join(f"{c:>7}" for c in cols)
          + "   | Hidden Gems" + "".join(f"{c:>7}" for c in ("hit10", "bad10", "noev", "unf",
                                                             "long", "old")))
    for name in CONFIGS:
        M = R[name]
        f = lambda k, M=M: f"{np.mean(M[k]):7.3f}" if M[k] else f"{'-':>7}"
        print(f"{name:<13}" + "".join(f(c) for c in cols) + "   |            "
              + "".join(f("g:" + c) for c in ("hit10", "bad10", "noev", "unf", "long", "old")))


def main(max_eval: int, seed: int, budgets: tuple[int, ...]):
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
    drop = derived_ids()
    safe_all = [m for m in pool_ids(store) if m not in drop]
    gems_all = [m for m, r in store.rows.items()
                if (r.get("mal_num_scoring_users") or 0) >= 300 and m not in drop
                and (r.get("mal_popularity") or 99999) > 1200
                and r.get("format_class") == "main" and r.get("nsfw") in (None, "white")
                and r.get("status") != "not_yet_aired"]
    popul = {m: float(r.get("mal_popularity") or 99999) for m, r in store.rows.items()}
    print(f"{len(users)} users ({len(A)} fit / {len(B)} stack / {len(C)} eval); pools: safe "
          f"{len(safe_all)}, gems {len(gems_all)}; {len(drop)} side stories / recaps left out "
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

    def genres_of(mal_ids):
        return Counter(g for m in mal_ids for g in (store.rows.get(m, {}).get("mal_genres") or []))

    for budget in budgets:
        R = defaultdict(lambda: defaultdict(list))
        for u in C[:max_eval]:
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, rng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            liked = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            test_ids = {m for m, _, _ in test_s}
            disliked = {m for m, s, _ in test_s if s < mean - 1}
            disliked |= {e["mal_id"] for e in es if e["status"] == "dropped" and e["at"] >= cutoff}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in rng.permutation(len(history))[:budget]]
            sc, im, li = examples(seen, cutoff, 0.5)
            if len(sc) < 5:
                continue
            scL, imL, liL = examples(seen, cutoff, None)
            listed = {e["mal_id"] for e in es} - test_ids - disliked
            pools = ([m for m in safe_all if m not in listed],
                     [m for m in gems_all if m not in listed])
            items = [m for m, _, _ in test_s]
            ids = list(dict.fromkeys(pools[0] + pools[1] + items))
            short = Parts(G, store, sc, im, li, ids, len(sc))
            long_ = Parts(G, store, scL, imL, liL, ids, len(sc))
            score_user(short, long_, ids, items, np.array([s for _, s, _ in test_s]), pools,
                       liked, disliked, prop, popul, store,
                       genres_of(e["mal_id"] for e in seen), (1.0, 0.0), R)
        report(f"sampled users, {budget} list entries ({len(R[next(iter(CONFIGS))]['r50'])} users with liked targets) "
               f"[{time.time() - t0:.0f}s]", R)

    # ------------------------------------------------------------ dennis --
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (settings().malrec_user,))["id"]
    le = query("""SELECT mal_id, score, status,
                         coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    n_real = len(rated)
    mean = float(np.mean([r["score"] for r in rated]))
    fitted = {}
    for h in (25, 30, 35, 40, 45, 50, 55, 60):
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        disliked = {r["mal_id"] for r in test if r["score"] < mean - 1}
        disliked |= {r["mal_id"] for r in le if r["status"] == "dropped" and r["at"] >= cutoff}
        listed = {r["mal_id"] for r in le} - test_ids - disliked
        pools = ([m for m in safe_all if m not in listed], [m for m in gems_all if m not in listed])
        items = [r["mal_id"] for r in test]
        ids = list(dict.fromkeys(pools[0] + pools[1] + items))
        parts = []
        for hl in (0.35, float("inf")):
            sc, im = user_examples(uid, cutoff, hl)
            parts.append(Parts(G, store, sc, im, user_listed(uid, cutoff, hl), ids, n_real))
        hist = [r["mal_id"] for r in le if r["at"] < cutoff and r["mal_id"] not in test_ids]
        fitted[h] = (parts, pools, items, test, ids, disliked, genres_of(hist))
    pr, tr = [], []
    for h in (30, 40, 50):
        (short, _), _, items, test, ids, _, _ = fitted[h]
        at = {m: j for j, m in enumerate(ids)}
        pr += list(short.pred({}, np.array([at[m] for m in items])))
        tr += [float(r["score"]) for r in test]
    c = calibration_from(pr, tr)
    cal = (c["slope"], c["intercept"])
    R = defaultdict(lambda: defaultdict(list))
    for h, ((short, long_), pools, items, test, ids, disliked, seen_g) in fitted.items():
        liked = {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)}
        score_user(short, long_, ids, items, np.array([float(r["score"]) for r in test]), pools,
                   liked, disliked, prop, popul, store, seen_g, cal, R)
    report(f"reference profile, 8 temporal splits ({n_real} ratings, calibration {cal})", R)
    # what each configuration puts on top of its real, current list
    print("\nreference profile today, Safe Bets top 10 per configuration:")
    sc, im = user_examples(uid, None, 0.35)
    scL, imL = user_examples(uid, None, float("inf"))
    listed = {r["mal_id"] for r in le}
    pool = [m for m in safe_all if m not in listed]
    ids = pool
    short = Parts(G, store, sc, im, user_listed(uid, None, 0.35), ids, n_real)
    long_ = Parts(G, store, scL, imL, user_listed(uid, None, float("inf")), ids, n_real)
    k = np.arange(len(pool))
    g = genres_of(listed)
    unfam = unfamiliar(pool, store, g, False)
    unfam_g = unfamiliar(pool, store, g, True)
    popz = zs(np.array([-math.log10(popul.get(m, 99999)) for m in pool]))
    longs = np.array([float((store.rows[m].get("num_episodes") or 0) >= 50) for m in pool])
    longx = np.array([float(is_long(store.rows[m], True)) for m in pool])
    for name, cfg in CONFIGS.items():
        key, _ = key_for(short, long_, k, short.relevance(pool), long_.relevance(pool), popz,
                         cfg, cal, unfam_g if cfg.get("ugen") else unfam, longs, longx)
        order = np.argsort(-np.where(np.isfinite(key), key, -np.inf))
        top = order[:10]
        print(f"  {name:<12} " + " | ".join(store.rows[pool[i]]["title"][:20] for i in top))
        pos = {pool[i]: r + 1 for r, i in enumerate(order)}
        print(f"  {'':<12} " + ", ".join(f"{lbl} #{pos[m]}" for m, lbl in WATCH.items() if m in pos))
    print(f"\n[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    a = ap.parse_args()
    CONFIGS = {1: ROUND1, 2: ROUND2, 3: ROUND3, 4: ROUND4, 5: ROUND5, 6: ROUND6,
               7: ROUND7, 8: ROUND8, 9: ROUND9}[a.round]
    main(a.max_eval, a.seed, tuple(int(x) for x in a.budgets.split(",")))
