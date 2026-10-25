"""The benchmark every ranking change is checked against (make bench).

Same protocol as the experiments, but frozen and run through the production
code, so a difference between two runs can only come from a code change:

  data      one snapshot of the sampled public lists, the item catalogue and
            the reference profile's list (`--snapshot`, kept on the server
            under experiments/bench/data, never in git). The live database
            keeps changing; the benchmark does not.
  population model
            fitted on 72 % of the snapshot's users, its stacker on 13 %; the
            rest are evaluated and never seen by anything fitted. Cached by a
            hash of the population code (cf.py, hybrid.py, items.py) - a run
            after a ranking change takes minutes, a change to the population
            layer refits it (about 15 minutes).
  per user  production all the way: recsys.hybrid.fit_user with the
            configured blend, recsys.scorer.HybridScorer, rank.score_candidates
            (relevance, popularity prior, risk) and rank.rerank (novelty,
            genre balance) - the Safe Bets build minus the database, with
            the long-memory model (safe_bets_memory_mix) and each tab's
            learned order when configured, as served. Pool: what
            eligible_candidates allows (not on the list at the cutoff, main
            format, not NSFW, >= 2000 scorers, aired, prerequisites
            completed), side stories, recaps and Up Next's continuations left
            out (recsys/pool.py). Entries added after the cutoff stay in it:
            a title dropped later is a miss the top 10 can show (bad10).

Groups: held-out users given 25 / 150 / 400 of their older list entries
(newest 20 % of their ratings are the future), and the reference profile
(settings.malrec_user) over 8 temporal splits.

Metrics, per user (paired between runs):
  rho / rmse   rank correlation and error of the shown score on the future
               ratings
  r50          share of liked future titles (>= max(own mean, 7)) in the
               Safe Bets top 50
  ips50        the same with each liked title weighted by 1 / the share of
               fit lists it is on, so following popularity alone scores
               little
  hit10        share of the top 10 they went on to like
  bad10        share of the top 10 they rated a point under their mean or
               dropped
  r50new / r50old
               r50 for the liked titles at most a year old at the cutoff and
               for the older ones (described: recall alone rewards a list
               for drifting to the new season)
  disc_* / gems_*
               the same four for the Discover and Hidden Gems lists (Discover
               without the Safe Bets list, as production builds it; Hidden Gems
               from its pool rule, recsys/popularity.py gems_popularity)
and per group: overlap (mean Jaccard of two users' top 20 - lower is more
personal) and pop# (median MAL popularity rank of the top 20).

The gate (audit 2026-10-23, P1-9): only the PRIMARY metrics of the sampled
groups decide, as one family of paired tests with Holm's correction at
ALPHA - a run that compared forty metrics at two standard errors each failed
by chance about once in two. "better"/"WORSE" is a change that survives the
correction; "better?"/"worse?" one beyond two standard errors that does not,
or any change of a metric that is only described; otherwise "noise" (or
"same" when nothing moved). A WORSE, or a reference profile whose rho loses
on more splits than it wins, fails the run (exit 1).

  python -m experiments.bench --snapshot        freeze the data (once)
  python -m experiments.bench                   run, compare with baseline.json
  python -m experiments.bench --save-baseline   run and make it the baseline
  python -m experiments.bench --freeze          the live tables as the first frozen
                                                run read them (an experiment's arms)
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import pickle
import resource
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from malrec import rank
from malrec.config import settings
from malrec.eval import spearman
from malrec.recsys import ltr, popularity
from malrec.recsys import pool as discovery_pool
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import fit_stacker, fit_user, split_user
from malrec.recsys.scorer import HybridScorer
from malrec.recsys.service import blend_mode, blend_opts
from malrec.surfaces import AGED, SPECS

HERE = Path(__file__).resolve().parent / "bench"
DATA = HERE / "data"
BASELINE = HERE / "baseline.json"
LATEST = HERE / "latest.json"
SRC = Path(__file__).resolve().parents[1] / "src" / "malrec"
POP_CODE = ("recsys/cf.py", "recsys/hybrid.py", "recsys/items.py", "recsys/series.py")

SEED = 3
BUDGETS = (25, 150, 400)
REF_SPLITS = (25, 30, 35, 40, 45, 50, 55, 60)
LIMIT = 60
STATUS = ("watching", "completed", "on_hold", "dropped", "plan_to_watch")
# what the gate decides on (per sampled group): the shown score's order, and
# each tab's recall and top-10 hits, plus Safe Bets' misses
PRIMARY = ("rho", "r50", "hit10", "bad10", "disc_r50", "disc_hit10", "gems_r50", "gems_hit10")
ALPHA = 0.05
HIGHER = {"rho": True, "rmse": False, "r50": True, "ips50": True, "hit10": True,
          "bad10": False, "agegap": False, "disc_r50": True, "disc_ips50": True,
          "disc_hit10": True, "disc_bad10": False, "disc_agegap": False, "gems_r50": True,
          "gems_ips50": True, "gems_hit10": True, "gems_bad10": False, "gems_agegap": False,
          "r50new": True, "r50old": True, "disc_r50new": True, "disc_r50old": True,
          "gems_r50new": True, "gems_r50old": True}
CFG = settings()


# ---------------------------------------------------------------- snapshot --

def snapshot() -> None:
    """Freeze the data. Streamed into columns (db.stream), so the sample can
    grow to tens of millions of entries without the snapshot needing more
    than a few hundred MB."""
    from array import array

    from malrec.db import one, query, stream
    from malrec.recsys.items import ItemStore
    from malrec.surfaces import derived_ids
    t0 = time.time()
    DATA.mkdir(parents=True, exist_ok=True)
    cols = (array("i"), array("i"), array("b"), array("b"), array("d"))
    code = {s: k for k, s in enumerate(STATUS)}
    for u, m, sc, st, a in stream("""
            SELECT r.user_id, r.mal_id, r.score, r.status,
                   extract(epoch FROM coalesce(r.finished_at::timestamptz, r.updated_at))::float8
              FROM cf_rating r JOIN cf_user u ON u.id = r.user_id
             WHERE u.state = 'done' AND r.updated_at IS NOT NULL
             ORDER BY r.user_id, 5, r.mal_id"""):
        for col, v in zip(cols, (u, m, sc, code.get(st, 1), a)):
            col.append(v)
    np.savez_compressed(DATA / "entries.npz", user=np.frombuffer(cols[0], dtype=np.int32),
                        mal_id=np.frombuffer(cols[1], dtype=np.int32),
                        score=np.frombuffer(cols[2], dtype=np.int8),
                        status=np.frombuffer(cols[3], dtype=np.int8),
                        at=np.frombuffer(cols[4], dtype=np.float64))
    ref_name = CFG.malrec_user
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (ref_name,))["id"]
    le = query("""SELECT mal_id, score, status,
                         coalesce(finished_at::timestamptz, updated_at, now()) AS at
                    FROM list_entry WHERE user_id=%s""", (uid,))
    cal = one("SELECT metrics->'calibration' c FROM model_run WHERE user_id=%s"
              " AND metrics ? 'calibration' ORDER BY id DESC LIMIT 1", (uid,))
    cal = (cal or {}).get("c") or {}
    with open(DATA / "static.pkl", "wb") as f:
        pickle.dump({"store": ItemStore.load(), "derived": set(derived_ids()),
                     "ref": {"le": le, "cal": (float(cal.get("slope", 1.0)),
                                               float(cal.get("intercept", 0.0)))},
                     "created": dt.datetime.now(dt.UTC).isoformat()}, f)
    print(f"snapshot: {len(set(cols[0]))} users, {len(cols[0])} entries, reference {len(le)}, "
          f"peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB "
          f"[{time.time() - t0:.0f}s]")


class Entries(Mapping):
    """The snapshot's list entries by user, kept as columns; one user's
    entries become dicts ({mal_id, score, status, at}) only when asked for -
    as dicts all at once they would not fit in memory on a large sample."""

    def __init__(self, user, mal_id, score, status, at):
        order = np.argsort(user, kind="stable")          # keeps each user's own order
        self.user, self.mal_id = user[order], mal_id[order]
        self.score, self.status, self.at = score[order], status[order], at[order]
        uniq, start, count = np.unique(self.user, return_index=True, return_counts=True)
        self._span = {int(u): (int(a), int(a + c)) for u, a, c in zip(uniq, start, count)}
        self._n_scored = dict(zip(uniq.tolist(),
                                  np.add.reduceat((self.score > 0).astype(np.int64),
                                                  start).tolist()))

    def __getitem__(self, u) -> list[dict]:
        a, b = self._span.get(int(u), (0, 0))
        return [{"mal_id": m, "score": sc, "status": STATUS[st],
                 "at": dt.datetime.fromtimestamp(t, dt.UTC)}
                for m, sc, st, t in zip(self.mal_id[a:b].tolist(), self.score[a:b].tolist(),
                                        self.status[a:b].tolist(), self.at[a:b].tolist())]

    def __iter__(self):
        return iter(self._span)

    def __len__(self) -> int:
        return len(self._span)

    def users_with_scored(self, n: int) -> list[int]:
        """Users with at least `n` scores, sorted."""
        return sorted(u for u, k in self._n_scored.items() if k >= n)

    def columns(self, users) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(user, mal_id, score) of these users' entries, in `users` order."""
        idx = np.concatenate([np.arange(*self._span[u]) for u in users if u in self._span]
                             or [np.zeros(0, dtype=np.int64)])
        return self.user[idx], self.mal_id[idx], self.score[idx]


def load_snapshot():
    z = np.load(DATA / "entries.npz")
    entries = Entries(z["user"], z["mal_id"], z["score"], z["status"], z["at"])
    with open(DATA / "static.pkl", "rb") as f:
        static = pickle.load(f)
    return entries, static


# ------------------------------------------------------------ population --

# --freeze: the live tables a run reads (franchise, series, members, relations)
# as the first frozen run read them, so the arms of one experiment see the
# same data even when a sync changes it between them; delete it to start over
FROZEN = DATA / "live.pkl"
LIVE: dict | None = None


def frozen(key: str, read):
    """read(), or with --freeze what it returned on the first frozen run."""
    if LIVE is None:
        return read()
    if key not in LIVE:
        LIVE[key] = read()
        FROZEN.write_bytes(pickle.dumps(LIVE))
    return LIVE[key]


# set by run(): a hash of the live series and franchise tables (groupings)
GROUPS = ""
# set by run(): the snapshot's day and every title's rank by age-fair member
# count (recsys/popularity.py), for the Hidden Gems pool rule
SNAP_DAY = dt.date(2026, 10, 10)
GEM_RANKS: dict[int, int] = {}            # the age-fair rank, for the tail metric
GEM_BARS: list[float] = [0.0, 0.0]          # popularity.bars, for the pool rule


def popularity_columns(store, day: dt.date) -> None:
    """Members and start dates are not in the snapshot: they come from the
    live table (ten days on, a few percent more members for new titles), the
    ranks are taken as of the snapshot's day."""
    from malrec.db import query
    rows = frozen("members", lambda: query(
        "SELECT mal_id, mal_num_list_users, start_date FROM anime"))
    for r in rows:
        row = store.rows.get(r["mal_id"])
        if row is not None:
            row["members"], row["start_date"] = r["mal_num_list_users"], r["start_date"]
    GEM_RANKS.clear()
    GEM_RANKS.update(popularity.grown_ranks(
        ((r["mal_id"], r["mal_num_list_users"], r["start_date"]) for r in rows), day))
    GEM_BARS[:] = popularity.bars(r["mal_num_list_users"] for r in rows)


def gem(row) -> bool:
    return popularity.is_gem(row, CFG.gems_popularity, (GEM_BARS[0], GEM_BARS[1]), SNAP_DAY)


def groupings(store) -> str:
    """Series and franchises come from the live tables, not the snapshot:
    they are derived from MAL's relations by code (recsys/series.py), and a
    change to that code is what a run should measure. Overwrites the
    snapshot store's franchise ids; returns a hash of both tables."""
    from malrec.db import query
    fr = {r["mal_id"]: r["franchise_id"] for r in frozen("franchise", lambda: query(
        "SELECT mal_id, franchise_id FROM franchise"))}
    se = {r["mal_id"]: r["series_id"] for r in frozen("series", lambda: query(
        "SELECT mal_id, series_id FROM series"))}
    for m, row in store.rows.items():
        row["franchise_id"] = fr.get(m, m)
    return hashlib.sha256(repr((sorted(fr.items()), sorted(se.items()))).encode()).hexdigest()[:8]


def code_hash() -> str:
    h = hashlib.sha256()
    for p in POP_CODE:
        h.update((SRC / p).read_bytes())
    # the stacker's training lists are series-weighted
    h.update(f"{GROUPS}:{CFG.series_weight_power}".encode())
    h.update(repr(vars(CFParams())).encode())
    h.update(str(CFG.relevance_ease).encode())
    snap = DATA / "entries.npz"         # a new snapshot is new data: refit
    if snap.exists():
        st = snap.stat()
        h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:12]


def population(entries, store, A, B, cache: bool = True, drop: frozenset[int] = frozenset()):
    """The population model and stacker on users A / B; `drop` titles are
    left out of everything (an experiment's hidden titles; then no cache)."""
    path = DATA / f"pop-{code_hash()}.pkl"
    cache = cache and not drop
    if cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f), "cached"
    u, m, sc = entries.columns(A)
    if drop:
        keep = ~np.isin(m, np.fromiter(drop, dtype=np.int64))
        u, m, sc = u[keep], m[keep], sc[keep]
    oa, oi = u.astype(np.int64), m.astype(np.int64)
    r = sc > 0
    pop = PopulationModel(CFParams()).fit(oa[r], oi[r], sc[r].astype(np.float64))
    pop.fit_occurrence(oa, oi)
    if CFG.relevance_ease:
        pop.fit_ease(oa, oi)
    stacker = fit_stacker(pop, {u: [(e["mal_id"], float(e["score"]), e["at"])
                                    for e in entries[u] if e["score"] > 0
                                    and e["mal_id"] not in drop] for u in B},
                          store, seed=SEED)
    if cache:
        for old in DATA.glob("pop-*.pkl"):
            old.unlink()
        with open(path, "wb") as f:
            pickle.dump((pop, stacker), f)
    return (pop, stacker), "fitted"


# --------------------------------------------------------------- per user --

def examples(seen, cutoff):
    """A list's entries as a user model's input (shared with the ranker's
    training, recsys/ltr.py)."""
    return ltr.history_examples(seen, cutoff)


def profile_of(listed: dict[int, float], store) -> dict[str, float]:
    """surfaces.genre_profile on in-memory data."""
    return ltr.genre_mix(listed, store)


# ------------------------------------------------------- learned order --

# set by run(): the learned order of each tab switched on (ltr.SURFACES)
RANKERS: dict[str, ltr.OrderRanker] = {}
# set by run(): per title, the share of fit lists it is on (+1 / +2), for
# IPS recall - a hit on a title everyone watches says little, so each liked
# title counts 1 / propensity and following popularity alone does not score
PROP: np.ndarray = np.zeros(0)


def order_ranker(entries, store, pop, stacker, B, pool, surface: str = "safe_bets",
                 taken_pool: list[int] | None = None, relations=None,
                 taken_ranker: ltr.OrderRanker | None = None) -> tuple[ltr.OrderRanker, str]:
    """A tab's learned order for the benchmark's population model, on its
    stacker users - as `malrec ranker train` does for the live one. Cached
    with the population model's hash, recsys/ltr.py's, rank.py's,
    recsys/pool.py's, the tab and the ranking settings' (ltr.fingerprint);
    the newest six are kept."""
    pool_key = (f"{CFG.gems_popularity}:{len(pool)}".encode()   # the shipped rule keeps old keys
                if surface == "hidden_gems" and CFG.gems_popularity != "rank" else b"")
    h = hashlib.sha256((SRC / "recsys/ltr.py").read_bytes() + (SRC / "rank.py").read_bytes()
                       + (SRC / "recsys/pool.py").read_bytes()
                       + str(CFG.ranker_train_as_served).encode()
                       + ltr.fingerprint().encode() + surface.encode()
                       + ",".join(ltr.FEATURES).encode() + pool_key).hexdigest()[:8]
    path = DATA / f"ltr-{code_hash()}-{h}.pkl"
    if path.exists():
        path.touch()
        return ltr.OrderRanker.loads(path.read_bytes()), "cached"
    r = ltr.train(pop, stacker, store, pool, {u: entries[u] for u in B}, surface=surface,
                  taken_pool=taken_pool, relations=relations, taken_ranker=taken_ranker)
    path.write_bytes(r.dumps())
    for old in sorted(DATA.glob("ltr-*.pkl"), key=lambda f: f.stat().st_mtime)[:-6]:
        old.unlink()
    return r, "trained"


def make_scorer(pop, stacker, store, sc, im, li, calibration, scorer_cls=HybridScorer,
                um=None, **attrs):
    """The user's production scorer. Experiments pass a HybridScorer subclass
    (and attributes for it) to try a change to one part, and a fitted `um`
    to share it between variants."""
    if um is None:
        um = fit_user(pop, stacker, store, sc, im, mode=blend_mode(len(sc)),
                      blend_of="personal", **blend_opts(), listed=li)
    scorer = scorer_cls(um, CFG.model_mode, {}, {}, calibration,
                        {e.mal_id: e.score for e in sc}, {})
    for k, v in attrs.items():
        setattr(scorer, k, v)
    return scorer


def year_of(store):
    def f(m: int) -> int | None:
        return (store.rows.get(m) or {}).get("season_year")
    return f


def ranked(scorer, rows, li, familiar, store, surface: str = "safe_bets",
           taken: frozenset[int] | set[int] = frozenset(), history: list[dict] | None = None,
           long_model=None, cutoff=None):
    """One tab's build (surfaces.build_surface minus the database): the scored
    pool (Candidates, best first) and the reranked list. `history` (the list
    entries the user model saw) feeds the learned order's taste features and,
    with `cutoff`, the release-age balance (surfaces.user_age_mix);
    `long_model` is Safe Bets' every-rating-equal model (safe_bets_memory_mix)."""
    ref_year = cutoff.year if cutoff is not None else None
    spec = SPECS[surface]
    if surface == "hidden_gems":
        rows = [r for r in rows if gem(store.rows[r["mal_id"]])]
    if taken:                                   # Discover starts where Safe Bets ends
        rows = [r for r in rows if r["mal_id"] not in taken]
    ranker = RANKERS.get(surface)
    learned = (ltr.RankContext(ranker, familiar, profile_of(li, store), len(scorer._scores),
                               getattr(CFG, ltr.SURFACES[surface]),
                               ltr.user_taste(history or [], store),
                               ltr.age_taste(history or [], store, ref_year)
                               if ref_year is not None else None)
               if ranker is not None else None)
    ages = (rank.age_mix(history or [], year_of(store), ref_year)
            if ref_year is not None and surface in AGED else None)
    mix = CFG.safe_bets_memory_mix if long_model is not None else 0.0
    real = rank.familiar_genres
    rank.familiar_genres = lambda _uid: familiar
    try:
        pool = rank.score_candidates(None, scorer, rows, enrich=False, long_model=long_model,
                                     mix=mix, learned=learned, **ltr.surface_key(surface))
    finally:
        rank.familiar_genres = real
    floor = CFG.safe_bets_floor
    if surface == "safe_bets" and floor > -10:
        mean_shown = float(scorer.calibrate(np.array([scorer.baseline]))[0])
        pool = [c for c in pool if c.predicted >= mean_shown + floor]
    top = rank.rerank(pool, spec.novelty_weight, spec.diversity_weight, LIMIT,
                      considered=[], profile=profile_of(li, store), ages=ages)
    return pool, top


def safe_bets(pop, stacker, store, sc, im, li, pool_rows, calibration, familiar,
              scorer_cls=HybridScorer, um=None, **attrs):
    """The Safe Bets build for one user, through the production functions."""
    scorer = make_scorer(pop, stacker, store, sc, im, li, calibration, scorer_cls, um, **attrs)
    _, top = ranked(scorer, pool_rows, li, familiar, store)
    return scorer, [c.mal_id for c in top]


def ips50(top: list[int], liked: set[int]) -> float:
    """Recall@50 with each liked title weighted by 1 / propensity (PROP)."""
    def w(m: int) -> float:
        return 1.0 / (float(PROP[m]) if 0 <= m < len(PROP) and PROP[m] > 0 else 1.0)
    return sum(w(t) for t in set(top[:50]) & liked) / sum(w(t) for t in liked)


def age_metrics(top: list[int], ages, prefix: str = "") -> dict:
    """How far the top 20's release-age mix is from the user's own (rank.age_mix
    of their history; total variation distance, 0 = the same mix, 1 = no
    overlap), the share of it at most a year old and five or more years old
    (old20, the back catalogue). Recall rewards recent
    titles by construction - what people rate next is mostly what just
    aired - so this is what catches a list drifting to the new season."""
    if ages is None or len(top) < 20:
        return {}
    b = [ages.bucket(m) for m in top[:20]]
    b = [x for x in b if x is not None]
    if not b:
        return {}
    q = np.bincount(b, minlength=len(ages.share)) / len(b)
    return {f"{prefix}agegap": float(0.5 * np.abs(q - ages.share).sum()),
            f"{prefix}new20": float(np.mean([x == 0 for x in b])),
            f"{prefix}old20": float(np.mean([x >= 2 for x in b]))}


def age_recall(top: list[int], liked: set[int], store, ref_year: int, prefix: str = "") -> dict:
    """recall@50 apart for the liked titles at most a year old at the cutoff
    (r50new, rank.AGE_EDGES' first bucket) and the older ones (r50old): what
    people rate next is mostly what just aired, so recall alone rewards a
    list for drifting to the new season."""
    def bucket(m: int) -> int | None:
        y = (store.rows.get(m) or {}).get("season_year")
        return None if y is None else rank.age_bucket(ref_year - y)
    out = {}
    for name, part in (("new", {x for x in liked if bucket(x) == 0}),
                       ("old", {x for x in liked if (bucket(x) or 0) >= 1})):
        if len(part) >= 2:
            out[f"{prefix}r50{name}"] = len(set(top[:50]) & part) / len(part)
    return out


def repeats(top: list[int], store, prefix: str = "") -> dict:
    """Share of the top 20 whose franchise already appeared higher up: 0
    with one title per franchise (rank.rerank); what a looser collapse
    lets through (results_grouplevel_2026-10-16.txt)."""
    if len(top) < 20:
        return {}
    fr = [store.rows[m].get("franchise_id") or m for m in top[:20]]
    return {f"{prefix}rep20": 1.0 - len(set(fr)) / len(fr)}


DMIX_EVERY = (2, 3, 5)


def interleave(main: list[int], extra: list[int], skip: set[int], every: int,
               limit: int = 60) -> list[int]:
    """`main` with every `every`-th slot taken by the next of `extra`; nothing
    from `skip`, nothing twice."""
    out: list[int] = []
    used = set(skip)
    a, b = iter(main), iter(extra)
    while len(out) < limit:
        src = (b, a) if (len(out) + 1) % every == 0 else (a, b)
        nxt = next((x for x in src[0] if x not in used), None)
        if nxt is None:
            nxt = next((x for x in src[1] if x not in used), None)
        if nxt is None:
            break
        out.append(nxt)
        used.add(nxt)
    return out


def list_metrics(top: list[int], liked: set[int], disliked: set[int], prefix: str) -> dict:
    """recall@50 (plain and IPS) / hit@10 / bad@10 of another tab's list."""
    out = {}
    if len(liked) >= 3:
        out[f"{prefix}r50"] = len(set(top[:50]) & liked) / len(liked)
        out[f"{prefix}ips50"] = ips50(top, liked)
    if len(top) >= 10:
        out[f"{prefix}hit10"] = len(set(top[:10]) & liked) / 10
        out[f"{prefix}bad10"] = len(set(top[:10]) & disliked) / 10
    return out


def user_metrics(scorer, top, items, truth, liked, disliked):
    pred = scorer.calibrate(scorer.predict_ids(items))
    ok = ~np.isnan(pred)
    out = {}
    if ok.sum() >= 5 and truth[ok].std() > 0:
        out["rho"] = float(spearman(pred[ok], truth[ok]))
        out["rmse"] = float(np.sqrt(np.mean((pred[ok] - truth[ok]) ** 2)))
    if len(liked) >= 3:
        out["r50"] = len(set(top[:50]) & liked) / len(liked)
        out["ips50"] = ips50(top, liked)
    if len(top) >= 10:
        out["hit10"] = len(set(top[:10]) & liked) / 10
        out["bad10"] = len(set(top[:10]) & disliked) / 10
    return out


def eligible(store, derived, min_scorers: int | None = None):
    """What eligible_candidates allows for a tab with `min_scorers` (default
    Safe Bets')."""
    min_scorers = SPECS["safe_bets"].min_scorers if min_scorers is None else min_scorers
    return [m for m, r in store.rows.items()
            if (r.get("mal_num_scoring_users") or 0) >= min_scorers
            and r.get("format_class") == "main" and r.get("nsfw") in (None, "white")
            and r.get("status") != "not_yet_aired" and m not in derived]


def row_of(store, m):
    r = store.rows[m]
    return {"mal_id": m, "title": r.get("title"), "franchise_id": r.get("franchise_id") or m,
            "mal_mean": r.get("mal_mean"), "mal_popularity": r.get("mal_popularity"),
            "known_franchise": False, "format_class": r.get("format_class")}


# -------------------------------------------------------------------- run --

def run(max_eval: int) -> dict:
    t0 = time.time()
    entries, static = load_snapshot()
    store, derived = static["store"], static["derived"]
    global GROUPS, SNAP_DAY
    GROUPS = groupings(store)
    SNAP_DAY = dt.datetime.fromisoformat(static["created"]).date()
    popularity_columns(store, SNAP_DAY)
    rng = np.random.default_rng(SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = population(entries, store, A, B)
    pool_all = eligible(store, derived)
    gems_all = eligible(store, derived, SPECS["hidden_gems"].min_scorers)
    # like the series and franchises, from the live relation table
    rel = frozen("relations", lambda: discovery_pool.relations(
        {m for m, r in store.rows.items() if r.get("format_class") == "main"},
        set(pool_all) | set(gems_all)))
    global PROP
    _, ma, _ = entries.columns(A)
    PROP = (np.bincount(ma.astype(np.int64)) + 1.0) / (len(A) + 2.0)
    RANKERS.clear()
    pools = {"safe_bets": (pool_all, None), "discover": (pool_all, pool_all),
             "hidden_gems": ([m for m in gems_all if gem(store.rows[m])], None)}
    for tab, setting in ltr.SURFACES.items():
        if getattr(CFG, setting) > 0:
            # as served (config.ranker_train_as_served): on the production pool,
            # Discover's queries without Safe Bets' learned list
            served = CFG.ranker_train_as_served
            RANKERS[tab], rhow = order_ranker(entries, store, pop, stacker, B, pools[tab][0],
                                              tab, pools[tab][1], rel if served else None,
                                              RANKERS.get("safe_bets")
                                              if tab == "discover" and served else None)
            how += f", {tab} order {rhow} (weight {getattr(CFG, setting):g})"
    print(f"snapshot {static['created'][:10]}: {len(users)} users ({len(A)} fit / {len(B)} stack /"
          f" {len(C)} eval), pool {len(pool_all)}, population model {how} [{time.time() - t0:.0f}s]",
          flush=True)
    out = {"snapshot": static["created"], "pop": code_hash(), "groups": {},
           "order_ranker_weight": CFG.order_ranker_weight,
           "settings": {k: getattr(CFG, k) for k in (
               "safe_bets_memory_mix", "safe_bets_key_risk", "order_ranker_ips",
               "discover_ranker_weight", "gems_ranker_weight", "age_calibration",
               "calibration_block", "series_weight_power", "gems_popularity",
               "gems_skip_safe_bets")}, "groupings": GROUPS}

    def genres(ids):
        return {g for m in ids for g in ((store.rows.get(m) or {}).get("mal_genres") or [])}

    def evaluate(hist, cutoff, listed, items, truth, liked, disliked, cal, full_hist):
        """Safe Bets, Discover and Hidden Gems for one user; their metrics and
        the Safe Bets top 20."""
        sc, im, li = examples(hist, cutoff)
        if len(sc) < 5:
            return None, None
        fam = genres(e["mal_id"] for e in hist)
        scorer = make_scorer(pop, stacker, store, sc, im, li, cal)
        long_model = None
        if CFG.safe_bets_memory_mix > 0:       # as surfaces.long_memory_model
            sc2, im2, li2 = ltr.history_examples(hist, cutoff, float("inf"))
            long_model = make_scorer(pop, stacker, store, sc2, im2, li2, cal)
        # what production keeps out for the whole list before the cutoff
        skip = listed | rel.exclusions(full_hist)
        rows = [row_of(store, m) for m in pool_all if m not in skip]
        _, sb = ranked(scorer, rows, li, fam, store, history=hist, long_model=long_model,
                       cutoff=cutoff)
        top = [c.mal_id for c in sb]
        _, disc = ranked(scorer, rows, li, fam, store, "discover", taken=set(top), history=hist,
                         cutoff=cutoff)
        gems_rows = [row_of(store, m) for m in gems_all if m not in skip]
        _, gems = ranked(scorer, gems_rows, li, fam, store, "hidden_gems", history=hist,
                         cutoff=cutoff, taken=set(top[:CFG.gems_skip_safe_bets]))
        # measured against the whole history before the cutoff, not the
        # budget's sample: the habit, not what the model happened to see
        ages = rank.age_mix(full_hist, year_of(store), cutoff.year)
        m = user_metrics(scorer, top, items, truth, liked, disliked)
        m |= age_metrics(top, ages) | repeats(top, store)
        m |= list_metrics([c.mal_id for c in disc], liked, disliked, "disc_")
        m |= age_metrics([c.mal_id for c in disc], ages, "disc_")
        m |= repeats([c.mal_id for c in disc], store, "disc_")
        m |= list_metrics([c.mal_id for c in gems], liked, disliked, "gems_")
        m |= age_metrics([c.mal_id for c in gems], ages, "gems_")
        m |= repeats([c.mal_id for c in gems], store, "gems_")
        # the lesser-known liked titles each list finds in its first 20, by
        # MAL's raw rank and by the age-fair one - fixed definitions, whatever
        # the pool rule
        tails = {"r": {x for x in liked if (store.rows[x].get("mal_popularity") or 99999)
                       > popularity.GEMS_FROM},
                 "g": {x for x in liked if GEM_RANKS.get(x, 99999) > popularity.GEMS_FROM}}
        gem_ids = [c.mal_id for c in gems]
        lists = {"": top, "disc_": [c.mal_id for c in disc], "gems_": gem_ids}
        # one Discover tab with every Nth slot a gem, in place of two tabs:
        # every arm measures each N, so it compares paired with plain Discover
        for n in DMIX_EVERY:
            lists[f"dmix{n}_"] = mixed = interleave(lists["disc_"], gem_ids, set(top), n)
            m |= list_metrics(mixed, liked, disliked, f"dmix{n}_")
            m |= age_metrics(mixed, ages, f"dmix{n}_")
        for prefix, ids in lists.items():
            for k, t in tails.items():
                if t:
                    m[f"{prefix}tl20{k}"] = len(set(ids[:20]) & t) / len(t)
        m["gems_sb20"] = len(set(gem_ids[:20]) & set(top[:20])) / 20
        for prefix in ("", "disc_", "gems_"):
            m |= age_recall(lists[prefix], liked, store, cutoff.year, prefix)
        return m, top[:20]

    for budget in BUDGETS:
        per, tops, pops = {}, [], []
        urng = np.random.default_rng(SEED + budget)
        for u in C:
            if len(per) >= max_eval:
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
            # the list at the cutoff: what was added later is still a candidate
            listed = {e["mal_id"] for e in history}
            liked = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            disliked = {m for m, s, _ in test_s if s < mean - 1}
            disliked |= {e["mal_id"] for e in es if e["status"] == "dropped" and e["at"] >= cutoff}
            m, top = evaluate(seen, cutoff, listed, [m for m, _, _ in test_s],
                              np.array([s for _, s, _ in test_s]), liked, disliked, (1.0, 0.0),
                              history)
            if m is None:
                continue
            per[str(u)] = m
            tops.append(set(top))
            pops.extend(store.rows[x].get("mal_popularity") or 99999 for x in top)
        pairs = list(itertools.islice(itertools.combinations(range(len(tops)), 2), 3000))
        out["groups"][f"sampled {budget}"] = {
            "users": per,
            "overlap": float(np.mean([len(tops[a] & tops[b]) / len(tops[a] | tops[b])
                                      for a, b in pairs])) if pairs else None,
            "pop": float(np.median(pops)) if pops else None}
        print(f"  sampled {budget}: {len(per)} users [{time.time() - t0:.0f}s]", flush=True)

    # the reference profile, 8 temporal splits
    ref = static["ref"]
    le = ref["le"]
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    per, tops, pops = {}, [], []
    for h in REF_SPLITS:
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        m, top = evaluate(hist, cutoff, {r["mal_id"] for r in hist},
                          [r["mal_id"] for r in test],
                          np.array([float(r["score"]) for r in test]),
                          {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)},
                          {r["mal_id"] for r in test if r["score"] < mean - 1}, ref["cal"], hist)
        if m is None:
            continue
        per[f"split {h}"] = m
        tops.append(set(top))
        pops.extend(store.rows[x].get("mal_popularity") or 99999 for x in top)
    out["groups"]["reference"] = {"users": per, "overlap": None, "pop": float(np.median(pops))}
    out["seconds"] = round(time.time() - t0)
    out["peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return out


# ---------------------------------------------------------------- compare --

def holm(pvalues: dict, alpha: float = ALPHA) -> set:
    """The keys whose null hypothesis Holm's step-down procedure rejects at
    family-wise error `alpha`."""
    out = set()
    for i, (k, pv) in enumerate(sorted(pvalues.items(), key=lambda kv: kv[1])):
        if pv > alpha / (len(pvalues) - i):
            break
        out.add(k)
    return out


def paired(gn: dict, go: dict, m: str) -> tuple[np.ndarray, float]:
    """The per-user changes of metric `m` between two runs' group, and the
    base run's mean on the same users."""
    common = [k for k, v in gn["users"].items() if m in v and m in go["users"].get(k, {})]
    d = np.array([gn["users"][k][m] - go["users"][k][m] for k in common])
    base = float(np.mean([go["users"][k][m] for k in common])) if common else float("nan")
    return d, base


def compare(new: dict, old: dict | None) -> bool:
    import math
    ok = True
    # the gate's family: every primary metric of every sampled group
    pvalues = {}
    for g, gn in new["groups"].items():
        go = (old or {}).get("groups", {}).get(g)
        if go is None or g == "reference":
            continue
        for m in PRIMARY:
            d, _ = paired(gn, go, m)
            if len(d) > 1 and d.std(ddof=1) > 0:
                z = abs(float(d.mean())) / float(d.std(ddof=1) / np.sqrt(len(d)))
                pvalues[(g, m)] = math.erfc(z / math.sqrt(2))
    decided = holm(pvalues)
    print()
    for g, gn in new["groups"].items():
        go = (old or {}).get("groups", {}).get(g)
        n = len(gn["users"])
        print(f"== {g} ({n} {'splits' if g == 'reference' else 'users'})")
        print(f"   {'metric':<8}{'now':>9}{'base':>9}{'change':>9}{'±2se':>8}  verdict")
        for m, higher in HIGHER.items():
            now = [v[m] for v in gn["users"].values() if m in v]
            if not now:
                continue
            line = f"   {m:<8}{np.mean(now):9.4f}"
            if go is None:
                print(line)
                continue
            d, base = paired(gn, go, m)
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
            mean_d = float(d.mean()) if len(d) else 0.0
            good = mean_d > 0 if higher else mean_d < 0
            if np.all(np.abs(d) < 1e-9):
                verdict = "same"
            elif g == "reference":
                win = int(((d > 1e-9) if higher else (d < -1e-9)).sum())
                lose = int(((d < -1e-9) if higher else (d > 1e-9)).sum())
                verdict = f"{win} better / {lose} worse splits"
                if m == "rho" and lose > win:
                    verdict += "  FAIL"
                    ok = False
            elif (g, m) in decided:
                verdict = "better" if good else "WORSE"
                ok = ok and good
            elif abs(mean_d) > 2 * se:
                verdict = "better?" if good else "worse?"
            else:
                verdict = "noise"
            if m not in PRIMARY and g != "reference":
                verdict += "  (described)"
            print(f"{line}{base:9.4f}{mean_d:+9.4f}{2 * se:8.4f}  {verdict}")
        for k, label in (("overlap", "overlap"), ("pop", "pop#")):
            if gn.get(k) is not None:
                b = f"{go[k]:9.3f}" if go and go.get(k) is not None else ""
                print(f"   {label:<8}{gn[k]:9.3f}{b}")
    if pvalues:
        print(f"\ngate: {len(pvalues)} primary comparisons, Holm at {ALPHA}: "
              f"{len(decided)} decided")
    if old and old.get("pop") != new.get("pop"):
        print("\nnote: the population code changed since the baseline (model refitted)")
    if old and old.get("snapshot") != new.get("snapshot"):
        print("\nnote: a different data snapshot than the baseline - compare with care")
    if old and old.get("order_ranker_weight", 0.0) != new.get("order_ranker_weight", 0.0):
        print(f"\nnote: learned order weight {new.get('order_ranker_weight', 0.0):g}, baseline "
              f"{old.get('order_ranker_weight', 0.0):g}")
    if old and old.get("settings") != new.get("settings"):
        print(f"\nnote: ranking settings {new.get('settings')}, baseline {old.get('settings')}")
    print(f"\n{'PASS' if ok else 'FAIL'}  [{new['seconds']}s, peak {new.get('peak_mb', '?')} MB]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="freeze the data (once)")
    ap.add_argument("--save-baseline", action="store_true")
    ap.add_argument("--max-eval", type=int, default=300)
    ap.add_argument("--freeze", action="store_true",
                    help="read the live tables as the first frozen run did (data/live.pkl)")
    ap.add_argument("--compare", nargs=2, metavar=("NEW", "BASE"),
                    help="compare two saved runs (latest.json, baseline.json) without running")
    a = ap.parse_args()
    if a.snapshot:
        snapshot()
        return 0
    if a.compare:
        return 0 if compare(json.loads(Path(a.compare[0]).read_text()),
                            json.loads(Path(a.compare[1]).read_text())) else 1
    if not (DATA / "entries.npz").exists():
        print("no snapshot yet: python -m experiments.bench --snapshot")
        return 2
    if a.freeze:
        global LIVE
        LIVE = pickle.loads(FROZEN.read_bytes()) if FROZEN.exists() else {}
        print(f"live tables frozen: {sorted(LIVE) or 'from this run'}", flush=True)
    new = run(a.max_eval)
    LATEST.write_text(json.dumps(new, indent=1))
    old = json.loads(BASELINE.read_text()) if BASELINE.exists() else None
    ok = compare(new, old)
    if a.save_baseline:
        BASELINE.write_text(json.dumps(new, indent=1))
        print(f"saved as the baseline ({BASELINE.name})")
        return 0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
