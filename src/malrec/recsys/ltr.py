"""A learned Safe Bets order (experiments/exp_ltr.py).

Safe Bets' order key (rank.order_terms) adds a dozen hand-weighted terms. A
LightGBM LambdaMART ranker learns how to combine them - and the raw signals
they are built from - across users: every (user, list size) is one query of
the key's top 600, labelled by what the user went on to rate. The served
key is a mix of the two orders, both put on the key's scale; with
`order_ranker_weight` 0.5 (the experiment's MIX) it beat the hand-tuned key
on recall@50 and hit@10 at every list size on the expanded sample
(2026-10-10), the reference profile's hit@10 +0.11.

The ranker belongs to one population model: its features are that model's
predictions, so it is trained on that model's stacker users (never in its
population) and used only while that model is active.
"""
from __future__ import annotations

import io
import logging
import math
import pickle
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import settings
from .hybrid import Example, fit_user, recency_weight, split_user, unit_hash, user_rng
from .series import weights as series_weights

if TYPE_CHECKING:
    from .cf import PopulationModel
    from .items import ItemStore
    from .pool import Relations

log = logging.getLogger(__name__)

BASE_FEATURES = ("shown", "rel_z", "pop_prior", "risk_long", "risk_disagree",
                 "risk_unfamiliar", "risk_acclaim", "pred_pop", "pred_personal", "pred_gap",
                 "personal_share", "log_n", "genre_fit", "mal_mean", "log_scorers",
                 "log_pop_rank", "year", "episodes", "novelty")
# what the user's own list says about long series (user_taste): with the
# title's length the ranker can learn who likes them, instead of one penalty
# for everyone
TASTE_FEATURES = ("user_long_share", "user_long_lift", "user_long_drop")
# The same for release age (rank.age_mix): the title's age when the list is
# built - not its year, which put this season's shows past anything seen in
# training, so the ranker read "newest" as "best" - and how much of the
# user's watching, and how well rated, was of titles that old
AGE_FEATURES = ("age", "user_age_share", "user_age_lift")
FEATURES = (tuple(f for f in BASE_FEATURES if f != "year") + TASTE_FEATURES
            + AGE_FEATURES)
LONG_PRIOR = 3.0      # pseudo-titles pulling a thin record toward "no opinion"
TRAIN_BUDGETS = (10, 25, 60, 150, 400)    # list sizes a training user is cut to
SHORTLIST = 600                            # the key's top this many are reordered
LIST_LIMIT = 60                            # a built tab's length (surfaces.build_all)
SEED = 3
LGBM: dict[str, Any] = {
    "objective": "lambdarank", "n_estimators": 600, "learning_rate": 0.03, "num_leaves": 15,
    "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_lambda": 5.0, "label_gain": [0, 1, 3], "lambdarank_truncation_level": 60,
    "verbose": -1}
# Popularity-corrected labels (order_ranker_ips > 0): a liked title counts
# (propensity / median) ** -ips, clipped to IPS_CLIP, so the ranker learns
# what is liked more than what is watched. Gains are carried as integer labels
# in IPS_STEPS-ths of a point (LightGBM's lambdarank takes no per-row weight).
IPS_CLIP = (0.25, 4.0)
IPS_STEPS = 4
IPS_MAX_LABEL = 48
# the tabs that can carry a learned order, with the setting that turns each on
SURFACES = {"safe_bets": "order_ranker_weight", "discover": "discover_ranker_weight",
            "hidden_gems": "gems_ranker_weight"}


# the settings the ranker's features depend on (the user model's fit and the
# hand-tuned key's terms): a ranker trained under others is not used. Not the
# long-memory mix: training orders without that model (it moves the key by
# at most safe_bets_memory_cap)
FINGERPRINT_SETTINGS = (
    "recency_half_life_years", "recency_floor", "use_affinity_features", "content_staff",
    "content_audience", "content_staff_aff", "stack_extras", "level_trend", "chosen_blend_min",
    "blend_size_n0", "blend_lam_max", "ranking_share_n0", "disagreement_penalty",
    "unfamiliar_genre_penalty", "long_series_penalty", "acclaim_penalty", "acclaim_evidence_e0",
    "safe_bets_popularity", "relevance_ease",
    "model_mode", "implicit_weight", "implicit_offsets", "not_interested_offset",
    "not_interested_weight", "not_interested_foldin", "relevance_weight",
    "relevance_weight_personal", "safe_bets_key_risk", "order_ranker_ips",
    "series_weight_power", "blend_lam_min", "blend_holdout", "blend_criterion",
    "disagreement_scale")
# Settings added after rankers were stored, with the value that leaves the
# key as it was: at that value they stay out of the fingerprint, so adding one
# does not retire every stored ranker.
FINGERPRINT_NEUTRAL = {"series_weight_power": 0.0, "blend_lam_min": 0.0,
                       "blend_holdout": "newest", "blend_criterion": "mse",
                       "disagreement_scale": "raw"}


def fingerprint() -> str:
    import hashlib
    import json
    cfg = settings()
    d = {k: getattr(cfg, k) for k in FINGERPRINT_SETTINGS
         if k not in FINGERPRINT_NEUTRAL or getattr(cfg, k) != FINGERPRINT_NEUTRAL[k]}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def surface_key(surface: str) -> dict:
    """How a tab's hand-built key is formed (shared by training and serving):
    the popularity prior and the share of the risk term."""
    cfg = settings()
    if surface == "safe_bets":
        return {"pop_prior": cfg.safe_bets_popularity, "risk_scale": cfg.safe_bets_key_risk}
    return {"pop_prior": 0.0, "risk_scale": 1.0}


def _z(v: np.ndarray) -> np.ndarray:
    sd = float(np.std(v))
    return (v - np.mean(v)) / sd if sd > 1e-9 else np.zeros_like(v)


@dataclass
class OrderRanker:
    model_str: str                 # LightGBM's text format
    meta: dict
    _booster: Any = field(default=None, repr=False, compare=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._booster is None:
            import lightgbm as lgb
            self._booster = lgb.Booster(model_str=self.model_str)
        return np.asarray(self._booster.predict(X), dtype=float)

    def dumps(self) -> bytes:
        buf = io.BytesIO()
        pickle.dump({"model_str": self.model_str, "meta": self.meta}, buf)
        return buf.getvalue()

    @staticmethod
    def loads(blob: bytes) -> OrderRanker:
        d = pickle.loads(blob)
        return OrderRanker(d["model_str"], d["meta"])


@dataclass
class RankContext:
    """What the ranker needs about one user besides their model."""
    ranker: OrderRanker
    familiar: set[str] | None      # every genre on their list (rank.familiar_genres)
    profile: dict[str, float]      # their genre mix (surfaces.genre_profile)
    n_scored: int                  # ratings their model was fitted on
    weight: float                  # share of the learned order in the key
    taste: dict[str, float] = field(default_factory=dict)   # user_taste()
    ages: AgeTaste | None = None                             # age_taste()


@dataclass
class AgeTaste:
    """What a list says about release age (AGE_FEATURES): per rank.AGE_EDGES
    bucket the share of started titles watched at that age and their rating
    lift over the user's mean, pulled toward 0 by LONG_PRIOR pseudo-titles;
    `ref_year` is the year candidates are aged from."""
    share: np.ndarray
    lift: np.ndarray
    ref_year: int


# ------------------------------------------------------------- features --

def user_taste(entries: list[dict], store: ItemStore) -> dict[str, float]:
    """TASTE_FEATURES from a list's entries (mal_id, score, status), over
    the titles the user actually started (plan-to-watch says nothing about
    what they watch): the share of long series among them - 0 for someone
    who does not watch them at all -, how much better or worse than their
    own mean they rate them, and the share of them dropped; the last two
    pulled toward 0 by LONG_PRIOR pseudo-titles, so two ratings say little."""
    from .scorer import is_long_series
    started = [e for e in entries if e["status"] != "plan_to_watch"]
    if not started:
        return dict.fromkeys(TASTE_FEATURES, 0.0)
    long_ = [e for e in started if is_long_series(store.rows.get(e["mal_id"]) or {})]
    scores = [float(e["score"]) for e in started if e["score"] > 0]
    mean = float(np.mean(scores)) if scores else 0.0
    lift = sum(float(e["score"]) - mean for e in long_ if e["score"] > 0)
    n_rated = sum(1 for e in long_ if e["score"] > 0)
    return {"user_long_share": len(long_) / len(started),
            "user_long_lift": lift / (n_rated + LONG_PRIOR),
            "user_long_drop": sum(e["status"] == "dropped" for e in long_)
                              / (len(long_) + LONG_PRIOR)}


def age_taste(entries: list[dict], store: ItemStore, ref_year: int) -> AgeTaste:
    """AgeTaste from a list's entries (mal_id, score, status, at)."""
    from ..rank import AGE_EDGES, age_bucket
    k = len(AGE_EDGES) + 1
    n, lift, rated = np.zeros(k), np.zeros(k), np.zeros(k)
    scores = [float(e["score"]) for e in entries if e["score"] > 0]
    mean = float(np.mean(scores)) if scores else 0.0
    for e in entries:
        y = (store.rows.get(e["mal_id"]) or {}).get("season_year")
        if e["status"] == "plan_to_watch" or y is None or e.get("at") is None:
            continue
        b = age_bucket(e["at"].year - y)
        n[b] += 1
        if e["score"] > 0:
            lift[b] += float(e["score"]) - mean
            rated[b] += 1
    return AgeTaste(n / n.sum() if n.sum() else n, lift / (rated + LONG_PRIOR), ref_year)


def features(model, terms, order: list[int], ids: list[int], ctx_familiar, profile,
             n_scored: int, store: ItemStore, taste: dict[str, float] | None = None,
             names: tuple[str, ...] = FEATURES, ages: AgeTaste | None = None) -> np.ndarray:
    """The `names` features for the rows at `order` of a pool; `terms` is the
    pool's rank.OrderTerms (before any learned shift).

    The ranker learns from users scored with the identity calibration (1, 0).
    A served account has its own line, which would move the calibrated score,
    the risk parts scaled by its slope and the popularity prior (in standard
    deviations of the key) off the scale the ranker was trained on (audit
    2026-10-23, P1-1); they are put back on that scale here, so training and
    serving see the same numbers for the same predictions."""
    from ..rank import novelty
    sid = [ids[k] for k in order]
    parts = model.risk_parts(sid, ctx_familiar)
    a = float(model.calibration[0]) if getattr(model, "calibration", None) else 1.0
    a = a if a > 1e-6 else 1.0
    shown = np.clip(terms.raw, 1.0, 10.0)
    ok = terms.ok
    sd = float(np.std((terms.shown + terms.rel)[ok])) if ok.any() else 0.0
    pop_scale = float(np.std((shown + terms.rel)[ok])) / sd if sd > 1e-9 else 1.0
    st, pe = model.um.predict_parts(sid) if model.um.inner is not None else (
        model.um.predict(sid), np.full(len(sid), np.nan))
    taste = taste or {}
    from ..rank import age_bucket
    X = np.zeros((len(sid), len(names)), dtype=np.float32)
    for j, (k, m) in enumerate(zip(order, sid)):
        r = store.rows.get(m) or {}
        gs = r.get("mal_genres") or []
        has_pe = not math.isnan(pe[j])
        col = {
            "shown": shown[k], "rel_z": terms.z[k], "pop_prior": terms.pop[k] * pop_scale,
            "risk_long": parts["long"][j], "risk_disagree": parts["disagree"][j] / a,
            "risk_unfamiliar": parts["unfamiliar"][j], "risk_acclaim": parts["acclaim"][j] / a,
            "pred_pop": st[j], "pred_personal": pe[j] if has_pe else st[j],
            "pred_gap": (pe[j] - st[j]) if has_pe else 0.0, "personal_share": terms.lam,
            "log_n": math.log1p(n_scored),
            "genre_fit": sum(profile.get(g, 0.0) for g in gs) / max(len(gs), 1),
            "mal_mean": r.get("mal_mean") or 7.0,
            "log_scorers": math.log1p(r.get("mal_num_scoring_users") or 0),
            "log_pop_rank": math.log10(max(r.get("mal_popularity") or 99999, 1)),
            "year": r.get("season_year") or 2010, "episodes": r.get("num_episodes") or 0,
            "novelty": novelty(r.get("mal_popularity")), **taste}
        y = r.get("season_year")
        if ages is not None:
            age = ages.ref_year - y if y is not None else math.nan
            b = age_bucket(age) if y is not None else None
            col["age"] = age
            col["user_age_share"] = ages.share[b] if b is not None else 0.0
            col["user_age_lift"] = ages.lift[b] if b is not None else 0.0
        X[j] = [col.get(n, 0.0) for n in names]
    return X


def shortlist(terms) -> list[int]:
    return [k for k in np.argsort(-terms.key) if terms.ok[k]][:SHORTLIST]


def learned_shift(model, terms, ids: list[int], ctx: RankContext, store: ItemStore) -> np.ndarray:
    """What to add to the key so it orders by the mix: the shortlist by
    mean + sd * ((1 - w) z(key) + w z(ranker)), on the key's own scale, and
    everything below the shortlist after it, in its old order."""
    shift = np.zeros(len(ids))
    order = shortlist(terms)
    if len(order) < 3:
        return shift
    key = terms.key[order]
    names = tuple(ctx.ranker.meta.get("features") or BASE_FEATURES)
    s = ctx.ranker.predict(features(model, terms, order, ids, ctx.familiar, ctx.profile,
                                    ctx.n_scored, store, ctx.taste, names, ctx.ages))
    w = ctx.weight
    mixed = float(np.mean(key)) + float(np.std(key)) * ((1 - w) * _z(key) + w * _z(s))
    shift[order] = mixed - key
    rest = np.zeros(len(ids), dtype=bool)
    rest[terms.ok] = True
    rest[order] = False
    if rest.any():
        gap = float(mixed.min()) - 1e-3 - float(terms.key[rest].max())
        if gap < 0:
            shift[rest] = gap
    return shift


# ------------------------------------------------------------- training --

def history_examples(seen: list[dict], cutoff, half_life: float | None = None
                     ) -> tuple[list[Example], list[Example], dict[int, float]]:
    """A list's entries as a user model's input: ratings, implicit ratings
    from unscored statuses, and the recency-weighted listed titles
    (`half_life` inf: every entry weighted equally, the long-memory model)."""
    cfg = settings()

    share = series_weights(e["mal_id"] for e in seen)

    def w(e):
        return recency_weight(e["at"], cutoff, half_life) * share.get(e["mal_id"], 1.0)
    scored = [Example(e["mal_id"], float(e["score"]), w(e), e["at"])
              for e in seen if e["score"] > 0]
    if len(scored) < 3:
        return scored, [], {}
    m = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
    implicit = [Example(e["mal_id"], float(min(max(m + cfg.implicit_offsets[e["status"]], 1), 10)),
                        cfg.implicit_weight * w(e), e["at"])
                for e in seen if e["score"] == 0 and e["status"] in cfg.implicit_offsets]
    return scored, implicit, {e["mal_id"]: w(e) for e in seen}


def genre_mix(listed: dict[int, float], store: ItemStore) -> dict[str, float]:
    """surfaces.genre_profile on in-memory data."""
    mix: dict[str, float] = {}
    for m, w in listed.items():
        gs = (store.rows.get(m) or {}).get("mal_genres") or []
        for g in gs:
            mix[g] = mix.get(g, 0.0) + w / len(gs)
    tot = sum(mix.values()) or 1.0
    return {g: v / tot for g, v in mix.items()}


def _labels(sid: list[int], test: dict[int, float], mean: float,
            ipw: dict[int, float] | None = None) -> np.ndarray:
    """0 / 1 (rated at least their mean, 7+) / 2 (mean + 1.5, 9+); with
    `ipw` (title -> inverse-propensity weight) the gain (1 or 3) times the
    weight, in IPS_STEPS-ths, as an integer label for an identity gain table."""
    hi, ok = max(mean + 1.5, 9.0), max(mean, 7.0)
    lv = [2 if test.get(m, 0) >= hi else 1 if test.get(m, 0) >= ok else 0 for m in sid]
    if ipw is None:
        return np.array(lv, dtype=np.int32)
    gain = (0, 1, 3)
    return np.array([0 if v == 0 else int(np.clip(round(gain[v] * ipw.get(m, 1.0) * IPS_STEPS),
                                                   1, IPS_MAX_LABEL))
                     for v, m in zip(lv, sid)], dtype=np.int32)


def inverse_propensity(pop: PopulationModel, pool: list[int], power: float) -> dict[int, float]:
    """(propensity / median) ** -power per pool title, clipped to IPS_CLIP;
    propensity = the share of the population's raters who rated it."""
    n = np.array([pop.n_raters[pop.index[m]] if m in pop.index else 0 for m in pool], dtype=float)
    p = (n + 1.0) / (n.max() + 2.0)
    w = np.clip((p / np.median(p)) ** -power, *IPS_CLIP)
    return dict(zip(pool, w.tolist()))


def train(pop: PopulationModel, stacker, store: ItemStore, pool: list[int],
          users: dict[int, list[dict]], seed: int = SEED,
          names: tuple[str, ...] | None = None, surface: str = "safe_bets",
          taken_pool: list[int] | None = None, relations: Relations | None = None,
          taken_ranker: OrderRanker | None = None) -> OrderRanker:
    """Fit the ranker. `users` are lists the population model never saw
    (entries with mal_id, score, status, at); `pool` is the tab's eligible
    pool. Each user is split in time; their older entries, cut to two of
    TRAIN_BUDGETS, make the query, the newer ratings the labels.

    A query is the pool less the user's whole list but the test titles.
    Discover starts where Safe Bets ends: with `taken_pool` (Safe Bets'
    pool) the hand-built Safe Bets top LIST_LIMIT leaves each Discover query.

    With `relations` (config.ranker_train_as_served) the query is what
    production offers at the cutoff instead: the pool less the list as it
    stood then - entries added later stay in, a later drop is a negative like
    any unrated title - and less what that list keeps out of the tab
    (recsys/pool.py: prerequisites, Up Next's continuations); Discover's
    Safe Bets top is ordered as served, with `taken_ranker` mixed in at
    order_ranker_weight; and each user draws from a random stream of their
    own. Measured no better (2026-10-24), so not the default."""
    import lightgbm as lgb

    from .. import rank
    from .scorer import HybridScorer
    from .service import blend_mode, blend_opts
    t0 = time.time()
    cfg = settings()
    names = names or FEATURES
    key = surface_key(surface)
    ipw = inverse_propensity(pop, pool, cfg.order_ranker_ips) if cfg.order_ranker_ips > 0 else None
    Xs, ys, groups, qusers = [], [], [], []
    shared_rng = np.random.default_rng(seed + 11)
    for u in sorted(users):
        es = users[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        if len(scored_all) < 20:
            continue
        # as served: a stream of the user's own, so their split and budgets
        # do not move when other users join or leave the training set
        trng = user_rng("ltr", seed, u) if relations is not None else shared_rng
        _, test_s, cutoff = split_user(scored_all, trng)
        mean = float(np.mean([s for _, s, _ in scored_all]))
        test_ids = {m for m, _, _ in test_s}
        history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
        if relations is not None:
            listed = {e["mal_id"] for e in history} | relations.exclusions(history)
        else:
            listed = {e["mal_id"] for e in es} - test_ids
        rows = [_row(store, m) for m in pool if m not in listed]
        for budget in trng.choice(TRAIN_BUDGETS, 2, replace=False):
            if len(history) < budget:
                continue
            seen = [history[i] for i in trng.permutation(len(history))[:budget]]
            sc, im, li = history_examples(seen, cutoff)
            if len(sc) < 5:
                continue
            um = fit_user(pop, stacker, store, sc, im, mode=blend_mode(len(sc)),
                          blend_of="personal", **blend_opts(), listed=li)
            scorer = HybridScorer(um, cfg.model_mode, {}, {}, (1.0, 0.0),
                                  {e.mal_id: e.score for e in sc}, {})
            fam = {g for e in seen for g in ((store.rows.get(e["mal_id"]) or {})
                                             .get("mal_genres") or [])}
            q_rows = rows
            taste, ages = user_taste(seen, store), age_taste(seen, store, cutoff.year)
            if taken_pool is not None:
                sb_rows = [_row(store, m) for m in taken_pool if m not in listed]
                sb_ctx = (RankContext(taken_ranker, fam, genre_mix(li, store), len(sc),
                                      cfg.order_ranker_weight, taste, ages)
                          if relations is not None and taken_ranker is not None
                          and cfg.order_ranker_weight > 0 else None)
                sb = rank.order_terms(scorer, sb_rows, familiar=fam, learned=sb_ctx,
                                      **surface_key("safe_bets"))
                gone = {sb_rows[k]["mal_id"] for k in np.argsort(-sb.key)[:LIST_LIMIT]}
                q_rows = [r for r in rows if r["mal_id"] not in gone]
            terms = rank.order_terms(scorer, q_rows, familiar=fam, **key)
            ids = [r["mal_id"] for r in q_rows]
            order = shortlist(terms)
            X = features(scorer, terms, order, ids, fam, genre_mix(li, store), len(sc), store,
                         taste, names, ages)
            y = _labels([ids[k] for k in order], {m: s for m, s, _ in test_s}, mean, ipw)
            if y.sum() == 0:
                continue                    # a query with no positive teaches nothing
            Xs.append(X); ys.append(y); groups.append(len(y)); qusers.append(u)
    X, y = np.vstack(Xs), np.concatenate(ys)
    log.info("order ranker: %d queries, %d candidates, %d positives [%.0fs]",
             len(groups), len(y), int((y > 0).sum()), time.time() - t0)
    # a fifth of the users for early stopping
    users_q = sorted(set(qusers))
    val_u = set(users_q[::5])
    tr = [k for k, u in enumerate(qusers) if u not in val_u]
    va = [k for k, u in enumerate(qusers) if u in val_u]
    ofs = np.cumsum([0] + groups)

    def take(idx):
        return (np.vstack([X[ofs[k]:ofs[k + 1]] for k in idx]),
                np.concatenate([y[ofs[k]:ofs[k + 1]] for k in idx]), [groups[k] for k in idx])
    Xt, yt, gt = take(tr)
    Xv, yv, gv = take(va)
    params = dict(LGBM)
    if ipw is not None:
        params["label_gain"] = [k / IPS_STEPS for k in range(IPS_MAX_LABEL + 1)]
    ranker = lgb.LGBMRanker(**params)
    ranker.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10, 50],
               callbacks=[lgb.early_stopping(50, verbose=False)])
    gain = ranker.booster_.feature_importance("gain")
    tot = float(gain.sum()) or 1.0
    meta = {"users": len(users_q), "queries": len(groups), "candidates": len(y),
            "positives": int((y > 0).sum()), "trees": int(ranker.best_iteration_ or 0),
            "gain": {n: round(float(v) / tot, 4) for n, v in zip(names, gain)},
            "features": list(names), "settings": fingerprint(), "surface": surface,
            "ips": cfg.order_ranker_ips, "pool": "production" if relations else "eligible",
            "seconds": round(time.time() - t0, 1)}
    return OrderRanker(ranker.booster_.model_to_string(num_iteration=ranker.best_iteration_),
                       meta)


def _row(store: ItemStore, m: int) -> dict:
    r = store.rows[m]
    return {"mal_id": m, "title": r.get("title"), "franchise_id": r.get("franchise_id") or m,
            "mal_mean": r.get("mal_mean"), "mal_popularity": r.get("mal_popularity"),
            "known_franchise": False, "format_class": r.get("format_class")}


def safe_bets_pool(store: ItemStore, derived: frozenset[int] | set[int]) -> list[int]:
    """What eligible_candidates allows for Safe Bets, from the item store."""
    return surface_pool(store, derived, "safe_bets")


# ------------------------------------------------------------ database --

_cache: dict[str, tuple[int, OrderRanker]] = {}      # surface -> (id, ranker)
_missing_warned: set[str] = set()                    # tabs context_for has warned about


def stacker_lists(max_users: int = 2400) -> dict[int, list[dict]]:
    """The active population model's stacker users (the role hash of
    service.fit_global) with whole lists, at most `max_users`."""
    from ..db import query
    ids = [r["id"] for r in query(
        "SELECT id FROM cf_user WHERE state='done' AND n_scored >= 20 ORDER BY id")]
    ids = [u for u in ids if unit_hash("role", 0, u) < 0.15][:max_users]
    out: dict[int, list[dict]] = {u: [] for u in ids}
    for r in query("""
            SELECT r.user_id, r.mal_id, r.score, r.status,
                   coalesce(r.finished_at::timestamptz, r.updated_at) AS at
              FROM cf_rating r WHERE r.user_id = ANY(%s) AND r.updated_at IS NOT NULL
             ORDER BY r.user_id, 5, r.mal_id""", (ids,)):
        out[r["user_id"]].append({"mal_id": r["mal_id"], "score": r["score"],
                                  "status": r["status"], "at": r["at"]})
    return out


def surface_pool(store: ItemStore, derived: frozenset[int] | set[int], surface: str) -> list[int]:
    """What eligible_candidates allows for `surface`, from the item store
    (Hidden Gems: in its pool as well, recsys/popularity.py)."""
    from ..surfaces import SPECS
    n = SPECS[surface].min_scorers
    out = [m for m, r in store.rows.items()
           if (r.get("mal_num_scoring_users") or 0) >= n
           and r.get("format_class") == "main" and r.get("nsfw") in (None, "white")
           and r.get("status") != "not_yet_aired" and m not in derived]
    if surface == "hidden_gems":
        from ..config import settings
        from .popularity import gem_ids
        gems = gem_ids(settings().gems_popularity)
        out = [m for m in out if m in gems]
    return out


def train_for_active(max_users: int = 2400, surface: str = "safe_bets") -> dict:
    """Train a ranker for the active population model's `surface` and store it."""
    from psycopg.types.json import Jsonb

    from ..db import conn, one, returned
    from ..surfaces import derived_ids
    from .pool import relations
    from .service import active_global, item_store
    row = one("SELECT id FROM global_model WHERE active")
    gm = active_global()
    if row is None or gm is None:
        raise RuntimeError("no active population model")
    store = item_store()
    users = stacker_lists(max_users)
    derived = derived_ids()
    tab_pool = surface_pool(store, derived, surface)
    taken = safe_bets_pool(store, derived) if surface == "discover" else None
    rel = (relations({m for m, r in store.rows.items() if r.get("format_class") == "main"},
                     set(tab_pool) | set(taken or ()))
           if settings().ranker_train_as_served else None)
    ranker = train(gm.pop, gm.stacker, store, tab_pool, users, surface=surface,
                   taken_pool=taken, relations=rel,
                   taken_ranker=active_ranker("safe_bets")
                   if taken is not None and rel is not None else None)
    with conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO order_ranker (global_model_id, surface, meta, artifact)"
                    " VALUES (%s,%s,%s,%s) RETURNING id",
                    (row["id"], surface, Jsonb(ranker.meta), ranker.dumps()))
        rid = returned(cur)["id"]
        # the newest two per population model and tab are plenty
        cur.execute("DELETE FROM order_ranker WHERE global_model_id=%s AND surface=%s"
                    " AND id NOT IN (SELECT id FROM order_ranker WHERE global_model_id=%s"
                    "  AND surface=%s ORDER BY id DESC LIMIT 2)",
                    (row["id"], surface, row["id"], surface))
        c.commit()
    log.info("order ranker %s (%s) for population model %s: %s", rid, surface, row["id"],
             ranker.meta)
    return {"id": rid, "global_model_id": row["id"], **ranker.meta}


_CURRENT = """SELECT id FROM order_ranker WHERE global_model_id =
                 (SELECT id FROM global_model WHERE active) AND meta->>'settings' = %s
                 AND surface = %s
               ORDER BY id DESC LIMIT 1"""


def enabled_surfaces() -> list[str]:
    """The tabs whose learned order is switched on."""
    cfg = settings()
    return [s for s, k in SURFACES.items() if getattr(cfg, k) > 0]


def has_ranker(surface: str = "safe_bets") -> bool:
    """Whether the active population model has a ranker for `surface`
    trained under the current ranking settings."""
    from ..db import one
    return one(_CURRENT, (fingerprint(), surface)) is not None


def active_ranker(surface: str = "safe_bets") -> OrderRanker | None:
    """The newest ranker of the active population model for `surface`,
    trained under the current ranking settings, if there is one."""
    from ..db import one, require_one
    row = one(_CURRENT, (fingerprint(), surface))
    if row is None:
        return None
    if _cache.get(surface, (None, None))[0] != row["id"]:
        blob = require_one("SELECT artifact FROM order_ranker WHERE id=%s", (row["id"],))
        _cache[surface] = (row["id"], OrderRanker.loads(bytes(blob["artifact"])))
    return _cache[surface][1]


def context_for(user_id: int, model, surface: str = "safe_bets") -> RankContext | None:
    """The ranker context for a user's build of `surface`, or None when that
    tab has no learned order (SURFACES), it is off, the model is not the
    hybrid one or the active population model has no ranker for it."""
    if surface not in SURFACES:
        return None
    w = getattr(settings(), SURFACES[surface])
    from .scorer import HybridScorer
    if w <= 0 or not isinstance(model, HybridScorer):
        return None
    ranker = active_ranker(surface)
    if ranker is None:
        if surface not in _missing_warned:
            # switched on but nothing to serve: the tab silently falls back to
            # its hand-built order until `malrec ranker train` has run (e.g.
            # after a setting in FINGERPRINT_SETTINGS changed)
            _missing_warned.add(surface)
            log.warning("%s: learned order switched on, but the active population model has"
                        " no ranker for the current settings (%s); hand-built order until"
                        " `malrec ranker train`", surface, fingerprint())
        return None
    import datetime as dt

    from ..db import query
    from ..rank import familiar_genres
    from ..surfaces import genre_profile
    from .service import item_store
    entries = query("SELECT mal_id, score, status,"
                    " coalesce(finished_at::timestamptz, updated_at) AS at"
                    " FROM list_entry WHERE user_id=%s", (user_id,))
    store = item_store()
    return RankContext(ranker, familiar_genres(user_id), genre_profile(user_id),
                       len(model._scores), w, user_taste(entries, store),
                       age_taste(entries, store, dt.datetime.now(dt.UTC).year))
