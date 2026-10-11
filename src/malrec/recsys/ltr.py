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
from .hybrid import Example, fit_user, recency_weight, split_user, unit_hash

if TYPE_CHECKING:
    from .cf import PopulationModel
    from .items import ItemStore

log = logging.getLogger(__name__)

BASE_FEATURES = ("shown", "rel_z", "pop_prior", "risk_long", "risk_disagree",
                 "risk_unfamiliar", "risk_acclaim", "pred_pop", "pred_personal", "pred_gap",
                 "personal_share", "log_n", "genre_fit", "mal_mean", "log_scorers",
                 "log_pop_rank", "year", "episodes", "novelty")
# what the user's own list says about long series (user_taste): with the
# title's length the ranker can learn who likes them, instead of one penalty
# for everyone
TASTE_FEATURES = ("user_long_share", "user_long_lift", "user_long_drop")
FEATURES = BASE_FEATURES + TASTE_FEATURES
LONG_PRIOR = 3.0      # pseudo-titles pulling a thin record toward "no opinion"
TRAIN_BUDGETS = (10, 25, 60, 150, 400)    # list sizes a training user is cut to
SHORTLIST = 600                            # the key's top this many are reordered
SEED = 3
LGBM: dict[str, Any] = {
    "objective": "lambdarank", "n_estimators": 600, "learning_rate": 0.03, "num_leaves": 15,
    "min_child_samples": 100, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_lambda": 5.0, "label_gain": [0, 1, 3], "lambdarank_truncation_level": 60,
    "verbose": -1}


# the settings the ranker's features depend on (the user model's fit and the
# hand-tuned key's terms): a ranker trained under others is not used
FINGERPRINT_SETTINGS = (
    "recency_half_life_years", "recency_floor", "use_affinity_features", "content_staff",
    "content_audience", "content_staff_aff", "stack_extras", "level_trend", "chosen_blend_min",
    "blend_size_n0", "blend_lam_max", "ranking_share_n0", "disagreement_penalty",
    "unfamiliar_genre_penalty", "long_series_penalty", "acclaim_penalty", "acclaim_evidence_e0",
    "safe_bets_popularity", "safe_bets_memory_mix", "safe_bets_memory_cap", "relevance_ease",
    "model_mode", "implicit_weight", "implicit_offsets", "not_interested_offset",
    "not_interested_weight", "not_interested_foldin", "relevance_weight",
    "relevance_weight_personal")


def fingerprint() -> str:
    import hashlib
    import json
    cfg = settings()
    d = {k: getattr(cfg, k) for k in FINGERPRINT_SETTINGS}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


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


def features(model, terms, order: list[int], ids: list[int], ctx_familiar, profile,
             n_scored: int, store: ItemStore, taste: dict[str, float] | None = None,
             names: tuple[str, ...] = FEATURES) -> np.ndarray:
    """The `names` features for the rows at `order` of a pool; `terms` is the
    pool's rank.OrderTerms (before any learned shift)."""
    from ..rank import novelty
    sid = [ids[k] for k in order]
    parts = model.risk_parts(sid, ctx_familiar)
    st, pe = model.um.predict_parts(sid) if model.um.inner is not None else (
        model.um.predict(sid), np.full(len(sid), np.nan))
    taste = taste or {}
    X = np.zeros((len(sid), len(names)), dtype=np.float32)
    for j, (k, m) in enumerate(zip(order, sid)):
        r = store.rows.get(m) or {}
        gs = r.get("mal_genres") or []
        has_pe = not math.isnan(pe[j])
        col = {
            "shown": terms.shown[k], "rel_z": terms.z[k], "pop_prior": terms.pop[k],
            "risk_long": parts["long"][j], "risk_disagree": parts["disagree"][j],
            "risk_unfamiliar": parts["unfamiliar"][j], "risk_acclaim": parts["acclaim"][j],
            "pred_pop": st[j], "pred_personal": pe[j] if has_pe else st[j],
            "pred_gap": (pe[j] - st[j]) if has_pe else 0.0, "personal_share": terms.lam,
            "log_n": math.log1p(n_scored),
            "genre_fit": sum(profile.get(g, 0.0) for g in gs) / max(len(gs), 1),
            "mal_mean": r.get("mal_mean") or 7.0,
            "log_scorers": math.log1p(r.get("mal_num_scoring_users") or 0),
            "log_pop_rank": math.log10(max(r.get("mal_popularity") or 99999, 1)),
            "year": r.get("season_year") or 2010, "episodes": r.get("num_episodes") or 0,
            "novelty": novelty(r.get("mal_popularity")), **taste}
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
                                    ctx.n_scored, store, ctx.taste, names))
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

def history_examples(seen: list[dict], cutoff) -> tuple[list[Example], list[Example],
                                                        dict[int, float]]:
    """A list's entries as a user model's input: ratings, implicit ratings
    from unscored statuses, and the recency-weighted listed titles."""
    cfg = settings()
    scored = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff), e["at"])
              for e in seen if e["score"] > 0]
    if len(scored) < 3:
        return scored, [], {}
    m = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
    implicit = [Example(e["mal_id"], float(min(max(m + cfg.implicit_offsets[e["status"]], 1), 10)),
                        cfg.implicit_weight * recency_weight(e["at"], cutoff), e["at"])
                for e in seen if e["score"] == 0 and e["status"] in cfg.implicit_offsets]
    return scored, implicit, {e["mal_id"]: recency_weight(e["at"], cutoff) for e in seen}


def genre_mix(listed: dict[int, float], store: ItemStore) -> dict[str, float]:
    """surfaces.genre_profile on in-memory data."""
    mix: dict[str, float] = {}
    for m, w in listed.items():
        gs = (store.rows.get(m) or {}).get("mal_genres") or []
        for g in gs:
            mix[g] = mix.get(g, 0.0) + w / len(gs)
    tot = sum(mix.values()) or 1.0
    return {g: v / tot for g, v in mix.items()}


def _labels(sid: list[int], test: dict[int, float], mean: float) -> np.ndarray:
    hi, ok = max(mean + 1.5, 9.0), max(mean, 7.0)
    return np.array([2 if test.get(m, 0) >= hi else 1 if test.get(m, 0) >= ok else 0
                     for m in sid], dtype=np.int32)


def train(pop: PopulationModel, stacker, store: ItemStore, pool: list[int],
          users: dict[int, list[dict]], seed: int = SEED,
          names: tuple[str, ...] | None = None) -> OrderRanker:
    """Fit the ranker. `users` are lists the population model never saw
    (entries with mal_id, score, status, at); `pool` is the Safe Bets
    eligible pool. Each user is split in time; their older entries, cut to
    two of TRAIN_BUDGETS, make the query, the newer ratings the labels."""
    import lightgbm as lgb

    from .. import rank
    from .scorer import HybridScorer
    from .service import blend_mode, blend_opts
    t0 = time.time()
    cfg = settings()
    names = names or FEATURES
    Xs, ys, groups, qusers = [], [], [], []
    trng = np.random.default_rng(seed + 11)
    for u in sorted(users):
        es = users[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        if len(scored_all) < 20:
            continue
        _, test_s, cutoff = split_user(scored_all, trng)
        mean = float(np.mean([s for _, s, _ in scored_all]))
        test_ids = {m for m, _, _ in test_s}
        history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
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
            terms = rank.order_terms(scorer, rows, pop_prior=cfg.safe_bets_popularity,
                                     familiar=fam)
            ids = [r["mal_id"] for r in rows]
            order = shortlist(terms)
            X = features(scorer, terms, order, ids, fam, genre_mix(li, store), len(sc), store,
                         user_taste(seen, store), names)
            y = _labels([ids[k] for k in order], {m: s for m, s, _ in test_s}, mean)
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
    ranker = lgb.LGBMRanker(**LGBM)
    ranker.fit(Xt, yt, group=gt, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10, 50],
               callbacks=[lgb.early_stopping(50, verbose=False)])
    gain = ranker.booster_.feature_importance("gain")
    tot = float(gain.sum()) or 1.0
    meta = {"users": len(users_q), "queries": len(groups), "candidates": len(y),
            "positives": int((y > 0).sum()), "trees": int(ranker.best_iteration_ or 0),
            "gain": {n: round(float(v) / tot, 4) for n, v in zip(names, gain)},
            "features": list(names), "settings": fingerprint(),
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
    from ..surfaces import SPECS
    n = SPECS["safe_bets"].min_scorers
    return [m for m, r in store.rows.items()
            if (r.get("mal_num_scoring_users") or 0) >= n
            and r.get("format_class") == "main" and r.get("nsfw") in (None, "white")
            and r.get("status") != "not_yet_aired" and m not in derived]


# ------------------------------------------------------------ database --

_cache: dict[str, Any] = {"id": None, "ranker": None}


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


def train_for_active(max_users: int = 2400) -> dict:
    """Train a ranker for the active population model and store it."""
    from psycopg.types.json import Jsonb

    from ..db import conn, one, returned
    from ..surfaces import derived_ids
    from .service import active_global, item_store
    row = one("SELECT id FROM global_model WHERE active")
    gm = active_global()
    if row is None or gm is None:
        raise RuntimeError("no active population model")
    store = item_store()
    users = stacker_lists(max_users)
    ranker = train(gm.pop, gm.stacker, store, safe_bets_pool(store, derived_ids()), users)
    with conn() as c, c.cursor() as cur:
        cur.execute("INSERT INTO order_ranker (global_model_id, meta, artifact)"
                    " VALUES (%s,%s,%s) RETURNING id",
                    (row["id"], Jsonb(ranker.meta), ranker.dumps()))
        rid = returned(cur)["id"]
        # the newest two per population model are plenty
        cur.execute("DELETE FROM order_ranker WHERE global_model_id=%s AND id NOT IN"
                    " (SELECT id FROM order_ranker WHERE global_model_id=%s"
                    "  ORDER BY id DESC LIMIT 2)", (row["id"], row["id"]))
        c.commit()
    log.info("order ranker %s for population model %s: %s", rid, row["id"], ranker.meta)
    return {"id": rid, "global_model_id": row["id"], **ranker.meta}


_CURRENT = """SELECT id FROM order_ranker WHERE global_model_id =
                 (SELECT id FROM global_model WHERE active) AND meta->>'settings' = %s
               ORDER BY id DESC LIMIT 1"""


def has_ranker() -> bool:
    """Whether the active population model has a ranker trained under the
    current ranking settings."""
    from ..db import one
    return one(_CURRENT, (fingerprint(),)) is not None


def active_ranker() -> OrderRanker | None:
    """The newest ranker of the active population model trained under the
    current ranking settings, if there is one."""
    from ..db import one, require_one
    row = one(_CURRENT, (fingerprint(),))
    if row is None:
        return None
    if _cache["id"] != row["id"]:
        blob = require_one("SELECT artifact FROM order_ranker WHERE id=%s", (row["id"],))
        _cache.update(id=row["id"], ranker=OrderRanker.loads(bytes(blob["artifact"])))
    return _cache["ranker"]


def context_for(user_id: int, model) -> RankContext | None:
    """The ranker context for a user's Safe Bets build, or None when the
    learned order is off, the model is not the hybrid one or the active
    population model has no ranker."""
    w = settings().order_ranker_weight
    from .scorer import HybridScorer
    if w <= 0 or not isinstance(model, HybridScorer):
        return None
    ranker = active_ranker()
    if ranker is None:
        return None
    from ..db import query
    from ..rank import familiar_genres
    from ..surfaces import genre_profile
    from .service import item_store
    entries = query("SELECT mal_id, score, status FROM list_entry WHERE user_id=%s", (user_id,))
    return RankContext(ranker, familiar_genres(user_id), genre_profile(user_id),
                       len(model._scores), w, user_taste(entries, item_store()))
