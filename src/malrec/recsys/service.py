"""Production entry points for the hybrid model.

    fit_global()      fit the population model + stacker from the CF sample
                      and store it as a new active version (`malrec population fit`)
    fit_for_user(id)  fold an app user in and fit their personal part

The population model and the item store are cached per process: both are
read-mostly and take seconds to load, whereas a user model takes well under
a second to fit, so user models are refitted on demand rather than stored.
"""
from __future__ import annotations

import datetime as dt
import itertools
import logging
import threading
import time
from collections import defaultdict

import numpy as np
from psycopg.types.json import Jsonb

from ..config import settings
from ..db import conn, execute, one, query
from .cf import CFParams, PopulationModel
from .hybrid import (
    STACK_FEATURES,
    Example,
    GlobalModel,
    UserModel,
    fit_stacker,
    fit_user,
    recency_weight,
    split_user,
    stacker_extras,
    unit_hash,
    user_rng,
)
from .items import ItemStore

log = logging.getLogger(__name__)

STORE_TTL = 600.0
_lock = threading.Lock()
_cache: dict = {"global": None, "global_id": None, "store": None, "store_at": 0.0}


# ------------------------------------------------------------------- data --

def load_cf_ratings(min_scored: int = 5) -> dict[int, list[tuple]]:
    """(mal_id, score, at) per sampled user."""
    rows = query("""
        SELECT r.user_id, r.mal_id, r.score,
               coalesce(r.finished_at::timestamptz, r.updated_at) AS at
          FROM cf_rating r JOIN cf_user u ON u.id = r.user_id
         WHERE u.state = 'done' AND u.n_scored >= %s AND r.score > 0
         ORDER BY r.user_id, at, r.mal_id
    """, (min_scored,))
    epoch = dt.datetime(2000, 1, 1, tzinfo=dt.UTC)
    out: dict[int, list[tuple]] = defaultdict(list)
    for r in rows:
        out[r["user_id"]].append((r["mal_id"], float(r["score"]), r["at"] or epoch))
    return out


def load_cf_entries() -> tuple[np.ndarray, np.ndarray]:
    """Every list entry of every sampled user, any status - for co-occurrence."""
    rows = query("""SELECT r.user_id, r.mal_id FROM cf_rating r
                      JOIN cf_user u ON u.id = r.user_id WHERE u.state = 'done'""")
    return (np.array([r["user_id"] for r in rows], dtype=np.int64),
            np.array([r["mal_id"] for r in rows], dtype=np.int64))


def user_listed(user_id: int, cutoff: dt.datetime | None = None,
                half_life: float | None = None,
                exclude: frozenset[int] = frozenset()) -> dict[int, float]:
    """Every entry on the user's list (any status) with its recency weight -
    the input to the co-occurrence relevance signal."""
    now = cutoff or dt.datetime.now(dt.UTC)
    rows = query("""SELECT mal_id, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                      FROM list_entry WHERE user_id = %s""", (user_id,))
    return {r["mal_id"]: recency_weight(r["at"], now, half_life)
            for r in rows if (cutoff is None or r["at"] < cutoff) and r["mal_id"] not in exclude}


def user_examples(user_id: int, cutoff: dt.datetime | None = None,
                  half_life: float | None = None, exclude: frozenset[int] = frozenset()
                  ) -> tuple[list[Example], list[Example]]:
    """An app user's real ratings and implicit pseudo-ratings, recency-weighted
    from `cutoff` (default: now). Anything dated at or after the cutoff is
    left out, which is what the temporal holdout relies on; so is anything in
    `exclude` (the prospective test's titles, whatever their dates)."""
    cfg = settings()
    now = cutoff or dt.datetime.now(dt.UTC)
    rows = [r for r in query("""SELECT mal_id, score, status,
                                       coalesce(finished_at::timestamptz, updated_at, now()) AS at
                                  FROM list_entry WHERE user_id = %s""", (user_id,))
            if r["mal_id"] not in exclude]
    scored = [Example(r["mal_id"], float(r["score"]), recency_weight(r["at"], now, half_life),
                      r["at"])
              for r in rows if r["score"] > 0 and (cutoff is None or r["at"] < cutoff)]
    if not scored:
        return [], []
    mean = float(np.average([e.score for e in scored], weights=[e.weight for e in scored]))
    implicit = []
    for r in rows:
        off = cfg.implicit_offsets.get(r["status"])
        if r["score"] > 0 or off is None or cfg.implicit_weight <= 0:
            continue
        if cutoff is not None and r["at"] >= cutoff:
            continue
        implicit.append(Example(r["mal_id"], float(min(max(mean + off, 1.0), 10.0)),
                                cfg.implicit_weight * recency_weight(r["at"], now, half_life),
                                r["at"]))
    return scored, implicit


# ------------------------------------------------------------ global model --

def fit_global(activate: bool = True, stack_share: float = 0.15, seed: int = 0,
               params: CFParams | None = None, exclude: set[int] | None = None) -> dict:
    """Fit on every sampled user except a held-out slice for the stacker.
    `exclude` keeps users out of the fit entirely (the gate's exam users)."""
    t0 = time.time()
    data = load_cf_ratings()
    for u in exclude or ():
        data.pop(u, None)
    # Who fits the population model and who fits the stacker is decided per
    # user by a stable hash, not a shuffle: a refit with new lists keeps
    # everyone else in the same role, so two fits differ by their data only.
    users = sorted(data)
    stack_ids = [u for u in users if unit_hash("role", seed, u) < stack_share]
    pop_ids = [u for u in users if unit_hash("role", seed, u) >= stack_share]
    cut = len(pop_ids)
    ua, ia, ra = [], [], []
    for u in pop_ids:
        for m, s, _ in data[u]:
            ua.append(u); ia.append(m); ra.append(s)
    pop = PopulationModel(params or CFParams()).fit(np.array(ua), np.array(ia), np.array(ra))
    ou, oi = load_cf_entries()
    fit_users = np.isin(ou, np.array(pop_ids))      # exam users are not in `users`
    pop.fit_occurrence(ou[fit_users], oi[fit_users])
    if settings().relevance_ease:
        pop.fit_ease(ou[fit_users], oi[fit_users])
    store = ItemStore.load()
    # The display calibration reuses the stacker's users. Splitting them
    # (so neither is judged on data it was fitted to) was tried on
    # 2026-09-27: the smaller stacker cost the reference profile 0.002 rho on
    # 4 of 8 splits, more than the slight optimism in the calibration is worth.
    held = {u: data[u] for u in stack_ids}
    cal_users = held
    stacker = fit_stacker(pop, held, store, seed=seed)
    names = STACK_FEATURES + stacker_extras(stacker)
    meta = {"users": len(users), "pop_users": cut, "stack_users": len(held),
            "calibration_users": "same as stacker", "exam_users_left_out": len(exclude or ()),
            "ratings": len(ra), "items": len(pop.items),
            "params": vars(pop.p), "stacker": dict(zip(
                names, [round(float(c), 4) for c in stacker.coef_])),
            "size_calibration": fit_size_calibration(pop, stacker, store, cal_users,
                                                     seed=seed),
            "seconds": round(time.time() - t0, 1)}
    blob = GlobalModel(pop, stacker, meta).dumps()
    with conn() as c, c.cursor() as cur:
        if activate:
            cur.execute("UPDATE global_model SET active=false WHERE active")
        cur.execute("INSERT INTO global_model (active, meta, artifact) VALUES (%s,%s,%s)"
                    " RETURNING id", (activate, Jsonb(meta), blob))
        gid = cur.fetchone()["id"]
        c.commit()
    invalidate()
    log.info("global model %s: %s", gid, meta)
    return {"id": gid, **meta}


CAL_SIZES = (5, 10, 25, 60, 150, 400)
TOP_K = 60            # "top of list": within the user's 60 best pool predictions
TOP_CAL_USERS = 300   # users per size whose whole pool is scored for the top line


def fit_size_calibration(pop: PopulationModel, stacker, store: ItemStore,
                         users: dict[int, list[tuple]], seed: int = 0,
                         sizes: tuple[int, ...] = CAL_SIZES) -> list[dict]:
    """The displayed-score map for users too new to calibrate on their own.

    A small account has no temporal holdout to fit its own map on, so its raw
    predictions were shown as they are - and the top of a ranked list then
    reads 9-10, because predictions for thin lists are both noisy and spread
    too wide. Here, for each list size, users the population model never saw
    are given that many of their older ratings and asked to predict their
    newest ones; the least-squares line from prediction to truth is that
    size's map.

    A second line per size (`top_slope` / `top_intercept`) is fitted only on
    the future ratings whose prediction would have made that user's top
    TOP_K of the candidate pool - what a list actually shows. Measured
    (experiments/exp_topcal.py) the general line *understates* those by
    0.16-0.44 points; the top line removes that bias. Being flatter it is
    worse for arbitrary titles, so it is used for list displays only.
    """
    from ..config import settings
    mode = settings().model_mode
    cfg = settings()
    pool = [m for m, r in store.rows.items()
            if (r.get("mal_num_scoring_users") or 0) >= cfg.min_scoring_users
            and r.get("format_class") == "main" and r.get("nsfw") in (None, "white")
            and r.get("status") != "not_yet_aired"]
    from ..uncertainty import coverage, residual_quantiles
    out = []
    for n in sizes:
        preds, truth, top_p, top_t = [], [], [], []
        owner: list[int] = []            # which user each pair came from
        uidx = 0
        scored_pool = 0
        # stable order and per-user draws: the same user gets the same
        # split and subset in every refit
        for u, ratings in sorted(users.items(), key=lambda kv: unit_hash("cal", seed, kv[0])):
            if len(ratings) < n + 8:
                continue
            rng = user_rng("cal", seed, u, n)
            train, test, cutoff = split_user(ratings, rng)
            if len(train) < n:
                continue
            tr = [train[i] for i in rng.permutation(len(train))[:n]]
            ex = [Example(m, sc, recency_weight(a, cutoff), a) for m, sc, a in tr]
            um = fit_user(pop, stacker, store, ex, [], mode=mode if mode != "personal" else "sized")
            items = [m for m, _, _ in test]
            p = um.predict(items)
            ok = ~np.isnan(p)
            t = np.array([sc for _, sc, _ in test])
            preds.extend(p[ok].tolist())
            truth.extend(t[ok].tolist())
            uidx += 1
            owner.extend([uidx] * int(ok.sum()))
            if scored_pool < TOP_CAL_USERS:
                scored_pool += 1
                own = {m for m, _, _ in ratings}
                pp = um.predict([m for m in pool if m not in own])
                pp = pp[~np.isnan(pp)]
                if len(pp) >= TOP_K:
                    top = ok & (np.nan_to_num(p, nan=-1e9) >= np.sort(pp)[-TOP_K])
                    top_p.extend(p[top].tolist())
                    top_t.extend(t[top].tolist())
        if len(preds) < 200:
            continue
        cal = calibration_from(preds, truth)
        row = {"n": n, "slope": cal["slope"], "intercept": cal["intercept"],
               "points": len(preds), "rmse_raw": cal.get("rmse_raw"),
               "rmse_calibrated": cal.get("rmse_calibrated")}
        shown = np.clip(cal["slope"] * np.array(preds) + cal["intercept"], 1, 10)
        tr = np.array(truth)
        row["res_q"] = residual_quantiles(shown, tr)
        # honest coverage: quantiles from even users, checked on odd users
        odd = np.array(owner) % 2 == 1
        if odd.any() and (~odd).any():
            from ..uncertainty import POP_BAND
            row["coverage80"] = round(coverage(residual_quantiles(shown[~odd], tr[~odd]),
                                               shown[odd], tr[odd], POP_BAND), 3)
        if len(top_p) >= 200:
            tc = calibration_from(top_p, top_t)
            if "rmse_raw" in tc:
                row.update({"top_slope": tc["slope"], "top_intercept": tc["intercept"],
                            "top_points": len(top_p)})
                ts = np.clip(tc["slope"] * np.array(top_p) + tc["intercept"], 1, 10)
                row["top_res_q"] = residual_quantiles(ts, top_t)
        out.append(row)
        log.info("size calibration n=%d: %s", n, out[-1])
    return out


def size_quantiles(table: list[dict] | None, n: int) -> list[float] | None:
    """Error quantiles for a list of `n` ratings: the nearest fitted size,
    top-of-list errors where available (list cards show the top)."""
    rows = [r for r in table or [] if "res_q" in r]
    if not rows:
        return None
    r = min(rows, key=lambda r: abs(np.log(max(n, 1)) - np.log(r["n"])))
    return r.get("top_res_q") or r["res_q"]


def size_calibration(table: list[dict] | None, n: int,
                     prefix: str = "") -> tuple[float, float] | None:
    """Slope and intercept for a list of `n` ratings, interpolated in log(n)
    between the fitted sizes and held flat beyond them. prefix="top_" gives
    the top-of-list line (sizes without one are skipped)."""
    s_key, i_key = f"{prefix}slope", f"{prefix}intercept"
    t = sorted((r for r in table or [] if s_key in r), key=lambda r: r["n"])
    if not t:
        return None
    if n <= t[0]["n"]:
        return t[0][s_key], t[0][i_key]
    if n >= t[-1]["n"]:
        return t[-1][s_key], t[-1][i_key]
    for a, b in itertools.pairwise(t):
        if a["n"] <= n <= b["n"]:
            f = (np.log(n) - np.log(a["n"])) / (np.log(b["n"]) - np.log(a["n"]))
            return (float(a[s_key] + f * (b[s_key] - a[s_key])),
                    float(a[i_key] + f * (b[i_key] - a[i_key])))
    return None


def active_global() -> GlobalModel | None:
    row = one("SELECT id FROM global_model WHERE active")
    if row is None:
        return None
    with _lock:
        if _cache["global_id"] != row["id"]:
            blob = one("SELECT artifact FROM global_model WHERE id=%s", (row["id"],))["artifact"]
            _cache["global"] = GlobalModel.loads(bytes(blob))
            _cache["global_id"] = row["id"]
        return _cache["global"]


def load_global(gid: int) -> GlobalModel:
    """A stored population model by id, active or not (uncached)."""
    blob = one("SELECT artifact FROM global_model WHERE id=%s", (gid,))["artifact"]
    return GlobalModel.loads(bytes(blob))


def activate_global(gid: int) -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute("UPDATE global_model SET active=false WHERE active")
        cur.execute("UPDATE global_model SET active=true WHERE id=%s", (gid,))
        c.commit()
    invalidate()


def item_store() -> ItemStore:
    with _lock:
        if _cache["store"] is None or time.time() - _cache["store_at"] > STORE_TTL:
            _cache["store"] = ItemStore.load()
            _cache["store_at"] = time.time()
        return _cache["store"]


def invalidate() -> None:
    with _lock:
        _cache.update({"global": None, "global_id": None, "store": None, "store_at": 0.0})


# ------------------------------------------------------------- user model --

def blend_opts() -> dict:
    """How the personal and population models are mixed, from config - one
    place, so serving, evaluation and the gate always mix them the same way."""
    cfg = settings()
    return {"size_n0": cfg.blend_size_n0, "lam_max": cfg.blend_lam_max}


def blend_mode(n_scored: int) -> str:
    """The configured mode, except that long histories choose their blend
    weight on their own newest ratings (config.chosen_blend_min)."""
    cfg = settings()
    if cfg.model_mode == "sized" and n_scored >= cfg.chosen_blend_min:
        return "blend"
    return cfg.model_mode


def fit_for_user(user_id: int, mode: str | None = None, half_life: float | None = None,
                 cutoff: dt.datetime | None = None) -> UserModel | None:
    """None when no population model has been fitted yet."""
    gm = active_global()
    if gm is None:
        return None
    scored, implicit = user_examples(user_id, cutoff, half_life)
    if not scored:
        return None
    mode = mode or blend_mode(len(scored))
    return fit_user(gm.pop, gm.stacker, item_store(), scored, implicit,
                    mode=mode, blend_of="personal", **blend_opts(),
                    listed=user_listed(user_id, cutoff, half_life))


# ------------------------------------------------------------- evaluation --

def temporal_holdout(user_id: int, holdouts: tuple[int, ...] = (30, 40, 50),
                     mode: str | None = None, gm: GlobalModel | None = None,
                     half_life: float | None = None, **opts) -> dict:
    """Train on older ratings, predict the newest - through the same path
    production uses. Unlike the legacy SQL holdout, the taste vector and every
    recency weight are computed from the cutoff, so nothing after it leaks in.
    Returns metrics plus the pooled out-of-sample predictions for calibration.

    The blend weight is the one the user's model has today (their full rating
    count), not the one the truncated training window would imply: the
    holdout scores the model that is actually served, and the display
    calibration fitted on it maps that model's predictions.
    """
    from ..eval import ndcg_at, spearman
    gm = gm or active_global()
    if gm is None:
        return {"note": "no population model fitted"}
    rated = query("""SELECT mal_id, score, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                       FROM list_entry WHERE user_id=%s AND score>0 ORDER BY at""", (user_id,))
    store = item_store()
    per, preds, truth = [], [], []
    for h in holdouts:
        if len(rated) < h + 25:
            continue
        test = rated[-h:]
        cutoff = test[0]["at"]
        scored, implicit = user_examples(user_id, cutoff, half_life)
        um = fit_user(gm.pop, gm.stacker, store, scored, implicit,
                      mode=mode or blend_mode(len(rated)), blend_of="personal",
                      **{**blend_opts(), "size_n": len(rated), **opts})
        items = [r["mal_id"] for r in test]
        p = um.predict(items)
        y = np.array([float(r["score"]) for r in test])
        ok = ~np.isnan(p)
        if ok.sum() < 5:
            continue
        per.append({"holdout": h, "spearman": spearman(p[ok], y[ok]),
                    "rmse": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))),
                    "ndcg@10": ndcg_at(p[ok], list(y[ok]), 10)})
        preds.extend(p[ok].tolist())
        truth.extend(y[ok].tolist())
    if not per:
        return {"note": "not enough rated history for a temporal holdout",
                "n_rated": len(rated)}
    return {"method": "temporal_holdout", "n_rated": len(rated),
            "spearman": round(float(np.mean([r["spearman"] for r in per])), 4),
            "rmse": round(float(np.mean([r["rmse"] for r in per])), 4),
            "ndcg@10": round(float(np.mean([r["ndcg@10"] for r in per])), 4),
            "per_holdout": per, "_preds": preds, "_truth": truth}


def calibration_from(preds: list[float], truth: list[float]) -> dict:
    """Least-squares map raw -> score on out-of-sample predictions; identity
    when there is too little to fit, or when the fit would invert the order."""
    if len(preds) < 20 or float(np.std(preds)) < 1e-6:
        return {"slope": 1.0, "intercept": 0.0}
    p, t = np.array(preds), np.array(truth)
    slope, intercept = np.polyfit(p, t, 1)
    if slope <= 0:
        return {"slope": 1.0, "intercept": 0.0}
    fitted = np.clip(slope * p + intercept, 1.0, 10.0)
    return {"slope": round(float(slope), 4), "intercept": round(float(intercept), 4),
            "rmse_raw": round(float(np.sqrt(((p - t) ** 2).mean())), 4),
            "rmse_calibrated": round(float(np.sqrt(((fitted - t) ** 2).mean())), 4)}


# ----------------------------------------------------- prospective check --

PROSPECTIVE_MIN = 20          # pooled pairs before the check can decide anything
PROSPECTIVE_TOLERANCE = 0.01  # rank correlation a candidate may lose on them


def prospective_holdout(user_id: int, gm: GlobalModel | None = None,
                        half_life: float | None = None, **opts) -> dict | None:
    """The model refitted as of the first time any of these titles was shown,
    predicting the titles the app recommended and the user rated afterwards
    (rec_log x list_entry). Those ratings, and the rated titles themselves
    whatever their dates, are kept out of the training data, so a candidate
    model is judged on exactly the ratings that did not exist when the app
    made its recommendations. None when the user has no such ratings yet."""
    from ..eval import PROSPECTIVE_SQL
    gm = gm or active_global()
    rows = query(PROSPECTIVE_SQL, (user_id,))
    if gm is None or not rows:
        return None
    cutoff = min(r["shown_at"] for r in rows)
    test = frozenset(r["mal_id"] for r in rows)
    scored, implicit = user_examples(user_id, cutoff, half_life, exclude=test)
    if not scored:
        return None
    um = fit_user(gm.pop, gm.stacker, item_store(), scored, implicit,
                  mode=blend_mode(len(scored)), blend_of="personal",
                  listed=user_listed(user_id, cutoff, half_life, exclude=test),
                  **{**blend_opts(), **opts})
    ids = [r["mal_id"] for r in rows]
    p = np.asarray(um.predict(ids), dtype=float)
    ok = ~np.isnan(p)
    return {"pred": p[ok], "truth": np.array([float(r["score"]) for r in rows])[ok],
            "shown": np.array([float(r["predicted"]) for r in rows])[ok]}


def _half_life(user_id: int) -> float | None:
    run = one("SELECT params FROM model_run WHERE user_id=%s ORDER BY id DESC LIMIT 1", (user_id,))
    return ((run or {}).get("params") or {}).get("half_life")


def prospective_compare(before: GlobalModel, after: GlobalModel, gate_user: str | None = None,
                        opts_before: dict | None = None, opts_after: dict | None = None) -> dict:
    """Current vs candidate on the prospective ratings of every approved
    account. Decides only with at least PROSPECTIVE_MIN pairs in total: the
    candidate may lose at most PROSPECTIVE_TOLERANCE of pair-weighted rank
    correlation overall, nor on the gate user once they have that many pairs
    of their own. Below that it reports "waiting" and blocks nothing.
    `opts_*` are passed to fit_user, so a setting can be tested the same way
    as a new population model."""
    from ..eval import spearman
    per, total = {}, 0
    for u in query("SELECT id, mal_username FROM app_user WHERE status='approved' ORDER BY id"):
        hl = _half_life(u["id"])
        b = prospective_holdout(u["id"], before, hl, **(opts_before or {}))
        if b is None:
            continue
        total += len(b["truth"])
        if len(b["truth"]) < 5:          # too few for a rank correlation of their own
            continue
        same = after is before and (opts_after or {}) == (opts_before or {})
        a = b if same else prospective_holdout(u["id"], after, hl, **(opts_after or {}))
        n = len(b["truth"])
        per[u["mal_username"]] = {
            "pairs": n,
            "rho_before": round(spearman(b["pred"], b["truth"]), 4),
            "rho_after": round(spearman(a["pred"], a["truth"]), 4),
            "rho_as_shown": round(spearman(b["shown"], b["truth"]), 4),
            "rmse_before": round(float(np.sqrt(np.mean((b["pred"] - b["truth"]) ** 2))), 3),
            "rmse_after": round(float(np.sqrt(np.mean((a["pred"] - a["truth"]) ** 2))), 3)}
    out: dict = {"pairs": total, "users": per}
    if total < PROSPECTIVE_MIN or not per:
        out.update(status="waiting", ok=True,
                   note=f"{total} of {PROSPECTIVE_MIN} recommended titles rated since shown")
        return out

    def pooled(k: str) -> float:
        w = [v["pairs"] for v in per.values()]
        return round(float(np.average([v[k] for v in per.values()], weights=w)), 4)

    out["rho_before"], out["rho_after"] = pooled("rho_before"), pooled("rho_after")
    ok = out["rho_after"] >= out["rho_before"] - PROSPECTIVE_TOLERANCE
    g = per.get(gate_user or "")
    if g and g["pairs"] >= PROSPECTIVE_MIN:
        ok = ok and g["rho_after"] >= g["rho_before"] - PROSPECTIVE_TOLERANCE
    out.update(status="passed" if ok else "failed", ok=ok)
    return out


# ------------------------------------------------------------ gated refit --

GATE_SPLITS = (25, 30, 35, 40, 45, 50, 55, 60)
EXAM_MIN_SCORED = 15          # exam users need a list long enough to split
EXAM_MIN_USERS = 100          # below this the exam is skipped (gate user rule only)
EXAM_TOLERANCE = 0.002        # mean rank correlation a candidate may lose on the exam
GATE_USER_VETO = 0.01         # ... and on the gate user's 8 splits, when the exam decides
GATE_TOLERANCE = 0.002        # without an exam: the old single-user rule
GATE_MAX_SPLIT_DROP = 0.02


EXAM_BATCH_DAYS = 2           # lists fetched this close to the newest one form its batch


def exam_users(since: dt.datetime) -> set[int]:
    """The newest batch of sampled lists (one rotation), if it was fetched
    after `since` (the active model's fit). The active model has never seen
    it and the candidate is fitted without it, so it is a fair exam for
    both - while every earlier batch the active model has not seen trains
    the candidate, so a refit after a failed gate still gains new data."""
    rows = query("""SELECT id FROM cf_user
                     WHERE state='done' AND n_scored >= %s AND fetched_at > %s
                       AND fetched_at >= (SELECT max(fetched_at) FROM cf_user WHERE state='done')
                                         - make_interval(days => %s)""",
                 (EXAM_MIN_SCORED, since, EXAM_BATCH_DAYS))
    return {r["id"] for r in rows}


def population_exam(gm: GlobalModel, data: dict[int, list[tuple]], users: set[int],
                    seed: int = 0) -> dict[int, float]:
    """Rank correlation on each exam user's newest ratings, predicted from
    their older ones, through the same fit an app user of that size gets."""
    from ..eval import spearman
    store = item_store()
    out: dict[int, float] = {}
    for u in sorted(users):
        ratings = data.get(u) or []
        if len(ratings) < EXAM_MIN_SCORED:
            continue
        # seeded per user, so both models see the same split
        train, test, cutoff = split_user(ratings, np.random.default_rng(seed + u))
        ex = [Example(m, sc, recency_weight(a, cutoff), a) for m, sc, a in train]
        um = fit_user(gm.pop, gm.stacker, store, ex, [], mode=blend_mode(len(ex)),
                      blend_of="personal", **blend_opts())
        p = np.asarray(um.predict([m for m, _, _ in test]), dtype=float)
        y = np.array([sc for _, sc, _ in test])
        ok = ~np.isnan(p)
        if ok.sum() >= 5 and y[ok].std() > 0:
            out[u] = spearman(p[ok], y[ok])
    return out


def gate_decision(user_before: list[float], user_after: list[float],
                  exam: dict | None) -> tuple[bool, str]:
    """Whether a candidate population model may replace the active one.

    With an exam (hundreds of users neither model has seen) the exam
    decides: the candidate fails only if it loses more than EXAM_TOLERANCE
    of mean rank correlation *and* more than one standard error of the
    paired difference - two equally good refits differ by about that much,
    and a bar inside the noise would reject a quarter of them. The gate user
    keeps a veto against a clear loss on their
    own 8 splits (mean drop above GATE_USER_VETO) - one profile's nested
    splits move by ~0.02 between honest refits, too noisy to decide alone.
    Without an exam the old single-user rule applies.
    """
    mb, ma = float(np.mean(user_before)), float(np.mean(user_after))
    if exam is not None:
        if exam["delta"] < -max(EXAM_TOLERANCE, exam.get("se", 0.0)):
            return False, "exam failed: the candidate predicts unseen lists worse"
        if ma < mb - GATE_USER_VETO:
            return False, "gate user veto: clearly worse on their own holdout"
        return True, "exam passed"
    worst = min(a - b for a, b in zip(user_after, user_before))
    if ma < mb - GATE_TOLERANCE or worst < -GATE_MAX_SPLIT_DROP:
        return False, "gate failed: the new model would cost the gate user accuracy"
    return True, "gate passed (no exam: too few new lists)"


def gated_fit(gate_user: str, dry_run: bool = False) -> dict:
    """Fit a new population model inactive and activate it only if it passes
    gate_decision and prospective_compare on the ratings given after
    recommendations (which waits, blocking nothing, until there are enough of
    them). `dry_run` reports the decision without activating anything."""
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (gate_user,))["id"]
    current = active_global()
    cur_row = one("SELECT id, created_at FROM global_model WHERE active")
    exam_ids = exam_users(cur_row["created_at"]) if cur_row else set()
    use_exam = len(exam_ids) >= EXAM_MIN_USERS
    new = fit_global(activate=current is None and not dry_run,
                     exclude=exam_ids if use_exam else None)
    if current is None:
        return {"activated": new["id"], "reason": "no previous model"}
    candidate = load_global(new["id"])
    # the gate user at their own half-life and real size: the model they get
    hl = _half_life(uid)
    before = temporal_holdout(uid, GATE_SPLITS, gm=current, half_life=hl)
    after = temporal_holdout(uid, GATE_SPLITS, gm=candidate, half_life=hl)
    b = [r["spearman"] for r in before["per_holdout"]]
    a = [r["spearman"] for r in after["per_holdout"]]
    exam = None
    if use_exam:
        data = load_cf_ratings()
        eb = population_exam(current, data, exam_ids)
        ea = population_exam(candidate, data, exam_ids)
        common = sorted(eb.keys() & ea.keys())
        if len(common) >= EXAM_MIN_USERS:
            d = np.array([ea[u] - eb[u] for u in common])
            exam = {"users": len(common),
                    "rho_before": round(float(np.mean([eb[u] for u in common])), 4),
                    "rho_after": round(float(np.mean([ea[u] for u in common])), 4),
                    "delta": round(float(d.mean()), 4),
                    "se": round(float(d.std(ddof=1) / np.sqrt(len(d))), 4),
                    "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum())}
    ok, reason = gate_decision(b, a, exam)
    pro = prospective_compare(current, candidate, gate_user)
    if ok and not pro["ok"]:
        ok, reason = False, "prospective check failed: worse on ratings given after recommendations"
    out = {"candidate": new["id"], "users": new["users"], "exam": exam,
           "exam_users_left_out": len(exam_ids) if use_exam else 0,
           "gate": {"user": gate_user, "half_life": hl, "rho_before": before["spearman"],
                    "rho_after": after["spearman"],
                    "worst_split_change": round(min(x - y for x, y in zip(a, b)), 4),
                    "better": sum(x > y + 1e-9 for x, y in zip(a, b)),
                    "worse": sum(x < y - 1e-9 for x, y in zip(a, b))},
           "prospective": pro, "passed": ok, "reason": reason}
    if ok and not dry_run:
        activate_global(new["id"])
        out["activated"] = new["id"]
    else:
        out["kept"] = cur_row["id"]
    # keep the active model and the two newest others as fallbacks (~24 MB each)
    execute("DELETE FROM global_model WHERE NOT active AND id NOT IN"
            " (SELECT id FROM global_model WHERE NOT active ORDER BY id DESC LIMIT 2)")
    log.info("gated population fit: %s", out)
    return out
