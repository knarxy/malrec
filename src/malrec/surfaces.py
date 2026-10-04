"""The recommendation surfaces the web app will render.

Each surface is the same pipeline with a different filter and a different
novelty/diversity trade-off, so they stay consistent with one another and with
the offline evaluation.

    discover        the default mix - best predicted fit, balanced novelty
    hidden_gems     high predicted fit that almost nobody has watched
    next_up         the next full season or film of something they finished
    side_stories    OVAs and specials from franchises they already know
    this_season     currently airing, ranked by fit
    safe_bets       highest predicted score, novelty ignored
    plan_to_watch   the user's own Plan to Watch, ranked by predicted fit
    coming_soon     announced seasons and films of franchises the user liked,
                    soonest first

Each surface keeps its scored shortlist (rec_candidate), so a filtered read
can re-rank it without refitting anything.

Every surface except `side_stories` is restricted to main formats. MAL labels
a 4-minute joke short and a full second season both as "sequel", so without
that split the continuations surface fills with specials and buries the thing
the user actually wants to watch next.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
from psycopg.types.json import Jsonb

from .config import settings
from .db import conn, query, scalar
from .model import TasteModel, load_or_train
from .rank import (
    Candidate,
    attach_reasons,
    model_drivers,
    relevance_bonus,
    rerank,
    retrieve,
    score_candidates,
)

log = logging.getLogger(__name__)

# in tab order: safe_bets is the primary list, discover picks up after it
SURFACES = ("safe_bets", "discover", "hidden_gems", "next_up", "coming_soon",
            "plan_to_watch", "side_stories", "this_season")
# surfaces ordered by rerank(); a filtered read re-runs it on the shortlist
RERANKED = ("discover", "hidden_gems", "this_season", "safe_bets")
POOL_KEEP = 300         # shortlist kept per surface for re-ranking
BUILD_LIMIT = 60        # list length build_all() stores

# Format classes, as defined by format_class() in db/migrations/005_formats.sql:
#   main = tv / movie / ona,  side = ova / special / tv_special
MAIN = ["main"]
SIDE = ["side"]


@dataclass
class SurfaceSpec:
    name: str
    novelty_weight: float
    diversity_weight: float
    min_scorers: int
    description: str
    formats: list[str] = field(default_factory=lambda: list(MAIN))


SPECS = {
    "discover": SurfaceSpec(
        "discover", 1.0, 0.07, 2000,
        "Beyond the safe bets: strong fits, balancing how well they match your "
        "taste against how likely you are to have heard of them already."),
    "hidden_gems": SurfaceSpec(
        "hidden_gems", 2.2, 0.05, 300,
        "Strong predicted fit among titles with a small audience."),
    "safe_bets": SurfaceSpec(
        "safe_bets", 0.0, 0.10, 2000,
        "Strong fits you have probably heard of. Ordered by predicted score, adjusted "
        "for your whole history and for how common a title is on lists like yours."),
    "next_up": SurfaceSpec(
        "next_up", 0.0, 0.0, 0,
        "The next full season or film of something you finished and liked."),
    "side_stories": SurfaceSpec(
        "side_stories", 0.0, 0.0, 0,
        "OVAs, specials and shorts from franchises you already know.",
        formats=list(SIDE)),
    "this_season": SurfaceSpec(
        "this_season", 0.5, 0.08, 100,
        "Currently airing, ranked by how well it fits you."),
    "coming_soon": SurfaceSpec(
        "coming_soon", 0.0, 0.0, 0,
        "Announced seasons and films of franchises you rated highly, soonest first."),
    "plan_to_watch": SurfaceSpec(
        "plan_to_watch", 0.0, 0.0, 0,
        "Your own Plan to Watch, ordered by how much we expect you to like each title.",
        formats=["main", "side"]),
}


def _current_season(today: dt.date | None = None) -> tuple[int, str]:
    d = today or dt.datetime.now(dt.UTC).date()
    return d.year, {1: "winter", 2: "spring", 3: "summer", 4: "fall"}[(d.month - 1) // 3 + 1]


def _user_mean(user_id: int) -> float:
    return float(scalar("SELECT user_mean(%s, %s::real)",
                        (user_id, settings().recency_half_life_years)) or 7.0)


# `next_up` bypasses the normal candidate pool: eligible_candidates deliberately
# excludes anything whose prerequisites are unmet, and a sequel to a finished
# show has its prerequisites met by definition, so it lands here instead of
# competing with genuine discoveries in `discover`.
NEXT_UP_SQL = """
    SELECT a.mal_id, a.title, coalesce(f.franchise_id, a.mal_id) AS franchise_id,
           a.mal_mean, a.mal_popularity,
           prev.title AS after_title, prev_le.score AS after_score, prev.mal_id AS after_id
      FROM relation r            -- (src, dst, 'sequel'): dst is src's sequel; 'side_story' likewise
      JOIN list_entry prev_le ON prev_le.mal_id = r.src AND prev_le.user_id = %(uid)s
      JOIN anime prev ON prev.mal_id = r.src
      JOIN anime a ON a.mal_id = r.dst
      LEFT JOIN franchise f ON f.mal_id = a.mal_id
     WHERE r.relation_type = ANY(%(rels)s)
       AND prev_le.status = 'completed'
       AND prev_le.score >= %(min_score)s
       AND NOT EXISTS (SELECT 1 FROM list_entry le
                        WHERE le.user_id = %(uid)s AND le.mal_id = a.mal_id)
       AND NOT EXISTS (SELECT 1 FROM feedback fb
                        WHERE fb.user_id = %(uid)s AND fb.mal_id = a.mal_id
                          AND fb.action IN ('not_interested','hidden','seen_it'))
       AND coalesce(a.status,'') <> 'not_yet_aired'
       -- a side story counts whatever MAL calls its format (the Lord of
       -- Mysteries specials are an "ONA"); a sequel only in the asked formats
       AND (format_class(a.media_type) = ANY(%(formats)s) OR r.relation_type = 'side_story')
       AND prereqs_satisfied(%(uid)s, a.mal_id)
     GROUP BY a.mal_id, a.title, f.franchise_id, a.mal_mean, a.mal_popularity,
              prev.title, prev_le.score, prev.mal_id
"""


def _continuations(user_id: int, model: TasteModel, limit: int,
                   format_classes: list[str]) -> list[Candidate]:
    """Shared by next_up (main formats) and side_stories (side formats)."""
    mu = _user_mean(user_id)
    rels = ["sequel"] if format_classes == MAIN else ["sequel", "side_story"]
    rows = query(NEXT_UP_SQL, {"uid": user_id, "min_score": max(round(mu) - 1, 6),
                               "formats": format_classes, "rels": rels})
    if not rows:
        return []
    cands = score_candidates(user_id, model, rows, shortlist=limit * 4)
    by_id = {r["mal_id"]: r for r in rows}
    deltas = sequel_deltas() if format_classes == MAIN else None
    for c in cands:
        src = by_id.get(c.mal_id, {})
        if deltas is not None and src.get("after_score"):
            # half the model, half "your score for the previous season plus how
            # raters of both typically move" (experiments/exp_sequel.py: RMSE
            # 1.187 -> 1.052, rho .749 -> .800 on 3,000 held-out sequels)
            anchor = float(src["after_score"]) + sequel_delta(deltas, src["after_id"], c.mal_id)
            c.predicted = 0.5 * c.predicted + 0.5 * anchor
            if c.shown is not None:
                c.shown = 0.5 * c.shown + 0.5 * anchor
        c.final = c.predicted
        c.reasons = [{"kind": "continues", "title": src.get("after_title"),
                      "your_score": src.get("after_score")}]
    cands.sort(key=lambda c: -c.final)
    # one per franchise: only the earliest unwatched sequel is actionable
    seen, out = set(), []
    for c in cands:
        if c.franchise_id in seen:
            continue
        seen.add(c.franchise_id)
        out.append(c)
    return out[:limit]


SEQUEL_SHRINK = 20.0
_deltas: dict = {"at": 0.0, "value": None}

DELTAS_SQL = """
    WITH pr AS (
        SELECT r.src AS p, r.dst AS s FROM relation r
          JOIN anime a ON a.mal_id = r.src JOIN anime b ON b.mal_id = r.dst
         WHERE r.relation_type = 'sequel'
           AND format_class(a.media_type) = 'main' AND format_class(b.media_type) = 'main')
    SELECT pr.p, pr.s, sum(cs.score - cp.score)::float AS total, count(*) AS n
      FROM pr
      JOIN cf_rating cp ON cp.mal_id = pr.p AND cp.score > 0
      JOIN cf_rating cs ON cs.user_id = cp.user_id AND cs.mal_id = pr.s AND cs.score > 0
     GROUP BY pr.p, pr.s
"""


def sequel_deltas() -> dict:
    """{(prequel, sequel): (sum of r_s - r_p, count)} over the sampled users,
    plus the global mean delta; cached for a day."""
    import time
    if _deltas["value"] is None or time.time() - _deltas["at"] > 86400:
        rows = query(DELTAS_SQL)
        pairs = {(r["p"], r["s"]): (r["total"], r["n"]) for r in rows}
        tot = sum(v[0] for v in pairs.values())
        n = sum(v[1] for v in pairs.values()) or 1
        _deltas.update(at=time.time(), value={"pairs": pairs, "global": tot / n})
    return _deltas["value"]


def sequel_delta(deltas: dict, prequel: int, sequel: int) -> float:
    """How raters typically move from prequel to sequel, shrunk toward the
    global mean delta by SEQUEL_SHRINK pseudo-pairs."""
    total, n = deltas["pairs"].get((prequel, sequel), (0.0, 0))
    return (total + SEQUEL_SHRINK * deltas["global"]) / (n + SEQUEL_SHRINK)


AIRING_WINDOW = "7 months"


def _this_season(user_id: int, model: TasteModel, spec: SurfaceSpec, limit: int) -> list[Candidate]:
    """This season's titles plus anything still airing that began within
    AIRING_WINDOW - a second cour carries over the season boundary, and in a
    season's first weeks few new shows have the ratings to qualify yet
    (2026-10-03: three days into fall the tab held one title while 41 summer
    shows were still airing). The window keeps out decade-long runners."""
    year, season = _current_season()
    rows = query(
        """
        SELECT c.mal_id, c.franchise_id, c.title, c.mal_mean, c.mal_popularity
          FROM eligible_candidates(%s, %s, false) c
          JOIN anime a ON a.mal_id = c.mal_id
         WHERE format_class(a.media_type) = ANY(%s)
           AND ((a.season_year = %s AND a.season = %s)
                OR (a.status = 'currently_airing'
                    AND a.start_date >= current_date - %s::interval))
        """,
        (user_id, spec.min_scorers, spec.formats, year, season, AIRING_WINDOW),
    )
    drop = derived_ids()
    rows = [r for r in rows if r["mal_id"] not in drop]
    if not rows:
        return [], [], []
    pool = score_candidates(user_id, model, rows, shortlist=limit * 4)
    seen: list[Candidate] = []
    return pool, rerank(pool, spec.novelty_weight, spec.diversity_weight, limit,
                        considered=seen, profile=genre_profile(user_id)), seen


_derived: dict = {"at": 0.0, "ids": frozenset()}


def derived_ids() -> frozenset[int]:
    """Side stories and recaps: titles MAL lists with a parent story, or as a
    summary of a fuller one ("full_story"). The discovery tabs leave them out
    - a side story belongs in Side Stories, next to what it adds to, and a
    recap either spoils a series the user has not seen or repeats one they
    have. (2026-10-03: "Lord of Mysteries Specials" led Hidden Gems, and
    Attack on Titan's compilation film sat in it for someone who had not
    seen the final season.) Cached for a day."""
    import time
    if not _derived["ids"] or time.time() - _derived["at"] > 86400:
        _derived.update(at=time.time(), ids=frozenset(r["src"] for r in query(
            "SELECT DISTINCT src FROM relation"
            " WHERE relation_type IN ('parent_story', 'full_story')")))
    return _derived["ids"]


def continuation_ids(user_id: int) -> set[int]:
    """What next_up shows (build_all builds it first): the discovery lists
    leave these to it, so a continuation appears once, with its
    prequel-anchored prediction. Only what it actually shows - a second
    sequel in the same franchise stays eligible elsewhere."""
    return {r["mal_id"] for r in query(
        "SELECT mal_id FROM recommendation WHERE user_id=%s AND surface='next_up'", (user_id,))}


def _hidden_gems(user_id: int, model: TasteModel, spec: SurfaceSpec, limit: int) -> list[Candidate]:
    skip = continuation_ids(user_id) | derived_ids()
    rows = [r for r in retrieve(user_id, min_scorers=spec.min_scorers,
                                media_types=spec.formats)
            if (r.get("mal_popularity") or 99999) > 1200 and r["mal_id"] not in skip]
    pool = score_candidates(user_id, model, rows, shortlist=settings().shortlist_size)
    seen: list[Candidate] = []
    return pool, rerank(pool, spec.novelty_weight, spec.diversity_weight, limit,
                        considered=seen, profile=genre_profile(user_id)), seen


PTW_SQL = """
    SELECT a.mal_id, a.title, coalesce(f.franchise_id, a.mal_id) AS franchise_id,
           a.mal_mean, a.mal_popularity
      FROM list_entry le JOIN anime a ON a.mal_id = le.mal_id
      LEFT JOIN franchise f ON f.mal_id = a.mal_id
     WHERE le.user_id = %s AND le.status = 'plan_to_watch'
"""


# Not yet aired, main format, in a franchise where the user's best score is at
# least their own mean. Franchise, not direct sequel, because the announced
# title is often two steps from what they watched (a film between seasons).
COMING_SQL = """
    SELECT a.mal_id, a.title, coalesce(f.franchise_id, a.mal_id) AS franchise_id,
           a.mal_mean, a.mal_popularity, a.start_date, a.season_year, a.season,
           best.title AS after_title, best.score AS after_score
      FROM anime a
      JOIN franchise f ON f.mal_id = a.mal_id
      JOIN LATERAL (
            SELECT b.title, le.score
              FROM franchise f2
              JOIN list_entry le ON le.mal_id = f2.mal_id AND le.user_id = %(uid)s
              JOIN anime b ON b.mal_id = f2.mal_id
             WHERE f2.franchise_id = f.franchise_id AND le.score > 0
             ORDER BY le.score DESC, b.mal_popularity NULLS LAST LIMIT 1) best ON true
     WHERE a.status = 'not_yet_aired'
       AND format_class(a.media_type) = 'main'
       AND best.score >= %(min_score)s
       AND NOT EXISTS (SELECT 1 FROM list_entry le
                        WHERE le.user_id = %(uid)s AND le.mal_id = a.mal_id
                          AND le.status <> 'plan_to_watch')
       AND NOT EXISTS (SELECT 1 FROM feedback fb
                        WHERE fb.user_id = %(uid)s AND fb.mal_id = a.mal_id
                          AND fb.action IN ('not_interested','hidden','seen_it'))
"""


def _coming_soon(user_id: int, model: TasteModel, limit: int) -> list[Candidate]:
    mu = _user_mean(user_id)
    rows = query(COMING_SQL, {"uid": user_id, "min_score": max(round(mu), 7)})
    if not rows:
        return []
    by_id = {r["mal_id"]: r for r in rows}
    cands = score_candidates(user_id, model, rows, shortlist=len(rows), enrich=False)
    far = dt.date(9999, 1, 1)
    for c in cands:
        r = by_id[c.mal_id]
        c.final = c.predicted
        c.reasons = [{"kind": "coming",
                      "date": r["start_date"].isoformat() if r["start_date"] else None},
                     {"kind": "continues", "title": r["after_title"],
                      "your_score": r["after_score"]}]
    # soonest first; undated announcements last, better-liked franchises first
    cands.sort(key=lambda c: (by_id[c.mal_id]["start_date"] or far,
                              -by_id[c.mal_id]["after_score"], -c.predicted))
    return cands[:limit]


def _plan_to_watch(user_id: int, model: TasteModel, limit: int) -> list[Candidate]:
    """Their own queue, by predicted fit alone: relevance and novelty answer
    "would they pick it up", which a queued title already has."""
    rows = query(PTW_SQL, (user_id,))
    if not rows:
        return []
    cands = score_candidates(user_id, model, rows, shortlist=len(rows), enrich=False)
    for c in cands:
        c.final = c.predicted
    cands.sort(key=lambda c: -c.predicted)
    return cands[:limit]


def genre_profile(user_id: int, exclude: list[str] | tuple = ()) -> dict[str, float]:
    """The user's genre mix: every listed title's genres, recency-weighted,
    each title's weight split over its genres. Genres the viewer has hidden
    with a filter are left out, so the balance does not try to add them."""
    from .recsys.service import user_listed
    listed = user_listed(user_id)
    if not listed:
        return {}
    rows = query("SELECT mal_id, mal_genres FROM anime WHERE mal_id = ANY(%s)", (list(listed),))
    mix: dict[str, float] = {}
    for r in rows:
        gs = [g for g in (r["mal_genres"] or []) if g not in exclude]
        for g in gs:
            mix[g] = mix.get(g, 0.0) + listed[r["mal_id"]] / len(gs)
    tot = sum(mix.values()) or 1.0
    return {g: v / tot for g, v in mix.items()}


def long_memory_model(user_id: int, model):
    """The user's hybrid model refitted with every rating weighted equally,
    for Safe Bets' ordering. None for the legacy per-user model."""
    if not hasattr(model, "predict_ids"):
        return None
    from .model import _build_scorer
    return _build_scorer(user_id, {"half_life": float("inf")}, model.metrics,
                         model.calibration)


def build_surface(user_id: int, surface: str = "discover", limit: int = 50,
                  model: TasteModel | None = None, run_id: int | None = None,
                  persist: bool = True) -> list[Candidate]:
    if surface not in SPECS:
        raise ValueError(f"unknown surface {surface!r}; expected one of {sorted(SPECS)}")
    spec = SPECS[surface]
    if model is None:
        model, run_id = load_or_train(user_id)

    pool: list[Candidate] | None = None
    seen: list[Candidate] = []
    if surface == "next_up":
        cands = _continuations(user_id, model, limit, MAIN)
    elif surface == "side_stories":
        cands = _continuations(user_id, model, limit, SIDE)
    elif surface == "plan_to_watch":
        cands = _plan_to_watch(user_id, model, max(limit, 500))
    elif surface == "coming_soon":
        cands = _coming_soon(user_id, model, max(limit, 200))
    elif surface == "this_season":
        pool, cands, seen = _this_season(user_id, model, spec, limit)
    elif surface == "hidden_gems":
        pool, cands, seen = _hidden_gems(user_id, model, spec, limit)
    else:
        skip = continuation_ids(user_id) | derived_ids()
        rows = [r for r in retrieve(user_id, min_scorers=spec.min_scorers,
                                    media_types=spec.formats) if r["mal_id"] not in skip]
        long_model, mix = None, 0.0
        if surface == "safe_bets" and settings().safe_bets_memory_mix > 0:
            long_model = long_memory_model(user_id, model)
            mix = settings().safe_bets_memory_mix if long_model is not None else 0.0
        if surface == "discover":
            # Both rank the same pool the same way, so without this 16-19 of
            # the top 20 were shared. Safe Bets keeps the sure things;
            # Discover starts where it ends.
            taken = {r["mal_id"] for r in query(
                "SELECT mal_id FROM recommendation WHERE user_id=%s AND surface='safe_bets'",
                (user_id,))}
            rows = [r for r in rows if r["mal_id"] not in taken]
        pool = score_candidates(user_id, model, rows, long_model=long_model, mix=mix,
                                pop_prior=settings().safe_bets_popularity
                                if surface == "safe_bets" else 0.0)
        floor = settings().safe_bets_floor
        if surface == "safe_bets" and floor > -10 and hasattr(model, "baseline"):
            # a safe bet is one the current model expects the user to rate at
            # least about their own mean, whatever the ordering bonuses say
            mean_shown = float(model.calibrate([model.baseline])[0])
            pool = [c for c in pool if c.predicted >= mean_shown + floor]
        cands = rerank(pool, spec.novelty_weight, spec.diversity_weight, limit,
                       considered=seen, profile=genre_profile(user_id))

    # What is kept for re-ranking: the best of the scored pool, plus anything
    # that made the default list from further down.
    kept = cands
    nov_scale = 1.0
    if pool is not None:
        # the head of the pool, plus everything the build's rerank looked at
        # (its genres shaped the diversity penalties of what came after)
        head = {c.mal_id for c in pool[:POOL_KEEP]}
        kept = pool[:POOL_KEEP] + [c for c in {x.mal_id: x for x in seen}.values()
                                   if c.mal_id not in head]
        # what rerank() applied, which depends on the whole pool
        from .recsys.scorer import novelty_scale
        nov_scale = novelty_scale(np.array([c.predicted for c in pool]),
                                  settings().novelty_ref_sd) if pool else 1.0

    if surface not in ("next_up", "side_stories", "coming_soon"):
        attach_reasons(user_id, kept, _user_mean(user_id), model)
    elif hasattr(model, "explain"):
        # continuations already carry their own "continues X" reason; add the
        # model breakdown so the ordering within them is explainable too
        extra = model.explain([c.mal_id for c in cands], [c.novelty for c in cands],
                              [c.relevance for c in cands])
        for c, reasons in zip(cands, extra):
            c.reasons.extend(r for r in reasons if r["kind"] in ("drivers", "tags"))
    else:
        for c in cands:
            drivers = model_drivers(model, c.row) if c.row else None
            if drivers:
                c.reasons.append(drivers)
    if persist:
        save(user_id, surface, cands, run_id, kept, nov_scale, limit)
    return cands


def save(user_id: int, surface: str, cands: list[Candidate], run_id: int | None,
         kept: list[Candidate] | None = None, nov_scale: float = 1.0,
         build_limit: int = BUILD_LIMIT) -> None:
    pool = kept
    kept = cands if kept is None else kept
    with conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM recommendation WHERE user_id=%s AND surface=%s",
                    (user_id, surface))
        cur.executemany(
            "INSERT INTO recommendation (user_id, surface, mal_id, rank, predicted_score,"
            " final_score, novelty, reasons, model_run_id)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(user_id, surface, c_.mal_id, i,
              c_.shown if c_.shown is not None else c_.predicted, c_.final, c_.novelty,
              Jsonb(c_.reasons), run_id) for i, c_ in enumerate(cands, 1)],
        )
        cur.execute("DELETE FROM rec_candidate WHERE user_id=%s AND surface=%s",
                    (user_id, surface))
        cur.executemany(
            "INSERT INTO rec_candidate (user_id, surface, mal_id, franchise_id, predicted,"
            " relevance_z, personal_share, novelty, base_rank, reasons, novelty_scale,"
            " build_limit, shown, bonus, pop_bonus, risk)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(user_id, surface, c_.mal_id, c_.franchise_id, c_.predicted, c_.relevance_z,
              c_.personal_share, c_.novelty, i, Jsonb(c_.reasons), float(nov_scale),
              len(cands) if pool is None else build_limit, c_.shown, c_.bonus, c_.pop_bonus,
              c_.risk)
             for i, c_ in enumerate(kept, 1)],
        )
        # the impression log behind `malrec learn-weights`
        cur.executemany(
            "INSERT INTO rec_log (user_id, surface, mal_id, rank, predicted, relevance_z,"
            " novelty, personal_share) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            [(user_id, surface, c_.mal_id, i, c_.predicted, c_.relevance_z, c_.novelty,
              c_.personal_share) for i, c_ in enumerate(cands[:50], 1)],
        )
        c.commit()


def build_all(user_id: int, limit: int = BUILD_LIMIT) -> dict[str, int]:
    model, run_id = load_or_train(user_id)
    out = {}
    # next_up first (the discovery lists leave its titles to it), then
    # safe_bets before discover (which starts where safe_bets ends)
    for s in ("next_up",) + tuple(x for x in SURFACES if x != "next_up"):
        out[s] = len(build_surface(user_id, s, limit=limit, model=model, run_id=run_id))
    return out


READ_SQL = """
    SELECT r.rank, r.mal_id, a.title, a.title_en, a.media_type, a.num_episodes,
           a.season_year, a.season, a.mal_mean, a.mal_popularity, a.picture_medium,
           a.picture_large, a.synopsis, a.mal_genres, a.mal_studios,
           r.predicted_score, r.final_score, r.novelty, r.reasons, r.generated_at,
           le.status AS list_status, le.score AS list_score
      FROM recommendation r JOIN anime a ON a.mal_id = r.mal_id
      LEFT JOIN list_entry le ON le.user_id = r.user_id AND le.mal_id = r.mal_id
     WHERE r.user_id = %s AND r.surface = %s
     ORDER BY r.rank
     LIMIT %s OFFSET %s
"""


def read_surface(user_id: int, surface: str, limit: int = 30, offset: int = 0) -> list[dict]:
    return with_likely(user_id, query(READ_SQL, (user_id, surface, limit, offset)))


def with_likely(user_id: int, items: list[dict]) -> list[dict]:
    """Adds the likely rating range and the chance of a 9+ to each item."""
    from .uncertainty import band, user_quantiles
    q, idx = user_quantiles(user_id)
    for it in items:
        it["likely"] = band(it.get("predicted_score"), q, idx)
    return items


# ---------------------------------------------------------------- re-ranking --

CAND_SQL = """
    SELECT c.mal_id, c.franchise_id, c.predicted, c.relevance_z, c.personal_share,
           c.novelty, c.base_rank, c.reasons, c.generated_at, c.novelty_scale, c.build_limit, c.shown, c.bonus,
           a.title, a.title_en, a.media_type, a.num_episodes, a.season_year, a.season,
           a.mal_mean, a.mal_popularity, a.picture_medium, a.picture_large, a.synopsis,
           a.mal_genres, a.mal_studios,
           le.status AS list_status, le.score AS list_score
      FROM rec_candidate c JOIN anime a ON a.mal_id = c.mal_id
      LEFT JOIN list_entry le ON le.user_id = c.user_id AND le.mal_id = c.mal_id
     WHERE c.user_id = %s AND c.surface = %s
     ORDER BY c.base_rank
"""


@dataclass
class Filters:
    """What the viewer narrowed the list to. Empty means everything."""
    media_types: list[str] = field(default_factory=list)   # tv, movie, ona, ova, ...
    eps_min: int | None = None
    eps_max: int | None = None
    year_min: int | None = None
    year_max: int | None = None
    exclude_genres: list[str] = field(default_factory=list)

    def active(self) -> bool:
        return bool(self.media_types or self.exclude_genres) or any(
            v is not None for v in (self.eps_min, self.eps_max, self.year_min, self.year_max))

    def keep(self, r: dict) -> bool:
        if self.media_types and (r.get("media_type") or "") not in self.media_types:
            return False
        eps = r.get("num_episodes")
        # a film is one "episode"; episode filters are about series length
        if r.get("media_type") != "movie" and eps:
            if self.eps_min is not None and eps < self.eps_min:
                return False
            if self.eps_max is not None and eps > self.eps_max:
                return False
        y = r.get("season_year")
        if y is not None:
            if self.year_min is not None and y < self.year_min:
                return False
            if self.year_max is not None and y > self.year_max:
                return False
        return not (set(self.exclude_genres) & set(r.get("mal_genres") or []))


class _Share:
    def __init__(self, lam: float):
        self.personal_share = lam


# Other tabs may show a title Safe Bets also lists (Hidden Gems most of
# all): leaving them out lost more good picks than it saved (exp_toplist.py
# round 10), so the card says it is in Safe Bets too. Safe Bets, the primary
# list, carries no such tag.
def mark_shared(user_id: int, surface: str, items: list[dict]) -> list[dict]:
    """Tag each item Safe Bets also lists with also_in='safe_bets'."""
    if surface == "safe_bets" or not items:
        return items
    ids = {r["mal_id"] for r in query(
        "SELECT mal_id FROM recommendation WHERE user_id=%s AND surface='safe_bets'"
        " AND mal_id = ANY(%s)", (user_id, [it["mal_id"] for it in items]))}
    for it in items:
        if it["mal_id"] in ids:
            it["also_in"] = "safe_bets"
    return items


def read_ranked(user_id: int, surface: str, limit: int = 30, offset: int = 0,
                filters: Filters | None = None) -> list[dict]:
    """The surface narrowed by filters, re-ranked the way it was built.

    Titles that have since landed on the list drop out of every other tab at
    the next load - planned ones too, which move to the Plan to Watch tab.
    """
    rows = query(CAND_SQL, (user_id, surface))
    if not rows:
        items = read_surface(user_id, surface, limit, offset)
        return items if surface == "plan_to_watch" else [
            it for it in items if it.get("list_status") is None]
    filters = filters or Filters()
    if surface != "plan_to_watch":
        rows = [r for r in rows if r["list_status"] is None]
    else:
        rows = [r for r in rows if r["list_status"] == "plan_to_watch"]
    rows = [r for r in rows if filters.keep(r)]
    spec = SPECS[surface]
    if surface in RERANKED:
        by_id = {r["mal_id"]: r for r in rows}
        cands = []
        for r in rows:
            rel = float(relevance_bonus(_Share(r["personal_share"]),
                                        np.array([r["relevance_z"]]))[0])
            cands.append(Candidate(
                mal_id=r["mal_id"], title=r["title"], franchise_id=r["franchise_id"],
                predicted=r["predicted"], novelty=r["novelty"], final=r["predicted"] + rel,
                relevance=rel, relevance_z=r["relevance_z"],
                personal_share=r["personal_share"], bonus=r["bonus"] or 0.0,
                row={"mal_genres": r["mal_genres"]}))
        # same novelty scale and cut-off as the build, so an unfiltered read
        # reproduces the stored list exactly
        scale = rows[0]["novelty_scale"] if rows else 1.0
        ranked = rerank(cands, spec.novelty_weight * scale, spec.diversity_weight,
                        limit=max(offset + limit, rows[0]["build_limit"]), scale_novelty=False,
                        profile=genre_profile(user_id, filters.exclude_genres))
        ordered = [(by_id[c.mal_id], c.final) for c in ranked]
    else:
        ordered = [(r, r["predicted"]) for r in rows]
    out = []
    for i, (r, final) in enumerate(ordered[offset:offset + limit], offset + 1):
        out.append({"rank": i, "mal_id": r["mal_id"], "title": r["title"],
                    "title_en": r["title_en"], "media_type": r["media_type"],
                    "num_episodes": r["num_episodes"], "season_year": r["season_year"],
                    "season": r["season"], "mal_mean": r["mal_mean"],
                    "mal_popularity": r["mal_popularity"],
                    "picture_medium": r["picture_medium"], "picture_large": r["picture_large"],
                    "synopsis": r["synopsis"], "mal_genres": r["mal_genres"],
                    "mal_studios": r["mal_studios"],
                    "predicted_score": r["shown"] if r["shown"] is not None else r["predicted"],
                    "final_score": float(final), "novelty": r["novelty"],
                    "reasons": r["reasons"], "generated_at": r["generated_at"],
                    "list_status": r["list_status"], "list_score": r["list_score"]})
    return with_likely(user_id, out)
