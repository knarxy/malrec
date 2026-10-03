"""HTTP surface of the web app.

Every endpoint says who may call it (see malrec.access):
  public     /health (just "ok") and the sign-in flow in malrec.auth
  approved   a signed-in, admin-approved account - its own data only;
             the admin may name any profile with ?user=
  admin      malrec.admin (the admin panel's API)
Catalogue syncs, retraining and evaluation are CLI-only (`malrec ...`): they
cost MAL requests or minutes of CPU and have no business being one HTTP call
away.
"""
from __future__ import annotations

import logging

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import admin, auth, onboarding, tasks, together
from . import feedback as fb
from .access import Approved, rate_limit, target_user
from .auth import require_csrf
from .config import settings
from .db import one, query, scalar
from .model import cached_model, feature_importance, load_latest
from .rank import familiar_genres, relevance_bonus
from .surfaces import SPECS, SURFACES, Filters, _Share, build_surface, read_ranked

log = logging.getLogger(__name__)

_docs = settings().enable_docs
app = FastAPI(
    title="malrec",
    version="0.2.0",
    description="Personalised anime recommendations from a MyAnimeList profile.",
    docs_url="/docs" if _docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if _docs else None,
)

if _docs:
    # only the Vite dev server is a separate origin; in deployment the app
    # and the API share one origin through nginx
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"],
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(together.router)

CSRF = [Depends(require_csrf)]


# --------------------------------------------------------------- onboarding --

@app.get("/onboard/{username}")
def onboard_status(username: str, v: Approved) -> dict:
    """Progress of the first-run setup (own profile, or any for the admin)."""
    if not onboarding.valid_username(username):
        raise HTTPException(422, "invalid username")
    uid = target_user(v, username)
    name = one("SELECT mal_username FROM app_user WHERE id=%s", (uid,))["mal_username"]
    job = onboarding.current(name)
    has_recs = bool(scalar("SELECT count(*) FROM recommendation WHERE user_id=%s", (uid,)))
    if (not job or job["state"] != "running") and tasks.pending(uid, "onboard"):
        # queued for the worker: report it as starting, not as a past result
        return {"state": "running", "step": "queued", "step_index": 0, "step_total": 6,
                "error": None, "ready": has_recs, "username": name}
    if not job:
        return {"state": "absent", "ready": has_recs, "username": name}
    return {"state": job["state"], "step": job["step"], "step_index": job["step_index"],
            "step_total": job["step_total"], "error": job["error"], "ready": has_recs,
            "username": name}


# ------------------------------------------------------------------- health --

@app.get("/health")
def health() -> dict:
    """Liveness only; details are in the admin panel."""
    scalar("SELECT 1")
    return {"status": "ok"}


# ----------------------------------------------------------- recommendations --

@app.get("/surfaces")
def surfaces(v: Approved) -> list[dict]:
    return [{"name": s, "description": SPECS[s].description} for s in SURFACES]


def _csv(v: str | None) -> list[str]:
    return [x.strip()[:40] for x in (v or "").split(",") if x.strip()][:40]


@app.get("/recommendations/{surface}", dependencies=[rate_limit("recs", 240, 60)])
def recommendations(
    surface: str,
    v: Approved,
    user: str | None = None,
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0, le=1000),
    types: str | None = Query(None, max_length=200),
    eps_min: int | None = Query(None, ge=0, le=10000),
    eps_max: int | None = Query(None, ge=0, le=10000),
    year_min: int | None = Query(None, ge=1900, le=2100),
    year_max: int | None = Query(None, ge=1900, le=2100),
    exclude_genres: str | None = Query(None, max_length=1000),
) -> dict:
    if surface not in SURFACES:
        raise HTTPException(404, f"unknown surface {surface!r}")
    uid = target_user(v, user)
    filters = Filters(_csv(types), eps_min, eps_max, year_min, year_max, _csv(exclude_genres))
    items = read_ranked(uid, surface, limit, offset, filters)
    if not items and not scalar(
            "SELECT count(*) FROM rec_candidate WHERE user_id=%s AND surface=%s", (uid, surface)):
        build_surface(uid, surface, limit=max(limit + offset, 50))
        items = read_ranked(uid, surface, limit, offset, filters)
    return {"surface": surface, "description": SPECS[surface].description,
            "count": len(items), "items": items}


@app.get("/explain/{mal_id}", dependencies=[rate_limit("explain", 120, 60)])
def explain(mal_id: int, v: Approved, user: str | None = None, surface: str = "discover",
            on_list: bool = Query(True, alias="list")) -> dict:
    """The full "Why this?" breakdown for one title."""
    uid = target_user(v, user)
    model, _ = cached_model(uid)
    if not hasattr(model, "breakdown"):
        raise HTTPException(409, "No breakdown available for this profile.")
    out = model.breakdown(mal_id, on_list=on_list)
    if out is None:
        raise HTTPException(404, "No prediction for that title.")
    cand = one("SELECT predicted, relevance_z, novelty, novelty_scale, personal_share, bonus,"
               " pop_bonus, risk FROM rec_candidate"
               " WHERE user_id=%s AND mal_id=%s ORDER BY surface=%s DESC LIMIT 1",
               (uid, mal_id, surface))
    if cand is not None and surface in SPECS:
        rel = float(relevance_bonus(_Share(cand["personal_share"]),
                                    np.array([cand["relevance_z"]]))[0])
        nov = (SPECS[surface].novelty_weight * float(cand["novelty_scale"])
               * float(cand["novelty"]))
        bonus, pop_b = float(cand["bonus"] or 0.0), float(cand["pop_bonus"] or 0.0)
        risk = float(cand["risk"] or 0.0)
        # The list is sorted by this order score, not by the predicted score
        # alone; every term that went into it is reported, so the placement of
        # two neighbours can always be read off the panel.
        out["ranking"] = {"relevance_z": round(float(cand["relevance_z"]), 2),
                          "relevance_bonus": round(rel, 3),
                          "novelty": round(float(cand["novelty"]), 2),
                          "novelty_bonus": round(nov, 3),
                          "memory_bonus": round(bonus - pop_b + risk, 3),
                          "risk_penalty": round(-risk, 3),
                          "popularity_bonus": round(pop_b, 3),
                          "predicted": round(float(cand["predicted"]), 2),
                          "order_score": round(float(cand["predicted"]) + rel + bonus + nov, 2)}
        if risk > 0.005 and hasattr(model, "risk_parts"):
            # the stored penalty, split by reason in the proportions the
            # current model gives (scaled, so the parts add up to it)
            parts = {k: float(v[0]) for k, v in
                     model.risk_parts([mal_id], familiar_genres(uid)).items()}
            total = sum(parts.values())
            if total > 1e-9:
                out["ranking"]["risk_parts"] = {k: round(-risk * v / total, 3)
                                                for k, v in parts.items() if v > 1e-9}
    return out


@app.get("/genres")
def genres(v: Approved) -> list[str]:
    """For the filter bar."""
    return [r["g"] for r in query(
        "SELECT g FROM (SELECT unnest(mal_genres) g FROM anime"
        " WHERE mal_num_scoring_users >= 300) x GROUP BY g HAVING count(*) >= 20 ORDER BY g")]


@app.post("/recommendations/rebuild", dependencies=CSRF + [rate_limit("rebuild", 6, 600)])
def rebuild(v: Approved) -> dict:
    """Rebuild the caller's own lists."""
    tasks.enqueue("rebuild", v.user_id)
    return {"status": "rebuilding"}


@app.get("/profile")
def profile(v: Approved, user: str | None = None) -> dict:
    """Headline numbers plus the taste summary the app shows on the overview."""
    uid = target_user(v, user)
    name = one("SELECT mal_username FROM app_user WHERE id=%s", (uid,))["mal_username"]
    stats = one(
        """
        SELECT count(*) AS entries,
               count(*) FILTER (WHERE score > 0) AS scored,
               count(*) FILTER (WHERE status = 'completed') AS completed,
               round(avg(score) FILTER (WHERE score > 0)::numeric, 2) AS mean_score
          FROM list_entry WHERE user_id = %s
        """,
        (uid,),
    )
    top_genres = query(
        """
        WITH mu AS (SELECT avg(score)::real m FROM list_entry
                     WHERE user_id=%(uid)s AND score>0),
        g AS (SELECT unnest(a.mal_genres) AS genre, le.score
                FROM list_entry le JOIN anime a ON a.mal_id = le.mal_id
               WHERE le.user_id=%(uid)s AND le.score>0)
        SELECT genre, count(*) AS n,
               round((avg(g.score) - (SELECT m FROM mu))::numeric, 2) AS lift
          FROM g GROUP BY genre HAVING count(*) >= 4
         ORDER BY lift DESC LIMIT 8
        """,
        {"uid": uid},
    )
    return {"user": name, **(stats or {}), "top_genres": top_genres}


@app.get("/anime/{mal_id}")
def anime_detail(mal_id: int, v: Approved, user: str | None = None) -> dict:
    """Catalogue facts plus the viewer's list status, score and predicted
    score (for opening any title, e.g. from search, like a card opens)."""
    row = one(
        """
        SELECT a.mal_id, a.title, a.title_en, a.media_type, a.num_episodes, a.status,
               a.season_year, a.season, a.start_date, a.mal_mean, a.mal_popularity,
               a.mal_num_scoring_users, a.picture_medium, a.picture_large, a.synopsis,
               a.mal_genres, a.mal_studios, a.source, a.rating,
               coalesce(f.franchise_id, a.mal_id) AS franchise_id,
               (SELECT array_agg(tag ORDER BY rank DESC)
                  FROM anime_tag t WHERE t.mal_id = a.mal_id AND t.rank >= 50) AS tags
          FROM anime a LEFT JOIN franchise f ON f.mal_id = a.mal_id
         WHERE a.mal_id = %s
        """,
        (mal_id,),
    )
    if not row:
        raise HTTPException(404, "unknown anime")
    uid = target_user(v, user)
    le = one("SELECT status, score FROM list_entry WHERE user_id=%s AND mal_id=%s",
             (uid, mal_id))
    row["list_status"] = le["status"] if le else None
    row["list_score"] = le["score"] if le else None
    row["predicted_score"] = None
    try:
        model, _ = cached_model(uid)
    except ValueError:
        model = None
    if model is not None and hasattr(model, "predict_ids"):
        raw = model.predict_ids([mal_id])
        if not np.isnan(raw[0]):
            row["predicted_score"] = round(float(model.calibrate(raw)[0]), 2)
            from .uncertainty import band, user_quantiles
            q, idx = user_quantiles(uid)
            row["likely"] = band(row["predicted_score"], q, idx)
    return row


@app.get("/search", dependencies=[rate_limit("search", 120, 60)])
def search(v: Approved, q: str = Query(min_length=2, max_length=100),
           limit: int = Query(20, ge=1, le=50)) -> list[dict]:
    """Trigram + full-text, so partial and misspelt titles still land."""
    return query(
        """
        SELECT mal_id, title, title_en, media_type, season_year, mal_mean, picture_medium,
               similarity(title, %(q)s) AS sim
          FROM anime
         WHERE title %% %(q)s OR title_en %% %(q)s
            OR search_tsv @@ plainto_tsquery('simple', %(q)s)
         ORDER BY sim DESC NULLS LAST, mal_num_scoring_users DESC NULLS LAST
         LIMIT %(limit)s
        """,
        {"q": q, "limit": limit},
    )


@app.get("/similar/{mal_id}", dependencies=[rate_limit("similar", 120, 60)])
def similar(mal_id: int, v: Approved, limit: int = Query(20, ge=1, le=50)) -> list[dict]:
    """Nearest neighbours in AniList tag space - the 'more like this' control."""
    return query(
        """
        SELECT a.mal_id, a.title, a.mal_mean, a.picture_medium,
               1 - (a.tag_vec <=> src.tag_vec) AS similarity
          FROM anime a, (SELECT tag_vec FROM anime WHERE mal_id = %s) src
         WHERE a.tag_vec IS NOT NULL AND src.tag_vec IS NOT NULL AND a.mal_id <> %s
         ORDER BY a.tag_vec <=> src.tag_vec
         LIMIT %s
        """,
        (mal_id, mal_id, limit),
    )


# ---------------------------------------------------------------- feedback --

class FeedbackIn(BaseModel):
    mal_id: int
    action: str = Field(description=f"one of {fb.ACTIONS}")
    surface: str | None = Field(None, max_length=40)


@app.post("/feedback", dependencies=CSRF + [rate_limit("feedback", 120, 600)])
def post_feedback(body: FeedbackIn, v: Approved) -> dict:
    """The caller's own feedback ("Not for me")."""
    if body.action not in ("not_interested", "hidden", "seen_it", "liked", "clicked"):
        raise HTTPException(422, "unsupported action")
    try:
        return fb.record(v.user_id, body.mal_id, body.action, body.surface)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.delete("/feedback/{mal_id}", dependencies=CSRF + [rate_limit("feedback", 120, 600)])
def delete_feedback(mal_id: int, v: Approved) -> dict:
    return {"removed": fb.undo(v.user_id, mal_id)}


@app.get("/feedback")
def get_feedback(v: Approved, limit: int = Query(100, ge=1, le=500)) -> dict:
    return {"summary": fb.summary(v.user_id), "history": fb.history(v.user_id, limit)}


# ------------------------------------------------------------------- model --

@app.get("/model")
def model_info(v: Approved, user: str | None = None) -> dict:
    uid = target_user(v, user)
    model, run_id = load_latest(uid)
    if model is None:
        raise HTTPException(404, "no model trained yet")
    return {"run_id": run_id, "algo": model.algo, "params": model.params,
            "metrics": model.metrics, "baseline": model.baseline,
            "top_features": feature_importance(model, 25)}


_learning_cache: dict = {"at": 0.0, "value": None}
LEARNING_TTL = 600.0


@app.get("/model/learning")
def model_learning(v: Approved) -> dict:
    """What in-app actions say the ranking weights should be (aggregate over
    all users, no personal data). Suggestions only - see malrec.learn. Kept for
    ten minutes: it scans every impression and fits ~100 regressions."""
    import time

    from .learn import learn_weights
    if _learning_cache["value"] is None or time.time() - _learning_cache["at"] > LEARNING_TTL:
        _learning_cache.update(at=time.time(), value=learn_weights(boot=100))
    return _learning_cache["value"]
