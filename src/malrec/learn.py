"""Learning the ranking weights from what people do in the app.

Every surface build logs what was shown with its ranking inputs (rec_log).
Outcomes come afterwards, from explicit actions only - in the app or on
MyAnimeList itself (seen by the next list sync):

    positive   queued it, put it on Plan to Watch, started it, completed it
               unscored, or rated it at or above their own mean
    negative   "Not for me", dropped it, or rated it more than a point below
               their mean

A title that was shown and ignored is not a negative: most of a 50-item list
is never even scrolled to, so silence says little. A logistic regression of
outcome on (predicted score, relevance, novelty) then says how many rating
points a standard deviation of relevance, or full novelty, is actually worth
to people - the exchange rate the ranking hard-codes today. With few events
the answer is noise, so nothing is suggested below `min_events`.
"""
from __future__ import annotations

import logging

import numpy as np

from .config import settings
from .db import query

log = logging.getLogger(__name__)

# Only what a discovery list put in front of the user: titles in their own
# Plan to Watch, continuations and announcements are shown because they chose
# them already, so acting on those says nothing about the ranking.
DISCOVERY = ("safe_bets", "discover", "hidden_gems", "this_season")

# Outcomes read the list as it is now, so an action taken on MyAnimeList
# itself counts exactly like one taken in the app once a sync has seen it.
OUTCOMES_SQL = """
    WITH shown AS (
        SELECT DISTINCT ON (l.user_id, l.mal_id) l.user_id, l.mal_id, l.predicted,
               l.relevance_z, l.novelty, l.personal_share, l.shown_at
          FROM rec_log l
         WHERE l.surface = ANY(%(surfaces)s)
         ORDER BY l.user_id, l.mal_id, l.shown_at),
    mu AS (SELECT user_id, avg(score) m FROM list_entry WHERE score > 0 GROUP BY user_id)
    SELECT s.*,
           EXISTS (SELECT 1 FROM feedback f WHERE f.user_id = s.user_id AND f.mal_id = s.mal_id
                     AND f.action = 'queued' AND f.created_at >= s.shown_at)
           OR coalesce(le.status IN ('plan_to_watch', 'watching')
                       OR (le.status = 'completed' AND le.score = 0)
                       OR (le.score > 0 AND le.score >= mu.m), false) AS positive,
           EXISTS (SELECT 1 FROM feedback f WHERE f.user_id = s.user_id AND f.mal_id = s.mal_id
                     AND f.action = 'not_interested' AND f.created_at >= s.shown_at)
           OR coalesce(le.status = 'dropped'
                       OR (le.score > 0 AND le.score < mu.m - 1), false) AS negative
      FROM shown s
      LEFT JOIN list_entry le ON le.user_id = s.user_id AND le.mal_id = s.mal_id
                             AND le.updated_at >= s.shown_at
      LEFT JOIN mu ON mu.user_id = s.user_id
"""


def learn_weights(min_events: int = 200, min_each: int = 30, boot: int = 200) -> dict:
    from sklearn.linear_model import LogisticRegression

    rows = [r for r in query(OUTCOMES_SQL, {"surfaces": list(DISCOVERY)})
            if r["positive"] != r["negative"]]
    pos = sum(r["positive"] for r in rows)
    neg = len(rows) - pos
    out = {"events": len(rows), "positive": pos, "negative": neg,
           "users": len({r["user_id"] for r in rows})}
    if len(rows) < min_events or min(pos, neg) < min_each:
        out["status"] = (f"not enough feedback yet: need {min_events} events with at least "
                         f"{min_each} of each outcome")
        return out

    X = np.array([[r["predicted"], r["relevance_z"], r["novelty"]] for r in rows], dtype=float)
    y = np.array([r["positive"] for r in rows], dtype=int)

    def rates(Xs, ys):
        c = LogisticRegression(C=1.0).fit(Xs, ys).coef_[0]
        if c[0] <= 1e-6:            # score must matter, or there is no exchange rate
            return None
        return c[1] / c[0], c[2] / c[0]

    point = rates(X, y)
    rng = np.random.default_rng(0)
    samples = [r for r in (rates(X[i], y[i]) for i in
                           (rng.integers(0, len(y), len(y)) for _ in range(boot))) if r]
    cfg = settings()
    if point is None or len(samples) < boot // 2:
        out["status"] = "predicted score does not explain the outcomes; no exchange rate"
        return out
    rel = np.array([s[0] for s in samples])
    nov = np.array([s[1] for s in samples])
    out.update({
        "status": "ok",
        "relevance_points_per_sd": round(point[0], 3),
        "relevance_ci90": [round(float(np.quantile(rel, q)), 3) for q in (0.05, 0.95)],
        "novelty_points": round(point[1], 3),
        "novelty_ci90": [round(float(np.quantile(nov, q)), 3) for q in (0.05, 0.95)],
        "current": {"relevance_weight": cfg.relevance_weight,
                    "relevance_weight_personal": cfg.relevance_weight_personal,
                    "novelty_weight": cfg.novelty_weight},
        "note": "Suggestions only. Change a weight when its interval excludes the current "
                "value, then re-run exp_final to confirm nothing regresses.",
    })
    return out
