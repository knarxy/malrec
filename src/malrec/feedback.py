"""Explicit feedback from the web app.

Feedback affects recommendations in two ways:
  * hard filter - `not_interested`, `hidden` and `seen_it` remove an anime from
    eligible_candidates immediately, no retrain needed.
  * soft signal - `liked` and `queued` act as extra positive training rows so
    the taste model learns from choices the user never scored on MAL.
"""
from __future__ import annotations

import logging

from .db import execute, query

log = logging.getLogger(__name__)

ACTIONS = ("not_interested", "queued", "hidden", "seen_it", "liked", "clicked")
# Actions that remove an anime from future candidate pools.
SUPPRESSING = ("not_interested", "hidden", "seen_it")
# An undo simply records the opposite; the SQL filters look at the latest row.
UNDO = {"not_interested": "clicked", "hidden": "clicked"}


def record(user_id: int, mal_id: int, action: str, surface: str | None = None) -> dict:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}; expected one of {ACTIONS}")
    execute(
        "INSERT INTO feedback (user_id, mal_id, action, surface) VALUES (%s,%s,%s,%s)",
        (user_id, mal_id, action, surface),
    )
    # Drop it from any cached surface so the web app sees the effect at once,
    # without waiting for the next rebuild.
    if action in SUPPRESSING:
        execute("DELETE FROM recommendation WHERE user_id=%s AND mal_id=%s",
                (user_id, mal_id))
        execute("DELETE FROM rec_candidate WHERE user_id=%s AND mal_id=%s",
                (user_id, mal_id))
    return {"user_id": user_id, "mal_id": mal_id, "action": action}


def undo(user_id: int, mal_id: int) -> int:
    return execute(
        "DELETE FROM feedback WHERE user_id=%s AND mal_id=%s AND action = ANY(%s)",
        (user_id, mal_id, list(SUPPRESSING)),
    )


def history(user_id: int, limit: int = 100) -> list[dict]:
    return query(
        """
        SELECT f.mal_id, a.title, f.action, f.surface, f.created_at
          FROM feedback f LEFT JOIN anime a ON a.mal_id = f.mal_id
         WHERE f.user_id = %s
         ORDER BY f.created_at DESC
         LIMIT %s
        """,
        (user_id, limit),
    )


def summary(user_id: int) -> dict:
    rows = query(
        "SELECT action, count(*) AS n FROM feedback WHERE user_id=%s GROUP BY action",
        (user_id,),
    )
    return {r["action"]: r["n"] for r in rows}


PSEUDO_SQL = """
    SELECT DISTINCT ON (f.mal_id) f.mal_id, f.action
      FROM feedback f
     WHERE f.user_id = %s AND f.action IN ('liked', 'queued')
       AND NOT EXISTS (SELECT 1 FROM list_entry le
                        WHERE le.user_id = %s AND le.mal_id = f.mal_id AND le.score > 0)
     ORDER BY f.mal_id, f.created_at DESC
"""


def pseudo_ratings(user_id: int, mean: float) -> list[tuple[int, float, float]]:
    """(mal_id, pseudo_score, weight) for in-app signals with no MAL score.

    Deliberately conservative: a thumbs-up is weaker evidence than an actual
    rating, so it enters training at a fraction of the weight and only nudges
    above the user's own mean rather than claiming a 10.
    """
    out = []
    for r in query(PSEUDO_SQL, (user_id, user_id)):
        if r["action"] == "liked":
            out.append((r["mal_id"], min(mean + 1.0, 10.0), 0.5))
        else:                                    # queued: mild interest only
            out.append((r["mal_id"], mean + 0.3, 0.25))
    return out
