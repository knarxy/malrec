"""Explicit feedback from the web app.

Feedback affects recommendations in two ways:
  * hard filter - `not_interested`, `hidden` and `seen_it` remove an anime from
    eligible_candidates immediately, no retrain needed.
  * soft signal - `not_interested` becomes an implicit rating under the
    user's mean (recsys.service.dismissal_examples) and takes the rest of its
    franchise out of the discovery tabs (surfaces.dismissed_franchises). The
    API queues a rebuild after it, so the lists follow within seconds.
`liked` and `queued` are recorded (learn-weights counts a queue as a
positive outcome) but do not train the model; an undo deletes the
suppressing rows (`undo`).
"""
from __future__ import annotations

import logging

from .db import execute, query

log = logging.getLogger(__name__)

ACTIONS = ("not_interested", "queued", "hidden", "seen_it", "liked", "clicked")
# Actions that remove an anime from future candidate pools.
SUPPRESSING = ("not_interested", "hidden", "seen_it")


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
