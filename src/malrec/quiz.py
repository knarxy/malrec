"""The rating round: well-chosen titles a user has probably seen.

Short MAL lists are usually incomplete, and with 10-60 ratings the model has
little to go on (README, "Against simple baselines"). So a signed-in user can
rate titles they have seen but never logged; each answer is a real rating,
written to MAL like any other rating from the app.

Which title next (config.quiz_strategy, chosen by experiments/exp_elicit.py):
  popular   most-listed titles first
  smart     P(seen) x informativeness - popularity times how common the
            title is on lists like theirs, times the spread of opinion about
            it (a title everyone loves tells us little)
Titles on the list or answered before (feedback quiz_unseen / quiz_skip /
rated) are never offered.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from .config import settings
from .db import query

ASK_POOL = 1500
_lock = threading.Lock()
_stats: dict = {"at": 0.0, "ask": [], "prop": {}, "info": {}}

STATS_SQL = """
    WITH um AS (SELECT user_id, avg(score) m FROM cf_rating WHERE score > 0 GROUP BY user_id),
    lists AS (SELECT mal_id, count(*) n FROM cf_rating GROUP BY mal_id),
    spread AS (SELECT r.mal_id, stddev_pop(r.score - um.m) sd, count(*) k
                 FROM cf_rating r JOIN um USING (user_id) WHERE r.score > 0 GROUP BY r.mal_id)
    SELECT l.mal_id, l.n, s.sd
      FROM lists l JOIN spread s USING (mal_id)
      JOIN anime a ON a.mal_id = l.mal_id
     WHERE s.k >= 30 AND format_class(a.media_type) = 'main'
       AND coalesce(a.nsfw, 'white') = 'white' AND a.status <> 'not_yet_aired'
     ORDER BY l.n DESC LIMIT %s
"""


def _load_stats() -> dict:
    with _lock:
        if time.time() - _stats["at"] < 3600 and _stats["ask"]:
            return _stats
        rows = query(STATS_SQL, (ASK_POOL,))
        total = max((query("SELECT count(*) n FROM cf_user WHERE state='done'")[0]["n"]), 1)
        _stats.update(at=time.time(), ask=[r["mal_id"] for r in rows],
                      prop={r["mal_id"]: r["n"] / total for r in rows},
                      info={r["mal_id"]: float(r["sd"] or 0) for r in rows})
        return _stats


def next_cards(user_id: int, model, n: int = 3) -> list[int]:
    """The next `n` titles to ask about, best first."""
    st = _load_stats()
    done = {r["mal_id"] for r in query(
        "SELECT mal_id FROM list_entry WHERE user_id=%s UNION "
        "SELECT mal_id FROM feedback WHERE user_id=%s AND action = ANY(%s)",
        (user_id, user_id, ["quiz_unseen", "quiz_skip", "rated", "not_interested"]))}
    cand = [m for m in st["ask"] if m not in done]
    if not cand:
        return []
    if settings().quiz_strategy == "popular" or model is None or not hasattr(model, "um"):
        return cand[:n]
    rel = model.um.fold.signals(cand)["rel"]
    sd = float(np.std(rel))
    relz = (rel - rel.mean()) / sd if sd > 1e-9 else np.zeros_like(rel)
    score = (np.array([st["prop"][m] for m in cand]) * np.exp(relz)
             * np.array([st["info"][m] for m in cand]))
    return [cand[i] for i in np.argsort(-score)[:n]]
