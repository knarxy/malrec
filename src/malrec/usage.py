"""What signed-in users actually see and open in the app, and the admin's
funnel built from it.

The app reports a few kinds of event in small batches (POST /me/events):
the app opened, a tab viewed, a card on screen for a second, a title's
details opened. Only the account's own lists count - an admin looking at
someone else's profile reports nothing. Each thing is one row per day with a
counter (app_event), kept 180 days and deleted with the account. Actions -
ratings, Plan to Watch, Not for me - are in feedback already.

Why: the lists that are built say nothing about what anybody looked at, so
neither "why does a new user do nothing?" nor "did a ranking change help?"
could be answered from data.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from .access import rate_limit
from .auth import ApprovedSession, require_csrf
from .db import conn, execute, query, scalar
from .surfaces import SURFACES

router = APIRouter()

KINDS = ("app_open", "tab_view", "card_seen", "details_open")
PLACES = frozenset(SURFACES) | {"search", ""}
KEEP_DAYS = 180
MAX_BATCH = 60


class EventIn(BaseModel):
    kind: Literal["app_open", "tab_view", "card_seen", "details_open"]
    surface: str = Field("", max_length=40)
    mal_id: int = Field(0, ge=0)
    rank: int | None = Field(None, ge=0, le=1000)


class EventsIn(BaseModel):
    events: list[EventIn] = Field(max_length=MAX_BATCH)


UPSERT = """
    INSERT INTO app_event (user_id, kind, surface, mal_id, rank)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (user_id, kind, surface, mal_id, day)
    DO UPDATE SET n = app_event.n + 1, last_at = now(),
                  rank = coalesce(EXCLUDED.rank, app_event.rank)
"""


@router.post("/me/events", dependencies=[Depends(require_csrf), rate_limit("events", 120, 600)])
def record(body: EventsIn, s: ApprovedSession) -> dict:
    uid = s["app_user_id"]
    rows = [(uid, e.kind, e.surface if e.surface in PLACES else "", e.mal_id, e.rank)
            for e in body.events]
    if rows:
        with conn() as c, c.cursor() as cur:
            cur.executemany(UPSERT, rows)
    return {"recorded": len(rows)}


def prune() -> int:
    """Housekeeping: forget events older than KEEP_DAYS."""
    return execute("DELETE FROM app_event WHERE day < current_date - %s", (KEEP_DAYS,)) or 0


# ------------------------------------------------------------------ funnel --

STEPS = ("signed_up", "approved", "list_loaded", "recs_ready", "visited", "saw_cards",
         "opened_details", "acted")

USERS_SQL = """
    SELECT u.id, u.mal_username, u.status,
           coalesce(u.requested_at, u.created_at) AS signed_up, u.approved_at,
           (SELECT count(*) FROM list_entry le WHERE le.user_id = u.id) AS entries,
           (SELECT count(*) FROM recommendation r WHERE r.user_id = u.id) AS recs,
           (SELECT min(e.first_at) FROM app_event e WHERE e.user_id = u.id) AS first_visit,
           (SELECT max(e.last_at) FROM app_event e WHERE e.user_id = u.id) AS last_visit,
           (SELECT min(e.first_at) FROM app_event e
             WHERE e.user_id = u.id AND e.kind = 'card_seen') AS first_card,
           (SELECT min(e.first_at) FROM app_event e
             WHERE e.user_id = u.id AND e.kind = 'details_open') AS first_details,
           (SELECT min(f.created_at) FROM feedback f WHERE f.user_id = u.id) AS first_action,
           (SELECT count(DISTINCT e.day) FROM app_event e
             WHERE e.user_id = u.id AND e.day > current_date - 7) AS days_7,
           (SELECT count(DISTINCT e.mal_id) FROM app_event e
             WHERE e.user_id = u.id AND e.kind = 'card_seen' AND e.day > current_date - 7) AS cards_7,
           (SELECT count(*) FROM app_event e
             WHERE e.user_id = u.id AND e.kind = 'details_open' AND e.day > current_date - 7) AS details_7,
           (SELECT count(*) FROM feedback f
             WHERE f.user_id = u.id AND f.created_at > now() - interval '7 days') AS actions_7
      FROM app_user u
     ORDER BY coalesce(u.requested_at, u.created_at) DESC NULLS LAST, u.id
"""


def funnel() -> dict:
    """Per account: how far it got (signed up -> approved -> list loaded ->
    lists built -> opened the app -> saw cards -> opened details -> acted),
    plus the last seven days. Events exist only since logging began
    (`since`); an account that was active before that shows no visit."""
    users = query(USERS_SQL)
    for u in users:
        u["steps"] = {
            "signed_up": True,
            "approved": u["status"] == "approved",
            "list_loaded": u["entries"] > 0,
            "recs_ready": u["recs"] > 0,
            "visited": u["first_visit"] is not None,
            "saw_cards": u["first_card"] is not None,
            "opened_details": u["first_details"] is not None,
            "acted": u["first_action"] is not None,
        }
    counts = [{"step": s, "users": sum(1 for u in users if u["steps"][s])} for s in STEPS]
    since = scalar("SELECT min(first_at) FROM app_event")
    return {"since": since, "steps": counts, "users": users}
