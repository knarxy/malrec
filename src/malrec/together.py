"""Watch together: recommendations for two approved accounts at once.

Consent first. A pair exists only once the invited account has accepted, and
either side can end it. An invitation can also be a link (single use, valid
LINK_DAYS days, only its hash stored): opening it signed in shows who sent it,
and accepting creates the accepted pair directly. The shared list reveals nothing but two predicted
scores per title - never either person's list or ratings. Inviting answers
the same whether or not the name belongs to an account, so the field cannot
be used to find out who uses the app.

The list: titles on neither person's list (Plan to Watch is fine), from both
people's scored shortlists, predicted for each; ranked by the mean of the two
predictions, and a title is dropped when either person would likely not enjoy
it (predicted more than MISERY points below their own average) - the "average
without misery" rule from the group-recommendation literature.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import secrets

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .access import Approved, rate_limit
from .auth import require_csrf
from .config import settings
from .db import execute, one, query

router = APIRouter(prefix="/together")
CSRF = [Depends(require_csrf)]
MISERY = 0.5
LINK_DAYS = 7
MAX_LINKS = 5        # open links per person; creating one more retires the oldest


class InviteIn(BaseModel):
    username: str = Field(min_length=2, max_length=24, pattern=r"^[A-Za-z0-9_-]+$")


@router.get("")
def overview(v: Approved) -> dict:
    rows = query(
        """SELECT p.id, p.status, p.inviter = %(me)s AS outgoing,
                  u.mal_username AS other
             FROM together_pair p
             JOIN app_user u ON u.id = CASE WHEN p.inviter = %(me)s THEN p.invitee ELSE p.inviter END
            WHERE (p.inviter = %(me)s OR p.invitee = %(me)s) AND u.status = 'approved'
            ORDER BY p.status, u.mal_username""", {"me": v.user_id})
    return {"partners": [r for r in rows if r["status"] == "accepted"],
            "incoming": [r for r in rows if r["status"] == "pending" and not r["outgoing"]],
            "outgoing": [r for r in rows if r["status"] == "pending" and r["outgoing"]],
            "links": query("""SELECT id, created_at, expires_at FROM together_link
                               WHERE inviter=%s AND used_at IS NULL AND expires_at > now()
                               ORDER BY id DESC""", (v.user_id,))}


# ------------------------------------------------------------------ links --

def _hash(token: str) -> bytes:
    return hashlib.sha256(token.encode()).digest()


def _open_link(token: str) -> dict:
    if not 20 <= len(token) <= 64:
        raise HTTPException(404, "This invitation link is invalid or has expired.")
    link = one("""SELECT l.id, l.inviter, u.mal_username AS inviter_name
                    FROM together_link l JOIN app_user u ON u.id = l.inviter
                   WHERE l.token_hash=%s AND l.used_at IS NULL AND l.expires_at > now()
                     AND u.status = 'approved'""", (_hash(token),))
    if link is None:
        raise HTTPException(404, "This invitation link is invalid or has expired.")
    return link


@router.post("/link", dependencies=CSRF + [rate_limit("together_link", 10, 3600)])
def create_link(v: Approved) -> dict:
    token = secrets.token_urlsafe(24)
    expires = dt.datetime.now(dt.UTC) + dt.timedelta(days=LINK_DAYS)
    execute("INSERT INTO together_link (inviter, token_hash, expires_at) VALUES (%s,%s,%s)",
            (v.user_id, _hash(token), expires))
    execute("""DELETE FROM together_link WHERE inviter=%s AND id NOT IN (
                 SELECT id FROM together_link WHERE inviter=%s AND used_at IS NULL
                    AND expires_at > now() ORDER BY id DESC LIMIT %s)""",
            (v.user_id, v.user_id, MAX_LINKS))
    base = settings().app_base_url.rstrip("/")
    return {"url": f"{base}/?together={token}", "expires_at": expires.isoformat()}


@router.get("/link/{token}", dependencies=[rate_limit("together_link_open", 30, 600)])
def peek_link(token: str, v: Approved) -> dict:
    link = _open_link(token)
    return {"inviter": link["inviter_name"], "own": link["inviter"] == v.user_id}


@router.post("/link/{token}/accept",
             dependencies=CSRF + [rate_limit("together_link_open", 30, 600)])
def accept_link(token: str, v: Approved) -> dict:
    link = _open_link(token)
    if link["inviter"] == v.user_id:
        raise HTTPException(409, "This is your own invitation link.")
    used = execute("UPDATE together_link SET used_by=%s, used_at=now()"
                   " WHERE id=%s AND used_at IS NULL", (v.user_id, link["id"]))
    if not used:
        raise HTTPException(404, "This invitation link is invalid or has expired.")
    execute("""INSERT INTO together_pair (inviter, invitee, status, accepted_at)
               VALUES (%s, %s, 'accepted', now())
               ON CONFLICT ((least(inviter, invitee)), (greatest(inviter, invitee)))
               DO UPDATE SET status='accepted', accepted_at=coalesce(together_pair.accepted_at, now())""",
            (link["inviter"], v.user_id))
    pair = one("""SELECT id FROM together_pair
                   WHERE least(inviter, invitee)=least(%s::int, %s::int)
                     AND greatest(inviter, invitee)=greatest(%s::int, %s::int)""",
               (link["inviter"], v.user_id, link["inviter"], v.user_id))
    return {"id": pair["id"], "other": link["inviter_name"], "status": "accepted"}


@router.delete("/link/{lid}", dependencies=CSRF)
def withdraw_link(lid: int, v: Approved) -> dict:
    execute("DELETE FROM together_link WHERE id=%s AND inviter=%s", (lid, v.user_id))
    return {"id": lid, "withdrawn": True}


@router.post("/invite", dependencies=CSRF + [rate_limit("together_invite", 20, 3600)])
def invite(body: InviteIn, v: Approved) -> dict:
    other = one("SELECT id FROM app_user WHERE lower(mal_username) = lower(%s)"
                " AND status = 'approved'", (body.username,))
    if other and other["id"] != v.user_id:
        execute("INSERT INTO together_pair (inviter, invitee) VALUES (%s, %s)"
                " ON CONFLICT DO NOTHING", (v.user_id, other["id"]))
    # same answer either way: the invite form must not reveal who is registered
    return {"status": "sent"}


def _pair(pid: int, v) -> dict:
    p = one("SELECT * FROM together_pair WHERE id=%s AND (inviter=%s OR invitee=%s)",
            (pid, v.user_id, v.user_id))
    if p is None:
        raise HTTPException(404, "unknown invitation")
    return p


@router.post("/{pid}/accept", dependencies=CSRF)
def accept(pid: int, v: Approved) -> dict:
    p = _pair(pid, v)
    if p["invitee"] != v.user_id:
        raise HTTPException(403, "Only the invited person can accept.")
    execute("UPDATE together_pair SET status='accepted', accepted_at=now() WHERE id=%s", (pid,))
    return {"id": pid, "status": "accepted"}


@router.delete("/{pid}", dependencies=CSRF)
def end(pid: int, v: Approved) -> dict:
    """Decline an invitation, withdraw one, or end a pair - either side."""
    _pair(pid, v)
    execute("DELETE FROM together_pair WHERE id=%s", (pid,))
    return {"id": pid, "ended": True}


def _predict(uid: int, ids: list[int]) -> tuple[np.ndarray, float]:
    """Displayed predictions for `ids` and the user's own average score."""
    from .model import cached_model
    model, _ = cached_model(uid)
    if model is None or not hasattr(model, "predict_ids"):
        raise HTTPException(409, "No recommendations built yet for one of you.")
    raw = model.predict_ids(ids)
    shown = model.calibrate_list(raw) if hasattr(model, "calibrate_list") else model.calibrate(raw)
    mean = one("SELECT avg(score)::float m FROM list_entry WHERE user_id=%s AND score>0",
               (uid,))["m"] or 7.0
    return np.asarray(shown, dtype=float), float(mean)


@router.get("/{pid}/list", dependencies=[rate_limit("together_list", 30, 600)])
def shared_list(pid: int, v: Approved, limit: int = 40) -> dict:
    p = _pair(pid, v)
    if p["status"] != "accepted":
        raise HTTPException(409, "The invitation has not been accepted yet.")
    other = p["invitee"] if p["inviter"] == v.user_id else p["inviter"]
    if one("SELECT status FROM app_user WHERE id=%s", (other,))["status"] != "approved":
        raise HTTPException(404, "unknown invitation")
    me = v.user_id
    ids = [r["mal_id"] for r in query(
        """SELECT DISTINCT c.mal_id FROM rec_candidate c
            WHERE c.user_id = ANY(%(u)s)
              AND c.surface IN ('safe_bets', 'discover', 'hidden_gems', 'this_season')
              AND NOT EXISTS (SELECT 1 FROM list_entry le WHERE le.user_id = ANY(%(u)s)
                                AND le.mal_id = c.mal_id AND le.status <> 'plan_to_watch')
              AND NOT EXISTS (SELECT 1 FROM feedback f WHERE f.user_id = ANY(%(u)s)
                                AND f.mal_id = c.mal_id AND f.action = 'not_interested')""",
        {"u": [me, other]})]
    if not ids:
        return {"partner": None, "items": []}
    a, ma = _predict(me, ids)
    b, mb = _predict(other, ids)
    ok = ~(np.isnan(a) | np.isnan(b)) & (a >= ma - MISERY) & (b >= mb - MISERY)
    score = np.where(ok, (a + b) / 2, -np.inf)
    order = [k for k in np.argsort(-score) if ok[k]][:max(1, min(limit, 60))]
    top = [ids[k] for k in order]
    meta = {r["mal_id"]: r for r in query(
        """SELECT mal_id, title, title_en, media_type, num_episodes, season_year, season,
                  mal_mean, mal_popularity, picture_medium, picture_large, mal_genres, mal_studios
             FROM anime WHERE mal_id = ANY(%s)""", (top,))}
    partner = one("SELECT mal_username FROM app_user WHERE id=%s", (other,))["mal_username"]
    items = []
    for rank, k in enumerate(order, 1):
        m = meta.get(ids[k])
        if m:
            items.append({**m, "rank": rank, "you": round(float(a[k]), 2),
                          "partner": round(float(b[k]), 2),
                          "together": round(float((a[k] + b[k]) / 2), 2)})
    return {"partner": partner, "items": items}
