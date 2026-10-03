"""Who is asking, and what they may see.

Everything except sign-in and a bare health check needs a MAL session. An
account starts 'pending' and sees nothing but its own status until the admin
approves it; 'rejected' and 'blocked' accounts cannot sign in at all. Users
only ever see their own data; the admin (config.admin_users - deliberately not
a database flag, so no request can grant it) may look at anyone's.

Rate limits are in-process and per client address, enough to stop a single
browser or script from hammering MAL through the app. They key on the address,
not the session cookie: a client chooses its own cookie, so a fresh made-up
value per request would otherwise get a fresh budget (security audit
2026-10-03).
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request

from .config import settings
from .db import one


@dataclass
class Viewer:
    user_id: int
    username: str
    status: str
    is_admin: bool
    session: dict


def is_admin_name(username: str) -> bool:
    admins = {a.strip().lower() for a in settings().admin_users.split(",") if a.strip()}
    return username.lower() in admins


def viewer(request: Request) -> Viewer | None:
    from .auth import current_session
    s = current_session(request)
    if s is None:
        return None
    u = one("SELECT status FROM app_user WHERE id=%s", (s["app_user_id"],))
    if u is None or u["status"] in ("rejected", "blocked"):
        return None
    return Viewer(s["app_user_id"], s["mal_username"], u["status"],
                  is_admin_name(s["mal_username"]) and u["status"] == "approved", s)


def require_signed_in(request: Request) -> Viewer:
    v = viewer(request)
    if v is None:
        raise HTTPException(401, "Sign in with MyAnimeList to do that.")
    return v


def require_approved(request: Request) -> Viewer:
    v = require_signed_in(request)
    if v.status != "approved":
        raise HTTPException(403, {"message": "Your account is waiting for approval.",
                                  "status": v.status})
    return v


def require_admin(request: Request) -> Viewer:
    v = require_approved(request)
    if not v.is_admin:
        raise HTTPException(403, "Admins only.")
    return v


SignedIn = Annotated[Viewer, Depends(require_signed_in)]
Approved = Annotated[Viewer, Depends(require_approved)]
Admin = Annotated[Viewer, Depends(require_admin)]


def target_user(v: Viewer, username: str | None) -> int:
    """The profile a request is about: the viewer's own, or - for the admin
    only - any approved profile named by `username`."""
    if not username or username.lower() == v.username.lower():
        return v.user_id
    if not v.is_admin:
        raise HTTPException(403, "You can only see your own recommendations.")
    row = one("SELECT id FROM app_user WHERE lower(mal_username) = lower(%s)", (username,))
    if row is None:
        raise HTTPException(404, "unknown user")
    return row["id"]


# ------------------------------------------------------------ rate limits --

_lock = threading.Lock()
_hits: dict[tuple[str, str], deque] = defaultdict(deque)
# Buckets of clients that went quiet are dropped once the table grows past
# this, so a stream of distinct addresses cannot grow it without bound.
_SWEEP_AT = 10_000
_MAX_WINDOW = 3600.0


def client_key(request: Request) -> str:
    """The real client address behind two proxies (a reverse proxy, then the app's nginx):
    each appends to X-Forwarded-For, so the client is second from the right.
    Anything further left is client-supplied and not trusted."""
    xff = [p.strip() for p in (request.headers.get("x-forwarded-for") or "").split(",") if p.strip()]
    if len(xff) >= 2:
        return xff[-2]
    return xff[-1] if xff else (request.client.host if request.client else "?")


def rate_limit(bucket: str, limit: int, window: float):
    """Dependency: at most `limit` requests per `window` seconds per client."""
    def dep(request: Request) -> None:
        who = client_key(request)
        now = time.monotonic()
        with _lock:
            if len(_hits) > _SWEEP_AT:
                for k in [k for k, d in _hits.items() if not d or now - d[-1] > _MAX_WINDOW]:
                    del _hits[k]
            q = _hits[(bucket, who)]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                raise HTTPException(429, "Too many requests - slow down a little.",
                                    headers={"Retry-After": str(int(window))})
            q.append(now)
    return Depends(dep)
