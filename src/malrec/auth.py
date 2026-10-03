"""Sign in with MyAnimeList (OAuth2 authorization code + PKCE) and the list
writes that depend on it.

Flow:
    /auth/login     -> 302 to MAL with a fresh state + PKCE verifier
    MAL             -> user approves, 302 back to /auth/callback?code&state
    /auth/callback  -> exchange code for tokens, learn who this is (@me),
                       open a session, 302 back into the app

Security choices worth knowing about:
  * tokens stay server-side; the browser gets an opaque HttpOnly cookie and
    the database stores only a hash of it;
  * `state` is single-use and expires after ten minutes, which is what stops a
    forged callback from signing someone into the attacker's account;
  * state-changing endpoints additionally require an X-Requested-With header,
    which a cross-site form cannot set and a cross-origin fetch cannot send
    without a CORS preflight we do not allow;
  * `return_to` is restricted to same-site paths, so the login cannot be used
    as an open redirect;
  * un-queueing only ever removes an entry that is still plan_to_watch. A show
    someone has since started or finished is never deleted from their list;
  * rating never changes the status of something being watched, on hold or
    dropped - only an unlisted or planned title becomes "completed";
  * a manual list sync is limited to one per SYNC_COOLDOWN per account, so the
    button cannot be used to hammer MyAnimeList.

MAL documents only the `plain` PKCE method, so the challenge is the verifier.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import secrets
from typing import Annotated
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from . import onboarding, refresh, tasks
from .access import rate_limit
from .clients.mal import MalApiError, MalClient
from .config import settings
from .db import conn, execute, one, query, scalar
from .ingest.store import get_or_create_user, set_picture
from .tokenbox import open_, seal

log = logging.getLogger(__name__)

AUTHORIZE_URL = "https://myanimelist.net/v1/oauth2/authorize"
TOKEN_URL = "https://myanimelist.net/v1/oauth2/token"
COOKIE = "malrec_session"
STATE_TTL = dt.timedelta(minutes=10)
CSRF_HEADER_VALUE = "malrec"
# One manual "sync with MyAnimeList" per account per this long. A list read is
# one or two requests, but nothing should let a button drive request volume.
SYNC_COOLDOWN = dt.timedelta(minutes=5)
LANGUAGES = ("en", "de")

router = APIRouter()


# --------------------------------------------------------------- helpers --

def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _safe_return_to(value: str | None) -> str:
    """Only same-site paths. '//evil.example' is a protocol-relative URL and
    '/\\evil.example' is treated as one by some browsers, so both are refused."""
    if not value or not value.startswith("/") or value.startswith("//") or "\\" in value:
        return "/"
    return value


def _token_request(data: dict) -> dict:
    cfg = settings()
    payload = {"client_id": cfg.mal_client_id, **data}
    if cfg.mal_client_secret:
        payload["client_secret"] = cfg.mal_client_secret
    r = httpx.post(TOKEN_URL, data=payload, timeout=20.0)
    if r.status_code != 200:
        # never log the body verbatim: on some errors it echoes the request
        log.warning("MAL token endpoint returned %s", r.status_code)
        raise HTTPException(502, "MyAnimeList refused the sign-in")
    return r.json()


def _expiry(expires_in: int | None) -> dt.datetime:
    return dt.datetime.now(dt.UTC) + dt.timedelta(seconds=int(expires_in or 3600))


def _refresh(row: dict) -> dict | None:
    try:
        tok = _token_request({"grant_type": "refresh_token",
                              "refresh_token": row["refresh_token"]})
    except HTTPException:
        execute("DELETE FROM mal_session WHERE token_hash=%s", (row["token_hash"],))
        return None
    execute(
        "UPDATE mal_session SET access_token=%s, refresh_token=%s, expires_at=%s"
        " WHERE token_hash=%s",
        (seal(tok["access_token"]), seal(tok.get("refresh_token", row["refresh_token"])),
         _expiry(tok.get("expires_in")), row["token_hash"]),
    )
    row = dict(row)
    row["access_token"] = tok["access_token"]
    return row


def current_session(request: Request) -> dict | None:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    row = one(
        "SELECT s.*, u.mal_username FROM mal_session s JOIN app_user u ON u.id = s.app_user_id"
        " WHERE s.token_hash = %s", (_hash(token),))
    if not row:
        return None
    row["access_token"] = open_(row["access_token"])
    row["refresh_token"] = open_(row["refresh_token"])
    now = dt.datetime.now(dt.UTC)
    if row["created_at"] < now - dt.timedelta(days=settings().session_days):
        execute("DELETE FROM mal_session WHERE token_hash=%s", (row["token_hash"],))
        return None
    if row["expires_at"] < now + dt.timedelta(minutes=2):
        row = _refresh(row)
        if row is None:
            return None
    execute("UPDATE mal_session SET last_used_at=now() WHERE token_hash=%s",
            (row["token_hash"],))
    return row


def require_session(request: Request) -> dict:
    s = current_session(request)
    if s is None:
        raise HTTPException(401, "Sign in with MyAnimeList to do that.")
    return s


Session = Annotated[dict, Depends(require_session)]


def require_approved_session(request: Request) -> dict:
    """The session of a signed-in, admin-approved account (malrec.access)."""
    from .access import require_approved
    return require_approved(request).session


def require_signed_in_session(request: Request) -> dict:
    """Any signed-in account, approved or still waiting (preferences only)."""
    from .access import require_signed_in
    return require_signed_in(request).session


ApprovedSession = Annotated[dict, Depends(require_approved_session)]
AnySession = Annotated[dict, Depends(require_signed_in_session)]


def require_csrf(x_requested_with: str | None = Header(default=None)) -> None:
    if x_requested_with != CSRF_HEADER_VALUE:
        raise HTTPException(403, "missing X-Requested-With header")


# -------------------------------------------------------------- sign in --

@router.get("/auth/login", dependencies=[rate_limit("login", 20, 600)])
def login(return_to: str = "/") -> RedirectResponse:
    cfg = settings()
    if not cfg.mal_client_id:
        raise HTTPException(500, "MAL_CLIENT_ID is not configured")
    execute("DELETE FROM oauth_state WHERE created_at < now() - %s::interval",
            (f"{int(STATE_TTL.total_seconds())} seconds",))
    # sessions past their lifetime are otherwise only removed when used again
    execute("DELETE FROM mal_session WHERE created_at < now() - %s::interval",
            (f"{int(cfg.session_days)} days",))
    # token_urlsafe only emits PKCE-unreserved characters; 96 chars sits
    # inside the 43-128 range the spec allows.
    verifier = secrets.token_urlsafe(72)[:96]
    state = secrets.token_urlsafe(24)
    execute("INSERT INTO oauth_state (state, code_verifier, return_to) VALUES (%s,%s,%s)",
            (state, verifier, _safe_return_to(return_to)))
    params = {
        "response_type": "code",
        "client_id": cfg.mal_client_id,
        "state": state,
        "code_challenge": verifier,
        "code_challenge_method": "plain",
    }
    # Optional per MAL's docs when exactly one redirect URL is registered.
    if cfg.mal_redirect_uri:
        params["redirect_uri"] = cfg.mal_redirect_uri
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    # MAL answers a redirect_uri that does not match the app's registration
    # with 401 + "WWW-Authenticate: Basic", which browsers render as a
    # password prompt that can never succeed. Check first, and explain.
    problem = _preflight(url)
    if problem:
        execute("DELETE FROM oauth_state WHERE state=%s", (state,))
        base = cfg.app_base_url.rstrip("/")
        return RedirectResponse(f"{base}/?auth_error={problem}", status_code=302)
    return RedirectResponse(url, status_code=302)


def _preflight(url: str) -> str | None:
    """'redirect_mismatch' if MAL would reject this authorize request, else
    None. Network trouble is not treated as a misconfiguration."""
    try:
        r = httpx.get(url, follow_redirects=False, timeout=10.0,
                      headers={"User-Agent": "malrec/0.1"})
    except httpx.HTTPError:
        return None
    if r.status_code == 401 and "invalid_client" in r.text:
        log.warning("MAL rejected the authorize request: the redirect URL %r is not the one"
                    " registered for this client", settings().mal_redirect_uri or "(omitted)")
        return "redirect_mismatch"
    return None


@router.get("/auth/callback")
def callback(code: str | None = None, state: str | None = None,
             error: str | None = None) -> RedirectResponse:
    cfg = settings()
    base = cfg.app_base_url.rstrip("/")
    if error or not code or not state:
        # MAL's error string is attacker-controllable here: pass on only the
        # codes the app knows how to explain
        known = error if error in ("access_denied",) else "cancelled"
        return RedirectResponse(f"{base}/?auth_error={known}", status_code=302)

    # single use: the row is consumed whether or not the rest succeeds
    st = one(
        "DELETE FROM oauth_state WHERE state=%s AND created_at > now() - %s::interval"
        " RETURNING code_verifier, return_to",
        (state, f"{int(STATE_TTL.total_seconds())} seconds"))
    if not st:
        return RedirectResponse(f"{base}/?auth_error=expired", status_code=302)

    exchange = {"grant_type": "authorization_code", "code": code,
                "code_verifier": st["code_verifier"]}
    # must be sent exactly when it was sent to /authorize
    if cfg.mal_redirect_uri:
        exchange["redirect_uri"] = cfg.mal_redirect_uri
    tok = _token_request(exchange)
    access, refresh = tok["access_token"], tok["refresh_token"]
    try:
        with MalClient(token=access) as mal:
            me = mal.me()
    except MalApiError:
        return RedirectResponse(f"{base}/?auth_error=profile", status_code=302)

    username = me["name"]
    uid = get_or_create_user(username)
    acct = one("UPDATE app_user SET last_login_at = now(),"
               " requested_at = coalesce(requested_at, now()) WHERE id=%s RETURNING status",
               (uid,))
    status = acct["status"] if acct else "pending"
    if status == "pending":
        # The admin named in the server's config approves themselves on first
        # sign-in; otherwise a fresh install has nobody who could approve
        # anyone. Config only, so no request can claim it.
        from .access import is_admin_name
        if is_admin_name(username):
            execute("UPDATE app_user SET status='approved', approved_at=now() WHERE id=%s",
                    (uid,))
            status = "approved"
    if status in ("rejected", "blocked"):
        # no session at all: nothing is fetched or stored for them
        return RedirectResponse(f"{base}/?auth_error=blocked", status_code=302)
    set_picture(uid, me.get("picture"))
    session_token = secrets.token_urlsafe(32)
    execute(
        "INSERT INTO mal_session (token_hash, app_user_id, mal_user_id, access_token,"
        " refresh_token, expires_at) VALUES (%s,%s,%s,%s,%s,%s)",
        (_hash(session_token), uid, me.get("id"), seal(access), seal(refresh),
         _expiry(tok.get("expires_in"))))

    # Build (or refresh) their recommendations using the token, which also
    # reads lists that are private and so could never be fetched anonymously.
    # A pending account gets a session (to see its status) but nothing is
    # fetched for it until the admin approves it.
    has_recs = scalar("SELECT count(*) FROM recommendation WHERE user_id=%s", (uid,))
    job = onboarding.current(username)
    if status == "approved" and not (job and job["state"] == "running"):
        tasks.enqueue("login_sync" if has_recs else "onboard", uid)

    resp = RedirectResponse(f"{base}{st['return_to']}", status_code=302)
    resp.set_cookie(COOKIE, session_token, httponly=True, samesite="lax",
                    secure=cfg.session_cookie_secure,
                    max_age=cfg.session_days * 86400, path="/")
    return resp


@router.get("/auth/config")
def auth_config() -> dict:
    """What the app needs to explain a sign-in problem: the callback URL that
    has to be registered on MyAnimeList. Contains nothing secret."""
    cfg = settings()
    return {"redirect_uri": cfg.mal_redirect_uri or None,
            "register_at": "https://myanimelist.net/apiconfig"}


@router.get("/auth/me")
def me(request: Request) -> dict:
    from .access import viewer
    v = viewer(request)
    if v is None:
        return {"signed_in": False}
    pic = one("SELECT picture_url FROM app_user WHERE id=%s", (v.user_id,))
    return {"signed_in": True, "username": v.username, "mal_user_id": v.session["mal_user_id"],
            "status": v.status, "is_admin": v.is_admin,
            "picture": pic["picture_url"] if pic else None}


@router.post("/auth/logout", dependencies=[Depends(require_csrf)])
def logout(request: Request) -> JSONResponse:
    token = request.cookies.get(COOKIE)
    if token:
        execute("DELETE FROM mal_session WHERE token_hash=%s", (_hash(token),))
    resp = JSONResponse({"signed_in": False})
    resp.delete_cookie(COOKIE, path="/")
    return resp


# ------------------------------------------------------ plan-to-watch queue --

def _mirror(user_id: int, mal_id: int, status: str | None) -> None:
    """Keep the local copy of the list in step with what was written to MAL,
    so the change shows up without waiting for the next full list sync."""
    with conn() as c, c.cursor() as cur:
        if status is None:
            cur.execute("DELETE FROM list_entry WHERE user_id=%s AND mal_id=%s",
                        (user_id, mal_id))
        else:
            cur.execute(
                "INSERT INTO list_entry (user_id, mal_id, status, score, updated_at)"
                " VALUES (%s,%s,%s,0,now())"
                " ON CONFLICT (user_id, mal_id) DO UPDATE SET status=EXCLUDED.status,"
                " updated_at=now()", (user_id, mal_id, status))
        c.commit()


def _log_action(user_id: int, mal_id: int, action: str) -> None:
    execute("INSERT INTO feedback (user_id, mal_id, action) VALUES (%s,%s,%s)",
            (user_id, mal_id, action))


@router.post("/me/queue/{mal_id}", dependencies=[Depends(require_csrf), rate_limit("mal_write", 120, 600)])
def queue(mal_id: int, s: ApprovedSession) -> dict:
    with MalClient(token=s["access_token"]) as mal:
        try:
            current = mal.my_list_status(mal_id)
            if current and current.get("status") != "plan_to_watch":
                # never downgrade something they are watching or have finished
                raise HTTPException(409, {
                    "message": f"Already on your list as {current['status'].replace('_', ' ')}.",
                    "list_status": current["status"]})
            if not current:
                mal.set_list_status(mal_id, "plan_to_watch")
        except MalApiError as e:
            raise _mal_error(e) from e
    _mirror(s["app_user_id"], mal_id, "plan_to_watch")
    _log_action(s["app_user_id"], mal_id, "queued")
    return {"mal_id": mal_id, "list_status": "plan_to_watch"}


@router.delete("/me/queue/{mal_id}", dependencies=[Depends(require_csrf), rate_limit("mal_write", 120, 600)])
def unqueue(mal_id: int, s: ApprovedSession) -> dict:
    with MalClient(token=s["access_token"]) as mal:
        try:
            current = mal.my_list_status(mal_id)
            if current and current.get("status") != "plan_to_watch":
                raise HTTPException(409, {
                    "message": f"On your list as {current['status'].replace('_', ' ')};"
                               " only Plan to Watch entries are removed from here.",
                    "list_status": current["status"]})
            if current:
                mal.delete_list_entry(mal_id)
        except MalApiError as e:
            raise _mal_error(e) from e
    _mirror(s["app_user_id"], mal_id, None)
    _log_action(s["app_user_id"], mal_id, "unqueued")
    return {"mal_id": mal_id, "list_status": None}


# ---------------------------------------------------------------- rating --

class RateIn(BaseModel):
    score: int = Field(ge=0, le=10, description="1-10, or 0 to clear the score")


@router.post("/me/rate/{mal_id}", dependencies=[Depends(require_csrf), rate_limit("mal_write", 120, 600)])
def rate(mal_id: int, body: RateIn, s: ApprovedSession, source: str | None = None) -> dict:
    """Score a title on MyAnimeList, then refresh recommendations.

    An unlisted or planned title is marked completed (with its episode count)
    when scored; anything already watching, on hold or dropped keeps its
    status. The local copy is updated at once, so the title leaves every
    surface on the next read, and the rebuild runs in the background.
    """
    uid = s["app_user_id"]
    with MalClient(token=s["access_token"]) as mal:
        try:
            current = mal.my_list_status(mal_id)
            status = (current or {}).get("status")
            if status is None and body.score == 0:
                raise HTTPException(422, "Not on your list, so there is no score to clear.")
            fields: dict = {"score": body.score}
            if status in (None, "plan_to_watch") and body.score > 0:
                fields["status"] = "completed"
                eps = scalar("SELECT num_episodes FROM anime WHERE mal_id=%s", (mal_id,))
                if eps:
                    fields["num_watched_episodes"] = eps
            mal.update_list_status(mal_id, fields)
        except MalApiError as e:
            raise _mal_error(e) from e
    new_status = fields.get("status", status)
    finished = dt.datetime.now(dt.UTC).date() if fields.get("status") == "completed" else None
    execute(
        "INSERT INTO list_entry (user_id, mal_id, status, score, episodes_watched,"
        " finished_at, updated_at) VALUES (%s,%s,%s,%s,%s,%s,now())"
        " ON CONFLICT (user_id, mal_id) DO UPDATE SET status=EXCLUDED.status,"
        " score=EXCLUDED.score, finished_at=coalesce(EXCLUDED.finished_at,"
        " list_entry.finished_at), episodes_watched=greatest(list_entry.episodes_watched,"
        " EXCLUDED.episodes_watched), updated_at=now()",
        (uid, mal_id, new_status, body.score, fields.get("num_watched_episodes", 0), finished))
    execute("INSERT INTO feedback (user_id, mal_id, action, surface) VALUES (%s,%s,'rated',%s)",
            (uid, mal_id, "quiz" if source == "quiz" else None))
    tasks.enqueue("rebuild", uid)
    return {"mal_id": mal_id, "list_status": new_status, "list_score": body.score,
            "refreshing": True}


# ------------------------------------------------------------ rating round --

@router.get("/me/quiz")
def quiz_next(s: ApprovedSession, n: int = 3) -> dict:
    """The next titles to ask the signed-in user about (see malrec.quiz)."""
    from .model import cached_model
    from .quiz import next_cards
    uid = s["app_user_id"]
    try:
        model, _ = cached_model(uid)
    except ValueError:                   # no scored entries yet
        model = None
    ids = next_cards(uid, model, max(1, min(n, 10)))
    rows = {r["mal_id"]: r for r in query(
        "SELECT mal_id, title, title_en, picture_large, picture_medium, season_year,"
        " media_type, num_episodes, mal_genres FROM anime WHERE mal_id = ANY(%s)", (ids,))}
    rated = scalar("SELECT count(*) FROM feedback WHERE user_id=%s AND action='rated'"
                   " AND surface='quiz'", (uid,)) or 0
    return {"cards": [rows[i] for i in ids if i in rows], "rated_in_round": rated}


@router.post("/me/quiz/{mal_id}/{answer}", dependencies=[Depends(require_csrf), rate_limit("mal_write", 120, 600)])
def quiz_answer(mal_id: int, answer: str, s: ApprovedSession) -> dict:
    """'unseen' or 'skip': remembered so the title is not asked again. A
    rating goes through POST /me/rate/{mal_id}?source=quiz instead."""
    if answer not in ("unseen", "skip"):
        raise HTTPException(422, "answer must be 'unseen' or 'skip'")
    execute("INSERT INTO feedback (user_id, mal_id, action, surface) VALUES (%s,%s,%s,'quiz')",
            (s["app_user_id"], mal_id, f"quiz_{answer}"))
    return {"mal_id": mal_id, "answer": answer}


# ------------------------------------------------------------------ sync --

@router.post("/me/sync", dependencies=[Depends(require_csrf)])
def sync(s: ApprovedSession) -> dict:
    """Re-read the whole list from MyAnimeList, then rebuild. At most once
    per SYNC_COOLDOWN per account; the answer says when the next is allowed."""
    uid = s["app_user_id"]
    row = one("UPDATE app_user SET last_manual_sync_at = now() WHERE id=%s AND"
              " (last_manual_sync_at IS NULL OR last_manual_sync_at < now() - %s::interval)"
              " RETURNING last_manual_sync_at",
              (uid, f"{int(SYNC_COOLDOWN.total_seconds())} seconds"))
    if row is None:
        last = scalar("SELECT last_manual_sync_at FROM app_user WHERE id=%s", (uid,))
        wait = int((last + SYNC_COOLDOWN - dt.datetime.now(dt.UTC)).total_seconds()) + 1
        raise HTTPException(429, {"message": "Synced recently - try again shortly.",
                                  "retry_after": max(wait, 1)},
                            headers={"Retry-After": str(max(wait, 1))})
    tasks.enqueue("sync", uid)
    return {"status": "syncing", "next_allowed_in": int(SYNC_COOLDOWN.total_seconds())}


@router.get("/me/sync")
def sync_status(s: ApprovedSession) -> dict:
    uid = s["app_user_id"]
    last = scalar("SELECT last_manual_sync_at FROM app_user WHERE id=%s", (uid,))
    wait = 0
    if last is not None:
        wait = max(0, int((last + SYNC_COOLDOWN - dt.datetime.now(dt.UTC)).total_seconds()))
    return {"last_sync_at": last, "next_allowed_in": wait, **refresh.state(uid)}


# ----------------------------------------------------------------- prefs --

class PrefsIn(BaseModel):
    lang: str | None = None
    filters: dict | None = None


FILTER_KEYS = {"media_types": list, "exclude_genres": list, "eps_min": int, "eps_max": int,
               "year_min": int, "year_max": int}


def _clean_filters(f: dict) -> dict:
    out: dict = {}
    for k, typ in FILTER_KEYS.items():
        v = f.get(k)
        if v is None:
            continue
        if typ is list:
            if not isinstance(v, list) or len(v) > 40:
                raise HTTPException(422, f"filters.{k} must be a short list")
            out[k] = [str(x)[:40] for x in v]
        else:
            try:
                out[k] = int(v)
            except (TypeError, ValueError) as e:
                raise HTTPException(422, f"filters.{k} must be a number") from e
    return out


@router.get("/me/prefs")
def get_prefs(s: AnySession) -> dict:
    return one("SELECT prefs FROM app_user WHERE id=%s", (s["app_user_id"],))["prefs"] or {}


@router.put("/me/prefs", dependencies=[Depends(require_csrf)])
def put_prefs(body: PrefsIn, s: AnySession) -> dict:
    """Language and filters, kept with the account so they
    follow the user to any browser."""
    patch: dict = {}
    if body.lang is not None:
        if body.lang not in LANGUAGES:
            raise HTTPException(422, f"lang must be one of {LANGUAGES}")
        patch["lang"] = body.lang
    if body.filters is not None:
        patch["filters"] = _clean_filters(body.filters)
    row = one("UPDATE app_user SET prefs = prefs || %s WHERE id=%s RETURNING prefs",
              (Jsonb(patch), s["app_user_id"]))
    return row["prefs"]


def _mal_error(e: MalApiError) -> HTTPException:
    if e.status == 401:
        return HTTPException(401, "Your MyAnimeList session expired. Please sign in again.")
    return HTTPException(502, "MyAnimeList did not accept the change. Try again shortly.")
