"""The admin panel's API. Every endpoint needs the admin's session
(malrec.access.Admin), changes also the CSRF header, and every change is
written to admin_log."""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
import threading
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from . import onboarding, tasks
from .access import Admin
from .auth import require_csrf
from .config import settings
from .db import execute, one, query, scalar

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin")
CSRF = [Depends(require_csrf)]


def _audit(v, action: str, target: str | None = None) -> None:
    execute("INSERT INTO admin_log (admin, action, target) VALUES (%s,%s,%s)",
            (v.username, action, target))


def _user(uid: int) -> dict:
    u = one("SELECT id, mal_username, status FROM app_user WHERE id=%s", (uid,))
    if u is None:
        raise HTTPException(404, "unknown user")
    return u


# ------------------------------------------------------------------ users --

USERS_SQL = """
    SELECT u.id, u.mal_username, u.status, u.requested_at, u.approved_at, u.last_login_at,
           u.last_sync_at, u.created_at,
           count(le.mal_id) AS entries, count(*) FILTER (WHERE le.score > 0) AS scored,
           (SELECT count(*) FROM recommendation r WHERE r.user_id = u.id) AS recs,
           (SELECT count(*) FROM mal_session s WHERE s.app_user_id = u.id) AS sessions,
           -- any request from a signed-in session, even just looking at the lists
           (SELECT max(s.last_used_at) FROM mal_session s WHERE s.app_user_id = u.id)
               AS last_active,
           u.last_manual_sync_at
      FROM app_user u LEFT JOIN list_entry le ON le.user_id = u.id
     GROUP BY u.id
     ORDER BY (u.status = 'pending') DESC, u.requested_at DESC NULLS LAST, u.id
"""


@router.get("/pending")
def pending(v: Admin) -> list[dict]:
    """Accounts waiting for approval, oldest first - for the app's
    notification bell, which polls this, so it stays a single cheap query."""
    return query("SELECT id, mal_username, requested_at FROM app_user WHERE status='pending'"
                 " ORDER BY requested_at NULLS LAST, id")


# the newest in-app action per account, with the title and (for a rating)
# the score it was given
LAST_ACTION_SQL = """
    SELECT DISTINCT ON (f.user_id) f.user_id, f.action, f.created_at AS at,
           coalesce(a.title_en, a.title) AS title,
           CASE WHEN f.action = 'rated' THEN le.score END AS score
      FROM feedback f
      LEFT JOIN anime a ON a.mal_id = f.mal_id
      LEFT JOIN list_entry le ON le.user_id = f.user_id AND le.mal_id = f.mal_id
     ORDER BY f.user_id, f.created_at DESC
"""
WEEK_SQL = """
    SELECT user_id, action, count(*) AS n FROM feedback
     WHERE created_at > now() - interval '7 days' GROUP BY 1, 2
"""


def _activity(rows: list[dict]) -> None:
    """Adds last_action (newest of in-app actions and the manual sync) and
    week (counts per action over 7 days) to each account row."""
    import datetime as _dt
    last = {r["user_id"]: r for r in query(LAST_ACTION_SQL)}
    week: dict[int, dict[str, int]] = {}
    for r in query(WEEK_SQL):
        week.setdefault(r["user_id"], {})[r["action"]] = r["n"]
    since = _dt.datetime.now(_dt.UTC) - _dt.timedelta(days=7)
    for u in rows:
        act = last.get(u["id"])
        action = ({"kind": act["action"], "at": act["at"], "title": act["title"],
                   "score": act["score"]} if act else None)
        synced = u.pop("last_manual_sync_at", None)
        if synced and (action is None or synced > action["at"]):
            action = {"kind": "sync", "at": synced, "title": None, "score": None}
        u["last_action"] = action
        w = dict(week.get(u["id"], {}))
        if synced and synced > since:
            w["sync"] = w.get("sync", 0) + 1          # the last one; earlier ones are not kept
        u["week"] = w


@router.get("/users")
def users(v: Admin) -> list[dict]:
    out = query(USERS_SQL)
    _activity(out)
    for u in out:
        job = onboarding.current(u["mal_username"])
        u["onboarding"] = job["state"] if job else None
        u["is_admin"] = u["mal_username"].lower() in {
            a.strip().lower() for a in settings().admin_users.split(",")}
    return out


@router.post("/users/{uid}/approve", dependencies=CSRF)
def approve(uid: int, v: Admin) -> dict:
    u = _user(uid)
    execute("UPDATE app_user SET status='approved', approved_at=now() WHERE id=%s", (uid,))
    _audit(v, "approve", u["mal_username"])
    if not scalar("SELECT count(*) FROM recommendation WHERE user_id=%s", (uid,)):
        tasks.enqueue("onboard", uid)
    return {"id": uid, "status": "approved"}


def _set_status(uid: int, v, status: str, end_sessions: bool) -> dict:
    u = _user(uid)
    if u["mal_username"].lower() == v.username.lower():
        raise HTTPException(409, "You cannot change your own account's status.")
    execute("UPDATE app_user SET status=%s WHERE id=%s", (status, uid))
    if end_sessions:
        execute("DELETE FROM mal_session WHERE app_user_id=%s", (uid,))
    _audit(v, status, u["mal_username"])
    return {"id": uid, "status": status}


@router.post("/users/{uid}/reject", dependencies=CSRF)
def reject(uid: int, v: Admin) -> dict:
    return _set_status(uid, v, "rejected", True)


@router.post("/users/{uid}/block", dependencies=CSRF)
def block(uid: int, v: Admin) -> dict:
    return _set_status(uid, v, "blocked", True)


@router.post("/users/{uid}/unblock", dependencies=CSRF)
def unblock(uid: int, v: Admin) -> dict:
    return _set_status(uid, v, "approved", False)


@router.post("/users/{uid}/logout", dependencies=CSRF)
def end_sessions(uid: int, v: Admin) -> dict:
    u = _user(uid)
    n = execute("DELETE FROM mal_session WHERE app_user_id=%s", (uid,))
    _audit(v, "end sessions", u["mal_username"])
    return {"id": uid, "ended": n}


@router.post("/users/{uid}/sync", dependencies=CSRF)
def sync(uid: int, v: Admin) -> dict:
    u = _user(uid)
    if u["status"] != "approved":
        raise HTTPException(409, "Only approved accounts are synced.")
    tasks.enqueue("sync", uid)
    _audit(v, "sync", u["mal_username"])
    return {"id": uid, "status": "syncing"}


@router.post("/users/{uid}/rebuild", dependencies=CSRF)
def rebuild(uid: int, v: Admin) -> dict:
    u = _user(uid)
    if u["status"] != "approved":
        raise HTTPException(409, "Only approved accounts are rebuilt.")
    tasks.enqueue("rebuild", uid)
    _audit(v, "rebuild", u["mal_username"])
    return {"id": uid, "status": "rebuilding"}


@router.delete("/users/{uid}", dependencies=CSRF)
def delete_user(uid: int, v: Admin) -> dict:
    """Removes the account and everything stored for it (list copy, models,
    recommendations, sessions, feedback). Nothing on MAL is touched."""
    u = _user(uid)
    if u["mal_username"].lower() == v.username.lower():
        raise HTTPException(409, "You cannot delete your own account.")
    from .account import erase
    erase(uid, u["mal_username"], scrub_audit=False)    # the admin keeps the record
    _audit(v, "delete", u["mal_username"])
    return {"id": uid, "deleted": True}


@router.get("/log")
def audit_log(v: Admin) -> list[dict]:
    return query("SELECT admin, action, target, at FROM admin_log ORDER BY id DESC LIMIT 100")


# ----------------------------------------------------------------- system --

# log file -> (job code, how often it should run, in hours). Codes and states
# are translated by the app (app/src/i18n.tsx, "adm.job.*", "adm.jobstate.*").
JOBS = {
    "users.log": ("list_sync", 26),
    "backup.log": ("backup", 26),
    "upcoming.log": ("upcoming", 24 * 8),
    "refresh.log": ("monthly", 24 * 32),
}
_FAIL = re.compile(r"Traceback|FAILED|failed|Error|skipped", re.IGNORECASE)


def _job_status(path: Path, max_age_h: float) -> dict:
    if not path.exists():
        return {"state": "never", "ok": None}
    age_h = (time.time() - path.stat().st_mtime) / 3600
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    tail = lines[-40:]
    # the last run only: from the last start marker, if the job writes one
    start = max((i for i, ln in enumerate(tail) if ln.startswith("===") and "done" not in ln),
                default=0)
    last = tail[start:]
    failed = any(_FAIL.search(ln) and "non-fatal" not in ln for ln in last)
    ok = not failed and age_h <= max_age_h
    return {"ok": ok, "state": "failed" if failed else
            ("overdue" if age_h > max_age_h else "ok"),
            "last_run": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(),
            "tail": last[-6:]}


def _auto_approve_status() -> dict:
    limit = settings().auto_approve_limit
    approved = scalar("SELECT count(*) FROM app_user WHERE status='approved'") or 0
    return {"limit": limit, "approved": approved,
            "left": max(limit - approved, 0) if limit > 0 else None}


@router.get("/system")
def system(v: Admin) -> dict:
    logs = Path(settings().logs_dir)
    gm = one("SELECT id, created_at, meta->>'users' AS users FROM global_model WHERE active")
    backup_line = ""
    if (logs / "backup.log").exists():
        bl = [ln for ln in (logs / "backup.log").read_text().splitlines() if ln.strip()]
        backup_line = bl[-1] if bl else ""
    return {
        "population_model": gm,
        "counts": {
            "anime": scalar("SELECT count(*) FROM anime"),
            "cf_lists": scalar("SELECT count(*) FROM cf_user WHERE state='done'"),
            "cf_ratings": scalar("SELECT count(*) FROM cf_rating"),
            "users": scalar("SELECT count(*) FROM app_user"),
            "pending": scalar("SELECT count(*) FROM app_user WHERE status='pending'"),
            "sessions": scalar("SELECT count(*) FROM mal_session"),
        },
        "last_backup": backup_line,
        "jobs": {name: {"label": label, **_job_status(logs / name, h)}
                 for name, (label, h) in JOBS.items()},
        "version": os.environ.get("MALREC_VERSION", ""),
        "worker": tasks.status(),
        "auto_approve": _auto_approve_status(),
    }


# ----------------------------------------------------------------- report --

_report: dict = {"at": None, "value": None, "running": False}
_rlock = threading.Lock()


def _compute_report() -> None:
    from .report import report
    try:
        value = report()
        with _rlock:
            _report.update(at=dt.datetime.now(dt.UTC).isoformat(), value=value)
    finally:
        with _rlock:
            _report["running"] = False


@router.get("/report")
def get_report(v: Admin) -> dict:
    with _rlock:
        return dict(_report)


@router.post("/report", dependencies=CSRF)
def run_report(v: Admin, background: BackgroundTasks) -> dict:
    with _rlock:
        if _report["running"]:
            return {"running": True}
        _report["running"] = True
    background.add_task(_compute_report)
    _audit(v, "report")
    return {"running": True}
