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

from . import onboarding, refresh
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
           (SELECT count(*) FROM mal_session s WHERE s.app_user_id = u.id) AS sessions
      FROM app_user u LEFT JOIN list_entry le ON le.user_id = u.id
     GROUP BY u.id
     ORDER BY (u.status = 'pending') DESC, u.requested_at DESC NULLS LAST, u.id
"""


@router.get("/users")
def users(v: Admin) -> list[dict]:
    out = query(USERS_SQL)
    for u in out:
        job = onboarding.current(u["mal_username"])
        u["onboarding"] = job["state"] if job else None
        u["is_admin"] = u["mal_username"].lower() in {
            a.strip().lower() for a in settings().admin_users.split(",")}
    return out


def _onboard_after_approval(uid: int, username: str) -> None:
    try:
        onboarding.reset_stuck(username)
        onboarding.run(username, token=refresh.token_for(uid))
    except Exception:
        log.exception("onboarding %s after approval failed", username)


@router.post("/users/{uid}/approve", dependencies=CSRF)
def approve(uid: int, v: Admin, background: BackgroundTasks) -> dict:
    u = _user(uid)
    execute("UPDATE app_user SET status='approved', approved_at=now() WHERE id=%s", (uid,))
    _audit(v, "approve", u["mal_username"])
    if not scalar("SELECT count(*) FROM recommendation WHERE user_id=%s", (uid,)):
        background.add_task(_onboard_after_approval, uid, u["mal_username"])
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
def sync(uid: int, v: Admin, background: BackgroundTasks) -> dict:
    u = _user(uid)
    if u["status"] != "approved":
        raise HTTPException(409, "Only approved accounts are synced.")
    background.add_task(refresh.sync_and_rebuild, u["mal_username"], uid, refresh.token_for(uid))
    _audit(v, "sync", u["mal_username"])
    return {"id": uid, "status": "syncing"}


@router.post("/users/{uid}/rebuild", dependencies=CSRF)
def rebuild(uid: int, v: Admin, background: BackgroundTasks) -> dict:
    u = _user(uid)
    if u["status"] != "approved":
        raise HTTPException(409, "Only approved accounts are rebuilt.")
    background.add_task(refresh.rebuild, uid)
    _audit(v, "rebuild", u["mal_username"])
    return {"id": uid, "status": "rebuilding"}


@router.delete("/users/{uid}", dependencies=CSRF)
def delete_user(uid: int, v: Admin) -> dict:
    """Removes the account and everything stored for it (list copy, models,
    recommendations, sessions, feedback). Nothing on MAL is touched."""
    u = _user(uid)
    if u["mal_username"].lower() == v.username.lower():
        raise HTTPException(409, "You cannot delete your own account.")
    execute("DELETE FROM job WHERE username=%s", (u["mal_username"],))
    execute("DELETE FROM app_user WHERE id=%s", (uid,))
    _audit(v, "delete", u["mal_username"])
    return {"id": uid, "deleted": True}


@router.get("/log")
def audit_log(v: Admin) -> list[dict]:
    return query("SELECT admin, action, target, at FROM admin_log ORDER BY id DESC LIMIT 100")


# ----------------------------------------------------------------- system --

JOBS = {                     # log file -> (label, how often it should run, in hours)
    "users.log": ("Listen-Abgleich (nächtlich)", 26),
    "backup.log": ("Datenbank-Sicherung (nächtlich)", 26),
    "upcoming.log": ("Demnächst-Aktualisierung (wöchentlich)", 24 * 8),
    "refresh.log": ("Monatliche Aktualisierung", 24 * 32),
}
_FAIL = re.compile(r"Traceback|FAILED|failed|Error|skipped", re.IGNORECASE)


def _job_status(path: Path, max_age_h: float) -> dict:
    if not path.exists():
        return {"state": "noch nie gelaufen", "ok": None}
    age_h = (time.time() - path.stat().st_mtime) / 3600
    lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    tail = lines[-40:]
    # the last run only: from the last start marker, if the job writes one
    start = max((i for i, ln in enumerate(tail) if ln.startswith("===") and "done" not in ln),
                default=0)
    last = tail[start:]
    failed = any(_FAIL.search(ln) and "non-fatal" not in ln for ln in last)
    ok = not failed and age_h <= max_age_h
    return {"ok": ok, "state": "fehlgeschlagen" if failed else
            ("überfällig" if age_h > max_age_h else "ok"),
            "last_run": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(),
            "tail": last[-6:]}


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
