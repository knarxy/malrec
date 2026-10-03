"""Onboarding a user, with progress the web app can display.

Setting up a new profile is not instant - reading the list is one request, but
building the recommendation graph costs one MAL call per listed entry at 1.5
requests a second. That is a few minutes for a typical list, so each step is
recorded as it starts and the UI polls for it.
"""
from __future__ import annotations

import logging
import re

from psycopg.types.json import Jsonb

from .db import execute, one, scalar
from .ingest import jobs as ingest
from .ingest.ondemand import ensure_for_user
from .ingest.store import get_or_create_user

log = logging.getLogger(__name__)

KIND = "onboard"

# MAL usernames: letters, digits, dash, underscore. Validated before it ever
# reaches a URL path so a crafted name cannot reshape the request.
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{2,24}$")

STEPS = [
    "Reading your MyAnimeList profile",
    "Finding your first recommendations",
    "Catching up on the catalogue",
    "Learning what you have watched",
    "Working out your taste",
    "Building your recommendations",
]


class OnboardError(Exception):
    pass


def valid_username(name: str) -> bool:
    return bool(USERNAME_RE.match(name or ""))


def current(username: str) -> dict | None:
    return one(
        "SELECT id, state, step, step_index, step_total, detail, error,"
        "       started_at, finished_at"
        "  FROM job WHERE username=%s AND kind=%s"
        " ORDER BY started_at DESC LIMIT 1",
        (username, KIND),
    )


def _start(username: str) -> int:
    row = one(
        "INSERT INTO job (kind, username, step, step_total) VALUES (%s,%s,%s,%s)"
        " ON CONFLICT DO NOTHING RETURNING id",
        (KIND, username, STEPS[0], len(STEPS)),
    )
    if not row:
        raise OnboardError("an onboarding run is already in progress for this user")
    return row["id"]


def _step(job_id: int, index: int) -> None:
    execute("UPDATE job SET step=%s, step_index=%s, updated_at=now() WHERE id=%s",
            (STEPS[index], index, job_id))


def _finish(job_id: int, detail: dict) -> None:
    execute(
        "UPDATE job SET state='done', step='Ready', step_index=step_total,"
        " detail=%s, updated_at=now(), finished_at=now() WHERE id=%s",
        (Jsonb(detail), job_id),
    )


def _fail(job_id: int, err: str) -> None:
    execute(
        "UPDATE job SET state='failed', error=%s, updated_at=now(), finished_at=now()"
        " WHERE id=%s",
        (err[:1000], job_id),
    )


def run(username: str, token: str | None = None) -> dict:
    """Full cold start for one user. Safe to re-run: every step is incremental,
    so a second run only picks up what changed."""
    from .model import train
    from .surfaces import build_all

    if not valid_username(username):
        raise OnboardError(f"{username!r} is not a valid MyAnimeList username")
    job_id = _start(username)
    detail: dict = {}
    try:
        _step(job_id, 0)
        listed = ingest.sync_user_list(username, enrich=False, token=token)
        detail["list"] = listed
        if listed["entries"] == 0:
            raise OnboardError(
                f"{username} has no anime on their list, or the list is private")

        uid = get_or_create_user(username)
        # With the population model in place a first Safe Bets list needs
        # nothing but the list itself, so it is shown within seconds while
        # the rest of the setup runs; the final build replaces it.
        _step(job_id, 1)
        detail["first"] = _first_recommendations(uid)

        # Only worth doing once; afterwards the catalogue is already broad.
        _step(job_id, 2)
        if (scalar("SELECT count(*) FROM anime") or 0) < 3000:
            detail["catalog"] = ingest.sync_catalog()

        _step(job_id, 3)
        detail["fetched"] = ensure_for_user(uid)

        _step(job_id, 4)
        model, run_id = train(uid)
        detail["model"] = {"run_id": run_id, "algo": model.algo, "metrics": model.metrics}

        _step(job_id, 5)
        detail["surfaces"] = build_all(uid, limit=60)

        _finish(job_id, detail)
        log.info("onboarded %s", username)
        return detail
    except Exception as e:
        log.exception("onboarding %s failed", username)
        _fail(job_id, f"{type(e).__name__}: {e}")
        raise


def _first_recommendations(user_id: int) -> dict:
    from .model import _hybrid_available, train
    from .surfaces import build_surface
    if not _hybrid_available():
        return {"skipped": "no population model"}
    try:
        model, run_id = train(user_id)
        n = len(build_surface(user_id, "safe_bets", limit=60, model=model, run_id=run_id))
        return {"safe_bets": n}
    except Exception as e:  # noqa: BLE001 - the full build below still runs
        log.warning("first recommendations for user %s failed: %s", user_id, e)
        return {"error": str(e)[:200]}


def reset_stuck(username: str) -> int:
    """Clear a run that died with the process still marked running."""
    return execute(
        "UPDATE job SET state='failed', error='interrupted', finished_at=now()"
        " WHERE username=%s AND kind=%s AND state='running'",
        (username, KIND),
    )
