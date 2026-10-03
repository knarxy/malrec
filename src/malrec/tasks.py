"""Background work, run by a separate worker process (`malrec worker`).

The API only enqueues: rebuilding a user's lists costs seconds of CPU, and
when it ran in the API's own threads two or three at once made the app slow
for everyone. The queue is a Postgres table (db/migrations/020_task_queue.sql),
claimed with FOR UPDATE SKIP LOCKED, so it needs nothing beyond the database.

  rebuild     rebuild every list (after a rating, or on request)
  sync        re-read the MAL list, then rebuild
  login_sync  re-read the MAL list on sign-in; rebuild only if it changed
  onboard     the first-run setup of a newly approved account

A worker runs one task per user at a time; a request that arrives while one
is waiting is absorbed by it. Progress for the app comes from the same table
(state()). With TASKS_INLINE=true (local development without a worker) a task
runs in a thread of the calling process instead.
"""
from __future__ import annotations

import datetime as dt
import logging
import signal
import threading
import time

from .config import settings
from .db import execute, one, scalar

log = logging.getLogger(__name__)

PRIORITY = {"rebuild": 0, "sync": 0, "login_sync": 1, "onboard": 1}
# How long the app waits on a sync or rebuild before telling the user it gave
# up; the task itself is bounded by the MAL client's timeouts and retries.
UI_TIMEOUT_S = 180
MAX_ATTEMPTS = 3
KEEP_DAYS = 14
HEARTBEAT = "/tmp/malrec-worker-alive"


def enqueue(kind: str, user_id: int) -> None:
    if kind not in PRIORITY:
        raise ValueError(f"unknown task kind {kind!r}")
    if settings().tasks_inline:
        threading.Thread(target=_run_inline, args=(kind, user_id), daemon=True).start()
        return
    execute("INSERT INTO task (kind, user_id, priority) VALUES (%s, %s, %s)"
            " ON CONFLICT (user_id, kind) WHERE state = 'queued' DO NOTHING",
            (kind, user_id, PRIORITY[kind]))


def pending(user_id: int, kind: str) -> bool:
    return bool(scalar("SELECT count(*) FROM task WHERE user_id=%s AND kind=%s"
                       " AND state IN ('queued', 'running')", (user_id, kind)))


def state(user_id: int) -> dict:
    """The app's view of the user's latest sync or rebuild: syncing /
    rebuilding while queued or running, then idle, failed or timeout."""
    row = one("SELECT kind, state, error, created_at, finished_at FROM task"
              " WHERE user_id=%s AND kind IN ('rebuild', 'sync')"
              " ORDER BY (state IN ('queued', 'running')) DESC, id DESC LIMIT 1", (user_id,))
    if row is None:
        return {"refresh": "idle", "refresh_error": None, "refreshed_at": None}
    st = row["state"]
    if st in ("queued", "running"):
        age = (dt.datetime.now(dt.UTC) - row["created_at"]).total_seconds()
        st = "timeout" if age > UI_TIMEOUT_S else (
            "syncing" if row["kind"] == "sync" else "rebuilding")
    elif st == "done":
        st = "idle"
    done = row["finished_at"].timestamp() if row["finished_at"] else None
    return {"refresh": st, "refresh_error": row["error"] if st == "failed" else None,
            "refreshed_at": done}


# ------------------------------------------------------------------ worker --

def run_task(kind: str, user_id: int) -> None:
    from . import onboarding, refresh
    from .ingest.jobs import sync_user_list
    from .model import invalidate_scorer
    from .surfaces import build_all
    name = scalar("SELECT mal_username FROM app_user WHERE id=%s", (user_id,))
    if name is None:
        return                                  # account deleted meanwhile
    if kind == "onboard":
        onboarding.reset_stuck(name)
        onboarding.run(name, token=refresh.token_for(user_id))
        return
    if kind in ("sync", "login_sync"):
        before = refresh.fingerprint(user_id)
        sync_user_list(name, enrich=True, token=refresh.token_for(user_id))
        if kind == "login_sync" and refresh.fingerprint(user_id) == before:
            return
    invalidate_scorer(user_id)
    build_all(user_id, limit=60)


def _run_inline(kind: str, user_id: int) -> None:
    try:
        run_task(kind, user_id)
    except Exception:
        log.exception("inline %s for user %s failed", kind, user_id)


CLAIM_SQL = """
    UPDATE task SET state = 'running', started_at = now(), attempts = attempts + 1
     WHERE id = (SELECT t.id FROM task t
                  WHERE t.state = 'queued'
                    AND NOT EXISTS (SELECT 1 FROM task r
                                     WHERE r.user_id = t.user_id AND r.state = 'running')
                  ORDER BY t.priority, t.id
                  LIMIT 1 FOR UPDATE SKIP LOCKED)
    RETURNING id, kind, user_id
"""


def claim() -> dict | None:
    return one(CLAIM_SQL)


def finish(task_id: int, error: str | None = None) -> None:
    execute("UPDATE task SET state=%s, error=%s, finished_at=now() WHERE id=%s",
            ("failed" if error else "done", error, task_id))


def recover() -> int:
    """At start: tasks a dead worker left running go back in the queue, or
    fail after MAX_ATTEMPTS. Assumes one worker process."""
    # one that has a queued twin, or has failed too often, just fails: the
    # twin will do the same work
    execute("UPDATE task t SET state='failed', error='interrupted', finished_at=now()"
            " WHERE state='running' AND (attempts >= %s OR EXISTS ("
            "   SELECT 1 FROM task q WHERE q.user_id=t.user_id AND q.kind=t.kind"
            "      AND q.state='queued'))", (MAX_ATTEMPTS,))
    return execute("UPDATE task SET state='queued' WHERE state='running'")


def housekeeping() -> None:
    execute("DELETE FROM task WHERE state IN ('done', 'failed')"
            " AND finished_at < now() - %s::interval", (f"{KEEP_DAYS} days",))


def work_once() -> bool:
    """Claim and run one task. False when the queue was empty."""
    t = claim()
    if t is None:
        return False
    started = time.time()
    try:
        run_task(t["kind"], t["user_id"])
        finish(t["id"])
        log.info("task %s %s for user %s done in %.1fs", t["id"], t["kind"], t["user_id"],
                 time.time() - started)
    except Exception as e:
        log.exception("task %s %s for user %s failed", t["id"], t["kind"], t["user_id"])
        finish(t["id"], f"{type(e).__name__}: {e}"[:300])
    return True


def worker(threads: int = 1, idle_sleep: float = 1.0) -> None:
    """Run until SIGTERM/SIGINT; a task in progress is finished first."""
    from pathlib import Path

    from .recsys.service import active_global, item_store
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    log.info("worker: %d requeued after a restart", recover())
    active_global()
    item_store()                       # warm the shared caches before the first task
    beat = Path(HEARTBEAT)

    def loop() -> None:
        last_house = 0.0
        while not stop.is_set():
            beat.touch()
            if time.time() - last_house > 3600:
                housekeeping()
                last_house = time.time()
            try:
                busy = work_once()
            except Exception:          # database briefly unreachable, etc.
                log.exception("worker loop")
                busy = False
            if not busy:
                stop.wait(idle_sleep)

    pool = [threading.Thread(target=loop, name=f"worker-{i}") for i in range(max(1, threads))]
    for th in pool:
        th.start()
    for th in pool:
        th.join()
    log.info("worker stopped")
