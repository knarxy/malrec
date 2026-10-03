"""Rebuilding a user's recommendations after their list changed.

Triggered by a rating from the app or the "sync with MyAnimeList" button.
Runs in the API process's background tasks. Requests that arrive while a
rebuild is running do not start a second one; they mark it dirty, and it
runs once more when it finishes - so five quick ratings cost two rebuilds,
not five.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

# How long the app waits on a sync before telling the user it gave up. The
# work itself is bounded by the MAL client's per-request timeouts and retries;
# this only bounds what the UI shows as "in progress".
SYNC_TIMEOUT_S = 180

_lock = threading.Lock()
_state: dict[int, dict] = {}


def state(user_id: int) -> dict:
    with _lock:
        st = dict(_state.get(user_id) or {"state": "idle"})
    if st["state"] in ("syncing", "rebuilding") and \
            time.time() - st.get("started", 0) > SYNC_TIMEOUT_S:
        st["state"] = "timeout"
    return {"refresh": st["state"], "refresh_error": st.get("error"),
            "refreshed_at": st.get("finished")}


def _begin(user_id: int, phase: str) -> dict | None:
    with _lock:
        st = _state.setdefault(user_id, {"state": "idle"})
        if st["state"] in ("syncing", "rebuilding") and \
                time.time() - st.get("started", 0) < SYNC_TIMEOUT_S:
            st["dirty"] = True
            return None
        st.update(state=phase, dirty=False, started=time.time(), error=None)
        return st


def _rebuild_loop(user_id: int, st: dict) -> None:
    from .model import invalidate_scorer
    from .surfaces import build_all
    while True:
        with _lock:
            st["state"] = "rebuilding"
            st["dirty"] = False
        invalidate_scorer(user_id)
        build_all(user_id, limit=60)
        with _lock:
            if not st.get("dirty"):
                st.update(state="idle", finished=time.time())
                return


def rebuild(user_id: int) -> None:
    st = _begin(user_id, "rebuilding")
    if st is None:
        return
    try:
        _rebuild_loop(user_id, st)
    except Exception as e:
        log.exception("rebuild for user %s failed", user_id)
        with _lock:
            st.update(state="failed", error=f"{type(e).__name__}: {e}"[:300])


def sync_and_rebuild(username: str, user_id: int, token: str) -> None:
    from .ingest.jobs import sync_user_list
    st = _begin(user_id, "syncing")
    if st is None:
        return
    try:
        sync_user_list(username, enrich=True, token=token)
        _rebuild_loop(user_id, st)
    except Exception as e:
        log.exception("sync for %s failed", username)
        with _lock:
            st.update(state="failed", error=f"{type(e).__name__}: {e}"[:300])


# ------------------------------------------------------------ nightly sync --
#
# Scores, status changes and Plan-to-Watch additions made on MyAnimeList itself
# only reach the model through a list sync. This re-reads every app user's
# list once a night (one or two requests each, plus a detail page for each
# title new to the catalogue) and rebuilds only the users whose list changed.

def _fingerprint(user_id: int) -> str | None:
    from .db import scalar
    return scalar("SELECT md5(string_agg(mal_id || ':' || status || ':' || score, ','"
                  " ORDER BY mal_id)) FROM list_entry WHERE user_id=%s", (user_id,))


def token_for(user_id: int) -> str | None:
    """A live MAL access token for this user, if they have signed in within the
    session lifetime - so private lists sync too. Refreshed when expired."""
    import datetime as dt

    from .auth import _refresh
    from .config import settings
    from .db import one
    row = one("SELECT * FROM mal_session WHERE app_user_id=%s AND created_at > now() - %s::interval"
              " ORDER BY last_used_at DESC NULLS LAST, created_at DESC LIMIT 1",
              (user_id, f"{settings().session_days} days"))
    if row is None:
        return None
    from .tokenbox import open_
    row["access_token"] = open_(row["access_token"])
    row["refresh_token"] = open_(row["refresh_token"])
    if row["expires_at"] < dt.datetime.now(dt.UTC) + dt.timedelta(minutes=2):
        row = _refresh(row)
    return row["access_token"] if row else None


def sync_all_users() -> list[dict]:
    from .db import query
    from .ingest.jobs import sync_user_list
    from .model import invalidate_scorer
    from .surfaces import build_all
    out = []
    for u in query("SELECT id, mal_username FROM app_user ORDER BY id"):
        uid, name = u["id"], u["mal_username"]
        before = _fingerprint(uid)
        try:
            sync_user_list(name, enrich=True, token=token_for(uid))
        except Exception as e:  # noqa: BLE001 - one private/renamed list must not stop the rest
            log.warning("nightly sync of %s failed: %s", name, e)
            out.append({"user": name, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        changed = _fingerprint(uid) != before
        if changed:
            invalidate_scorer(uid)
            try:
                build_all(uid)
            except Exception as e:  # noqa: BLE001 - e.g. an account with no scores yet
                log.warning("rebuild of %s after sync failed: %s", name, e)
                out.append({"user": name, "changed": True,
                            "error": f"{type(e).__name__}: {e}"[:200]})
                continue
        out.append({"user": name, "changed": changed})
    return out
