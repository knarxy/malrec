"""Rebuilding a user's recommendations after their list changed.

Rebuilds and list syncs run in the worker (malrec.tasks); this module keeps
what both the worker and the nightly job need: a live MAL token for a user,
the fingerprint that tells whether a list changed, and the nightly sync.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def state(user_id: int) -> dict:
    """The app's view of the user's latest sync or rebuild (malrec.tasks)."""
    from .tasks import state as task_state
    return task_state(user_id)


# ------------------------------------------------------------ nightly sync --
#
# Scores, status changes and Plan-to-Watch additions made on MyAnimeList itself
# only reach the model through a list sync. This re-reads every app user's
# list once a night (one or two requests each, plus a detail page for each
# title new to the catalogue) and rebuilds only the users whose list changed.

def fingerprint(user_id: int) -> str | None:
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
        before = fingerprint(uid)
        try:
            sync_user_list(name, enrich=True, token=token_for(uid))
        except Exception as e:  # noqa: BLE001 - one private/renamed list must not stop the rest
            log.warning("nightly sync of %s failed: %s", name, e)
            out.append({"user": name, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        changed = fingerprint(uid) != before
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
