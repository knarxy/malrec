"""Removing an account and everything stored for it - from the account menu
(DELETE /me) or the admin panel. Nothing on MyAnimeList is touched."""
from __future__ import annotations

from .db import conn


def erase(user_id: int, username: str, scrub_audit: bool = True) -> None:
    """Everything keyed to the account goes with its app_user row (list copy,
    models, recommendations, candidates, feedback, impressions, sessions and
    MAL tokens, pairings, queued tasks - all ON DELETE CASCADE). The rest is
    keyed by name: onboarding jobs, the admin log (the name is replaced), and
    the public-list sample the population model learns from. App users are
    kept out of that sample by the hash of their name; after deletion only
    that hash stays, marked excluded, so the list is not sampled later
    either. One transaction, so a failure leaves the account whole."""
    from .ingest.fullfetch import name_hash
    h = name_hash(username)
    with conn() as c:
        c.execute("DELETE FROM job WHERE lower(username) = lower(%s)", (username,))
        c.execute("DELETE FROM cf_rating WHERE user_id IN"
                  " (SELECT id FROM cf_user WHERE name_hash = %s)", (h,))
        c.execute("INSERT INTO cf_user (name_hash, name, source, state)"
                  " VALUES (%s, NULL, 'deleted_account', 'excluded')"
                  " ON CONFLICT (name_hash) DO UPDATE SET name = NULL, state = 'excluded',"
                  " n_entries = NULL, n_scored = NULL", (h,))
        if scrub_audit:
            c.execute("UPDATE admin_log SET target = '(deleted account)'"
                      " WHERE lower(target) = lower(%s)", (username,))
        c.execute("DELETE FROM app_user WHERE id = %s", (user_id,))
    from .model import invalidate_scorer
    invalidate_scorer(user_id)
