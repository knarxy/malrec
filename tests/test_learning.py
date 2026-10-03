"""Which later actions count as outcomes of a recommendation.

Runs the production outcome query on a scratch user. The point under test:
an action taken on MyAnimeList itself (seen by a list sync) counts exactly
like one taken in the app, and titles shown only in the user's own Plan to
Watch never count.
"""
from __future__ import annotations

import pytest

from malrec import learn
from malrec.db import execute, query, scalar
from malrec.ingest.store import get_or_create_user

SCRATCH = "malrec_test_learning"


@pytest.fixture()
def uid():
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    u = get_or_create_user(SCRATCH)
    yield u
    execute("DELETE FROM app_user WHERE mal_username=%s", (SCRATCH,))


def _show(uid, mal_id, surface="discover"):
    execute("INSERT INTO rec_log (user_id, surface, mal_id, rank, predicted, shown_at)"
            " VALUES (%s,%s,%s,1,7.5, now() - interval '2 days')", (uid, surface, mal_id))


def _list(uid, mal_id, status, score=0):
    execute("INSERT INTO list_entry (user_id, mal_id, status, score, updated_at)"
            " VALUES (%s,%s,%s,%s, now())", (uid, mal_id, status, score))


def test_mal_side_actions_are_outcomes(uid):
    # the user's mean is 8 (from two older, unshown ratings)
    execute("INSERT INTO list_entry (user_id, mal_id, status, score, updated_at) VALUES"
            " (%s, 900001, 'completed', 9, now() - interval '1 year'),"
            " (%s, 900002, 'completed', 7, now() - interval '1 year')", (uid, uid))
    cases = {1001: ("plan_to_watch", 0, True, False),   # added to PTW on MAL
             1002: ("watching", 0, True, False),        # started it
             1003: ("completed", 9, True, False),       # rated above mean
             1004: ("completed", 5, False, True),       # rated well below mean
             1005: ("dropped", 0, False, True)}         # dropped it
    for m, (st, sc, _, _) in cases.items():
        _show(uid, m)
        _list(uid, m, st, sc)
    _show(uid, 1006)
    execute("INSERT INTO feedback (user_id, mal_id, action) VALUES (%s, 1006, 'not_interested')",
            (uid,))
    _show(uid, 1007, surface="plan_to_watch")            # their own queue: never an outcome
    _list(uid, 1007, "completed", 10)
    rows = {r["mal_id"]: r for r in query(learn.OUTCOMES_SQL, {"surfaces": list(learn.DISCOVERY)})
            if r["user_id"] == uid}
    for m, (_, _, pos, neg) in cases.items():
        assert (rows[m]["positive"], rows[m]["negative"]) == (pos, neg), m
    assert (rows[1006]["positive"], rows[1006]["negative"]) == (False, True)
    assert 1007 not in rows
