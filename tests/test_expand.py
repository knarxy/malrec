"""Growing the population sample (fullfetch.expand_discover), with MAL faked."""
from __future__ import annotations

import pytest

from malrec.db import execute, query, scalar
from malrec.ingest import fullfetch

PREFIX = "zz_expand_test_"


@pytest.fixture()
def fake_forum(monkeypatch):
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    calls = []

    def raw_get(mal, path, params, tries=6):
        calls.append((path, dict(params)))
        if path == "/forum/boards":
            return 200, {"categories": [{"boards": [{"id": 1}, {"id": 2}]}]}
        if path == "/forum/topics":
            b, off = params["board_id"], params["offset"]
            if off >= 100:                      # one page of topics per board
                return 200, {"data": []}
            return 200, {"data": [{"id": b * 10 + k,
                                   "created_by": {"name": f"{PREFIX}author{b}{k}"},
                                   "last_post_created_by": {"name": f"{PREFIX}last{b}{k}"}}
                                  for k in range(3)]}
        if path.startswith("/forum/topic/"):
            t = path.rsplit("/", 1)[1]
            return 200, {"data": {"posts": [{"created_by": {"name": f"{PREFIX}reply{t}{k}"}}
                                            for k in range(4)]}}
        return 404, {}
    monkeypatch.setattr(fullfetch, "_raw_get", raw_get)
    monkeypatch.setattr(fullfetch, "_app_user_hashes",
                        lambda: {fullfetch.name_hash(f"{PREFIX}author10")})
    saved = query("SELECT cursor, done FROM fetch_state WHERE stage='cf_expand'")
    execute("DELETE FROM fetch_state WHERE stage='cf_expand'")
    yield calls
    execute("DELETE FROM cf_user WHERE name LIKE %s OR name_hash = %s",
            (PREFIX + "%", fullfetch.name_hash(f"{PREFIX}author10")))
    execute("DELETE FROM fetch_state WHERE stage='cf_expand'")
    if saved:
        from psycopg.types.json import Jsonb
        execute("INSERT INTO fetch_state (stage, cursor, done) VALUES ('cf_expand', %s, %s)",
                (Jsonb(saved[0]["cursor"]), saved[0]["done"]))


def test_discovery_finds_posters_skips_app_users_and_resumes(fake_forum):
    added = fullfetch.expand_discover(None, want=1000)
    rows = query("SELECT name, state, source FROM cf_user WHERE name LIKE %s", (PREFIX + "%",))
    names = {r["name"] for r in rows}
    # 2 boards x 3 topics: author + last poster + 4 repliers each, one an app user
    assert added == 2 * 3 * 6 - 1
    assert f"{PREFIX}reply100" in names and all(r["state"] == "pending" for r in rows)
    assert scalar("SELECT state FROM cf_user WHERE name_hash = %s",
                  (fullfetch.name_hash(f"{PREFIX}author10"),)) == "excluded"
    # every board is exhausted, the cursor says so, and a rerun asks for nothing new
    n = len(fake_forum)
    assert fullfetch.expand_discover(None, want=1000) == 0
    assert len(fake_forum) == n
