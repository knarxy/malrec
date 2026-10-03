"""Watch together: consent, privacy, no username probing."""
from __future__ import annotations

import datetime as dt
import secrets

import pytest
from fastapi.testclient import TestClient

from malrec import auth
from malrec.api import app
from malrec.db import execute, scalar
from malrec.ingest.store import get_or_create_user

CSRF = {"X-Requested-With": "malrec"}
N = {"a": "malrec_w_a", "b": "malrec_w_b", "c": "malrec_w_c", "p": "malrec_w_pend"}


@pytest.fixture()
def people():
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    ids, cl = {}, {}
    for k, name in N.items():
        uid = get_or_create_user(name)
        execute("UPDATE app_user SET status=%s WHERE id=%s",
                ("pending" if k == "p" else "approved", uid))
        tok = secrets.token_urlsafe(16)
        execute("INSERT INTO mal_session (token_hash, app_user_id, mal_user_id, access_token,"
                " refresh_token, expires_at) VALUES (%s,%s,1,'a','r',%s)",
                (auth._hash(tok), uid, dt.datetime.now(dt.UTC) + dt.timedelta(days=1)))
        c = TestClient(app)
        c.cookies.set(auth.COOKIE, tok)
        ids[k], cl[k] = uid, c
    yield ids, cl
    for name in N.values():
        execute("DELETE FROM app_user WHERE mal_username=%s", (name,))


def test_invite_does_not_reveal_accounts(people):
    ids, cl = people
    a = cl["a"]
    for name in ("no_such_user_x", N["p"], N["a"]):        # unknown, pending, self
        assert a.post("/together/invite", headers=CSRF, json={"username": name}).json() \
            == {"status": "sent"}
    assert scalar("SELECT count(*) FROM together_pair WHERE inviter=%s", (ids["a"],)) == 0
    assert a.post("/together/invite", json={"username": N["b"]}).status_code == 403  # CSRF


def test_consent_and_privacy(people):
    _, cl = people
    cl["a"].post("/together/invite", headers=CSRF, json={"username": N["b"]})
    pid = cl["b"].get("/together").json()["incoming"][0]["id"]
    assert cl["a"].post(f"/together/{pid}/accept", headers=CSRF).status_code == 403
    assert cl["a"].get(f"/together/{pid}/list").status_code == 409          # not accepted
    assert cl["c"].get(f"/together/{pid}/list").status_code == 404          # a stranger
    assert cl["b"].post(f"/together/{pid}/accept", headers=CSRF).status_code == 200
    assert cl["a"].get("/together").json()["partners"][0]["other"] == N["b"]
    assert cl["c"].delete(f"/together/{pid}", headers=CSRF).status_code == 404
    assert cl["b"].delete(f"/together/{pid}", headers=CSRF).status_code == 200
    assert cl["a"].get("/together").json()["partners"] == []


def test_pending_accounts_cannot_use_it(people):
    _, cl = people
    assert cl["p"].get("/together").status_code == 403
    assert TestClient(app).get("/together").status_code == 401


def test_invite_link_single_use(people):
    _, cl = people
    assert cl["a"].post("/together/link").status_code == 403                  # CSRF
    url = cl["a"].post("/together/link", headers=CSRF).json()["url"]
    token = url.split("together=")[1]
    assert scalar("SELECT count(*) FROM together_link WHERE token_hash = %s",
                  (token.encode(),)) == 0                                    # only the hash
    assert cl["p"].get(f"/together/link/{token}").status_code == 403         # not approved
    assert cl["b"].get(f"/together/link/{token}").json() == {"inviter": N["a"], "own": False}
    assert cl["a"].post(f"/together/link/{token}/accept", headers=CSRF).status_code == 409
    r = cl["b"].post(f"/together/link/{token}/accept", headers=CSRF)
    assert r.status_code == 200 and r.json()["other"] == N["a"]
    assert cl["a"].get("/together").json()["partners"][0]["other"] == N["b"]
    # used once: the next person gets nothing, and neither does a made-up token
    assert cl["c"].get(f"/together/link/{token}").status_code == 404
    assert cl["c"].post(f"/together/link/{token}/accept", headers=CSRF).status_code == 404
    assert cl["c"].get("/together/link/" + "x" * 32).status_code == 404


def test_invite_link_accepts_pending_pair_and_expires(people):
    ids, cl = people
    cl["a"].post("/together/invite", headers=CSRF, json={"username": N["b"]})
    token = cl["b"].post("/together/link", headers=CSRF).json()["url"].split("together=")[1]
    assert cl["a"].post(f"/together/link/{token}/accept", headers=CSRF).status_code == 200
    assert scalar("SELECT count(*) FROM together_pair WHERE status='accepted' AND "
                  "%s IN (inviter, invitee)", (ids["a"],)) == 1
    token = cl["a"].post("/together/link", headers=CSRF).json()["url"].split("together=")[1]
    execute("UPDATE together_link SET expires_at = now() - interval '1 minute' WHERE inviter=%s",
            (ids["a"],))
    assert cl["c"].post(f"/together/link/{token}/accept", headers=CSRF).status_code == 404
