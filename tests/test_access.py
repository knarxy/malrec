"""Access control: who can see and do what (malrec.access, malrec.admin).

The app is meant to be reachable from the internet: every rule here is one a
stranger or an unapproved account would otherwise get past.
"""
from __future__ import annotations

import datetime as dt
import secrets

import pytest
from fastapi.testclient import TestClient

from malrec import access, auth
from malrec.api import app
from malrec.config import settings
from malrec.db import execute, one, scalar
from malrec.ingest.store import get_or_create_user

CSRF = {"X-Requested-With": "malrec"}
NAMES = {"admin": "malrec_t_admin", "user": "malrec_t_user", "pending": "malrec_t_pend",
         "other": "malrec_t_other", "new": "malrec_t_new"}


@pytest.fixture()
def accounts(monkeypatch):
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    monkeypatch.setattr(settings(), "admin_users", NAMES["admin"])
    access._hits.clear()
    ids, clients = {}, {}
    for role in ("admin", "user", "pending", "other"):
        uid = get_or_create_user(NAMES[role])
        execute("UPDATE app_user SET status=%s WHERE id=%s",
                ("pending" if role == "pending" else "approved", uid))
        tok = secrets.token_urlsafe(16)
        execute("INSERT INTO mal_session (token_hash, app_user_id, mal_user_id, access_token,"
                " refresh_token, expires_at) VALUES (%s,%s,1,'a','r',%s)",
                (auth._hash(tok), uid, dt.datetime.now(dt.UTC) + dt.timedelta(days=1)))
        c = TestClient(app)
        c.cookies.set(auth.COOKIE, tok)
        ids[role], clients[role] = uid, c
    yield ids, clients
    for n in NAMES.values():
        execute("DELETE FROM app_user WHERE mal_username=%s", (n,))
    execute("DELETE FROM admin_log WHERE admin=%s", (NAMES["admin"],))


def test_signed_out_sees_nothing(accounts):
    c = TestClient(app)
    assert c.get("/health").json() == {"status": "ok"}
    for path in ("/recommendations/safe_bets", "/profile", "/search?q=naruto", "/anime/5114",
                 "/surfaces", "/model", "/admin/users", "/me/quiz"):
        assert c.get(path).status_code == 401, path
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404
    assert c.get("/users").status_code == 404          # removed
    assert c.post("/onboard", json={"username": "x"}).status_code in (404, 405)


def test_pending_account_only_sees_its_status(accounts):
    _, cl = accounts
    c = cl["pending"]
    me = c.get("/auth/me").json()
    assert me["status"] == "pending" and me["is_admin"] is False
    assert c.get("/recommendations/safe_bets").status_code == 403
    assert c.post("/me/rate/5114", headers=CSRF, json={"score": 8}).status_code == 403
    assert c.get("/me/quiz").status_code == 403
    assert c.put("/me/prefs", headers=CSRF, json={"lang": "de"}).status_code == 200


def test_users_only_see_their_own_data(accounts):
    _, cl = accounts
    c = cl["user"]
    assert c.get(f"/profile?user={NAMES['other']}").status_code == 403
    assert c.get(f"/recommendations/safe_bets?user={NAMES['other']}").status_code == 403
    assert c.get(f"/onboard/{NAMES['other']}").status_code == 403
    assert c.get("/profile").json()["user"] == NAMES["user"]
    assert c.get("/admin/users").status_code == 403
    assert c.post("/feedback", json={"mal_id": 1, "action": "not_interested"}).status_code == 403


def test_admin_can_see_anyone_and_needs_csrf(accounts):
    ids, cl = accounts
    c = cl["admin"]
    assert c.get(f"/profile?user={NAMES['other']}").json()["user"] == NAMES["other"]
    names = [u["mal_username"] for u in c.get("/admin/users").json()]
    assert NAMES["pending"] in names
    pid = ids["pending"]
    assert c.post(f"/admin/users/{pid}/reject").status_code == 403        # no CSRF header
    assert c.post(f"/admin/users/{ids['admin']}/block", headers=CSRF).status_code == 409
    assert c.post(f"/admin/users/{pid}/reject", headers=CSRF).status_code == 200
    assert one("SELECT status FROM app_user WHERE id=%s", (pid,))["status"] == "rejected"
    assert scalar("SELECT count(*) FROM mal_session WHERE app_user_id=%s", (pid,)) == 0
    assert c.get("/admin/log").json()[0]["action"] == "rejected"


def test_blocked_account_is_signed_out_at_once(accounts):
    ids, cl = accounts
    execute("UPDATE app_user SET status='blocked' WHERE id=%s", (ids["other"],))
    assert cl["other"].get("/profile").status_code == 401
    assert cl["other"].get("/auth/me").json() == {"signed_in": False}


def test_sign_in_creates_a_pending_account_and_fetches_nothing(accounts, monkeypatch):
    started = []
    monkeypatch.setattr(auth, "_token_request",
                        lambda data: {"access_token": "a", "refresh_token": "r", "expires_in": 3600})

    class Me:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *e): pass
        def me(self): return {"name": NAMES["new"], "id": 99}
    monkeypatch.setattr(auth, "MalClient", Me)
    monkeypatch.setattr(auth, "_onboard_quietly", lambda *a: started.append(a))

    def callback():
        st = secrets.token_urlsafe(16)
        execute("INSERT INTO oauth_state (state, code_verifier, return_to) VALUES (%s,'v','/')",
                (st,))
        return TestClient(app).get(f"/auth/callback?code=c&state={st}", follow_redirects=False)

    r = callback()
    assert r.status_code == 302 and auth.COOKIE in r.cookies
    assert one("SELECT status FROM app_user WHERE mal_username=%s", (NAMES["new"],))["status"] \
        == "pending"
    assert started == []
    execute("UPDATE app_user SET status='blocked' WHERE mal_username=%s", (NAMES["new"],))
    r = callback()
    assert "auth_error=blocked" in r.headers["location"] and auth.COOKIE not in r.cookies

    # the configured admin approves themselves on a fresh install, but a
    # blocked account stays blocked even if listed
    execute("UPDATE app_user SET status='pending' WHERE mal_username=%s", (NAMES["new"],))
    monkeypatch.setattr(settings(), "admin_users", f"{NAMES['admin']},{NAMES['new']}")
    callback()
    assert one("SELECT status FROM app_user WHERE mal_username=%s", (NAMES["new"],))["status"] \
        == "approved"
    execute("UPDATE app_user SET status='blocked' WHERE mal_username=%s", (NAMES["new"],))
    assert "auth_error=blocked" in callback().headers["location"]


def test_login_is_rate_limited(accounts, monkeypatch):
    monkeypatch.setattr(auth, "_preflight", lambda url: None)
    c = TestClient(app)
    codes = [c.get("/auth/login", follow_redirects=False).status_code for _ in range(21)]
    assert codes[:20] == [302] * 20 and codes[20] == 429


def test_profile_picture_only_from_mal_cdn():
    from malrec.db import one
    from malrec.ingest.store import get_or_create_user, set_picture
    try:
        uid = get_or_create_user("malrec_pic_test")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    try:
        ok = "https://cdn.myanimelist.net/s/common/userimages/abc.webp"
        for url, stored in ((ok, ok), ("https://evil.example/x.png", None),
                            ("http://cdn.myanimelist.net/x.png", None), (None, None)):
            set_picture(uid, url)
            assert one("SELECT picture_url FROM app_user WHERE id=%s", (uid,))["picture_url"] == stored
    finally:
        from malrec.db import execute
        execute("DELETE FROM app_user WHERE id=%s", (uid,))


def test_made_up_cookies_do_not_reset_the_rate_limit(accounts, monkeypatch):
    """The limiter keys on the client, not on a cookie the client chooses."""
    monkeypatch.setattr(auth, "_preflight", lambda url: None)
    codes = []
    for _ in range(21):
        c = TestClient(app)
        c.cookies.set(auth.COOKIE, secrets.token_urlsafe(16))
        codes.append(c.get("/auth/login", follow_redirects=False).status_code)
    assert codes[20] == 429


def test_callback_passes_on_only_known_error_codes():
    c = TestClient(app)
    for error, shown in (("access_denied", "access_denied"),
                         ("x&return_to=//evil.example", "cancelled"), (None, "cancelled")):
        params = {"error": error} if error else {}
        r = c.get("/auth/callback", params=params, follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].endswith(f"/?auth_error={shown}")


def test_rate_limit_table_is_swept(monkeypatch):
    monkeypatch.setattr(access, "_SWEEP_AT", 5)
    access._hits.clear()
    for i in range(8):
        access._hits[("b", f"10.0.0.{i}")] = access.deque()       # quiet clients
    from starlette.requests import Request
    dep = access.rate_limit("b", 10, 60).dependency
    dep(Request({"type": "http", "headers": [], "client": ("10.9.9.9", 1)}))
    assert set(access._hits) == {("b", "10.9.9.9")}
