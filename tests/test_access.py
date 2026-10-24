"""Access control: who can see and do what (malrec.access, malrec.admin).

The app is meant to be reachable from the internet: every rule here is one a
stranger or an unapproved account would otherwise get past.
"""
from __future__ import annotations

import datetime as dt
import secrets
from typing import ClassVar

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
        c.cookies.set(auth.cookie_name(), tok)
        ids[role], clients[role] = uid, c
    yield ids, clients
    for n in NAMES.values():
        execute("DELETE FROM app_user WHERE mal_username=%s", (n,))
    execute("DELETE FROM admin_log WHERE admin=%s", (NAMES["admin"],))


def _callback(client: TestClient | None = None, nonce: str | None = None):
    """MAL's redirect back, for a sign-in this client started: its state in
    the database, the matching nonce in its cookie."""
    client = client or TestClient(app)
    st, own = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
    execute("INSERT INTO oauth_state (state, code_verifier, return_to, browser)"
            " VALUES (%s,'v','/',%s)", (st, auth._hash(own)))
    client.cookies.set(auth.cookie_name(auth.OAUTH_COOKIE), own if nonce is None else nonce)
    return client.get(f"/auth/callback?code=c&state={st}", follow_redirects=False)


class _Me:
    """MalClient's @me as `who` says."""
    who: ClassVar[dict] = {}

    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *e): pass
    def me(self): return dict(self.who)


@pytest.fixture()
def mal_signin(accounts, monkeypatch):
    monkeypatch.setattr(auth, "_token_request",
                        lambda data: {"access_token": "a", "refresh_token": "r", "expires_in": 3600})
    monkeypatch.setattr(auth, "MalClient", _Me)
    monkeypatch.setattr(auth.tasks, "enqueue", lambda kind, uid: None)
    yield _Me.who
    execute("DELETE FROM app_user WHERE mal_username LIKE 'malrec_t_%%'")


def test_a_callback_only_signs_in_the_browser_that_started_it(mal_signin):
    """Login CSRF (audit SEC-3): a state alone, sent to someone else's
    browser, must not sign them into the account it was issued for."""
    mal_signin.update(name=NAMES["new"], id=4242)
    for nonce in ("", "someone-elses"):
        r = _callback(nonce=nonce)
        assert "auth_error=expired" in r.headers["location"]
        assert auth.cookie_name() not in r.cookies
    r = _callback()
    assert r.status_code == 302 and auth.cookie_name() in r.cookies


def test_the_account_follows_the_mal_user_not_the_name(mal_signin):
    """SEC-2: renamed on MAL, the same account; a freed name taken by someone
    else, a new account - and never the admin's."""
    mal_signin.update(name=NAMES["new"], id=4242)
    _callback()
    uid = scalar("SELECT id FROM app_user WHERE mal_user_id = 4242")
    mal_signin.update(name="malrec_t_renamed")
    _callback()
    assert one("SELECT id, mal_username FROM app_user WHERE mal_user_id = 4242") == {
        "id": uid, "mal_username": "malrec_t_renamed"}
    # someone else now holds the old account's current name
    mal_signin.update(id=4343)
    _callback()
    assert one("SELECT mal_username FROM app_user WHERE id = %s", (uid,))["mal_username"] \
        == "malrec_t_renamed#4242"
    assert scalar("SELECT id FROM app_user WHERE mal_user_id = 4343") not in (None, uid)
    # the admin's name in another MAL user's hands: refused
    execute("UPDATE app_user SET mal_user_id = 4444 WHERE mal_username = %s", (NAMES["admin"],))
    mal_signin.update(name=NAMES["admin"], id=4545)
    assert "auth_error=blocked" in _callback().headers["location"]
    assert scalar("SELECT mal_user_id FROM app_user WHERE mal_username = %s",
                  (NAMES["admin"],)) == 4444


def test_a_session_from_before_the_host_prefix_is_moved_over(accounts, monkeypatch):
    """Over https the session cookie is __Host-malrec_session; an older
    session's plain cookie still works once and is replaced, a newer one's is
    refused (only another site of the domain could have set it)."""
    ids, _ = accounts
    monkeypatch.setattr(settings(), "session_cookie_secure", True)
    old, new = secrets.token_urlsafe(16), secrets.token_urlsafe(16)
    for tok, made in ((old, auth.PLAIN_COOKIE_BEFORE - dt.timedelta(days=1)),
                      (new, auth.PLAIN_COOKIE_BEFORE + dt.timedelta(minutes=1))):
        execute("INSERT INTO mal_session (token_hash, app_user_id, mal_user_id, access_token,"
                " refresh_token, expires_at, created_at) VALUES (%s,%s,1,'a','r',%s,%s)",
                (auth._hash(tok), ids["user"], dt.datetime.now(dt.UTC) + dt.timedelta(days=1),
                 made))
    monkeypatch.setattr(auth, "PLAIN_COOKIE_BEFORE", dt.datetime.now(dt.UTC) + dt.timedelta(days=30))
    c = TestClient(app, base_url="https://testserver")
    c.cookies.set(auth.COOKIE, old)
    r = c.get("/auth/me")
    assert r.json()["signed_in"] is True
    assert "__Host-malrec_session=" in r.headers.get("set-cookie", "")
    monkeypatch.setattr(auth, "PLAIN_COOKIE_BEFORE",
                        dt.datetime.now(dt.UTC) - dt.timedelta(days=2))
    c = TestClient(app, base_url="https://testserver")
    c.cookies.set(auth.COOKIE, new)
    assert c.get("/auth/me").json() == {"signed_in": False}


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
    monkeypatch.setattr(auth.tasks, "enqueue", lambda kind, uid: started.append((kind, uid)))

    def callback():
        return _callback()

    r = callback()
    assert r.status_code == 302 and auth.cookie_name() in r.cookies
    assert one("SELECT status FROM app_user WHERE mal_username=%s", (NAMES["new"],))["status"] \
        == "pending"
    assert started == []
    execute("UPDATE app_user SET status='blocked' WHERE mal_username=%s", (NAMES["new"],))
    r = callback()
    assert "auth_error=blocked" in r.headers["location"] and auth.cookie_name() not in r.cookies

    # the configured admin approves themselves on a fresh install, but a
    # blocked account stays blocked even if listed
    execute("UPDATE app_user SET status='pending' WHERE mal_username=%s", (NAMES["new"],))
    monkeypatch.setattr(settings(), "admin_users", f"{NAMES['admin']},{NAMES['new']}")
    callback()
    assert one("SELECT status FROM app_user WHERE mal_username=%s", (NAMES["new"],))["status"] \
        == "approved"
    assert [k for k, _ in started] == ["onboard"], "approved and new: onboarding is queued"
    execute("UPDATE app_user SET status='blocked' WHERE mal_username=%s", (NAMES["new"],))
    assert "auth_error=blocked" in callback().headers["location"]


def test_login_is_rate_limited(accounts, monkeypatch):
    monkeypatch.setattr(auth, "_preflight", lambda url: None)
    monkeypatch.setattr(settings(), "mal_client_id", settings().mal_client_id or "test-client")
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
        c.cookies.set(auth.cookie_name(), secrets.token_urlsafe(16))
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


def _request(client: str, xff: str | None = None):
    from starlette.requests import Request
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request({"type": "http", "headers": headers, "client": (client, 1)})


def test_rate_limit_table_keeps_the_recently_seen_within_its_size(monkeypatch):
    monkeypatch.setattr(access, "_MAX_KEYS", 3)
    access._hits.clear()
    dep = access.rate_limit("b", 10, 60).dependency
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.1", "10.0.0.4"):
        dep(_request(ip))
    # the least recently seen (.2) made room; .1 came back and stays
    assert list(access._hits) == [("b", "10.0.0.3"), ("b", "10.0.0.1"), ("b", "10.0.0.4")]
    assert len(access._hits[("b", "10.0.0.1")]) == 2


def test_client_key_is_the_address_the_app_nginx_trusts():
    # the app's nginx sends exactly one address; the API on its own port none
    assert access.client_key(_request("172.26.0.5", "203.0.113.9")) == "203.0.113.9"
    assert access.client_key(_request("127.0.0.1")) == "127.0.0.1"
    # one IPv6 subscriber is one client, however many addresses they rotate
    assert (access.client_key(_request("::1", "2001:db8:1:2:aaaa::1"))
            == access.client_key(_request("::1", "2001:db8:1:2:bbbb::9"))
            == "2001:db8:1:2::/64")
    assert access.client_key(_request("::1", "::ffff:198.51.100.7")) == "198.51.100.7"
    # whatever is not an address does not make a key of its own
    assert access.client_key(_request("::1", "x" * 8000)) == "?"


def test_only_the_admin_sees_who_is_waiting(accounts):
    _, cl = accounts
    rows = cl["admin"].get("/admin/pending").json()
    assert [r["mal_username"] for r in rows if r["mal_username"].startswith("malrec_t_")] \
        == [NAMES["pending"]]
    for role in ("user", "pending"):
        assert cl[role].get("/admin/pending").status_code == 403
    assert TestClient(app).get("/admin/pending").status_code == 401


def test_an_account_can_delete_itself_and_everything_stored(accounts):
    ids, cl = accounts
    uid, name = ids["pending"], NAMES["pending"]              # waiting accounts may too
    execute("INSERT INTO list_entry (user_id, mal_id, status, score) VALUES (%s, 1, 'completed', 8)",
            (uid,))
    execute("INSERT INTO job (kind, username, state) VALUES ('onboard', %s, 'done')", (name,))
    from malrec.ingest.fullfetch import name_hash
    execute("DELETE FROM cf_user WHERE name_hash=%s", (name_hash(name),))
    execute("INSERT INTO cf_user (name_hash, name, source, state) VALUES (%s, %s, 'test', 'done')",
            (name_hash(name), name))
    execute("INSERT INTO admin_log (admin, action, target) VALUES (%s, 'approve', %s)",
            (NAMES["admin"], name))
    c = cl["pending"]
    assert c.request("DELETE", "/me", json={"confirm": name}).status_code == 403    # no CSRF
    assert c.request("DELETE", "/me", headers=CSRF, json={"confirm": "someone"}).status_code == 422
    r = c.request("DELETE", "/me", headers=CSRF, json={"confirm": name.upper()})
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert scalar("SELECT count(*) FROM app_user WHERE id=%s", (uid,)) == 0
    for sql in ("SELECT count(*) FROM list_entry WHERE user_id=%s",
                "SELECT count(*) FROM mal_session WHERE app_user_id=%s"):
        assert scalar(sql, (uid,)) == 0
    assert scalar("SELECT count(*) FROM job WHERE username=%s", (name,)) == 0
    # the sample keeps only the anonymous hash, excluded from future sampling
    row = one("SELECT name, state FROM cf_user WHERE name_hash=%s", (name_hash(name),))
    assert row == {"name": None, "state": "excluded"}
    execute("DELETE FROM cf_user WHERE name_hash=%s", (name_hash(name),))
    assert scalar("SELECT count(*) FROM admin_log WHERE target=%s", (name,)) == 0
    assert c.get("/auth/me").json() == {"signed_in": False}


def test_nobody_can_delete_someone_elses_account(accounts):
    """DELETE /me acts only on the caller's own session; there is no way to
    name another account, and a signed-out visitor gets nothing."""
    ids, cl = accounts
    other = NAMES["other"]
    # signed out, with or without a made-up cookie
    assert TestClient(app).request("DELETE", "/me", headers=CSRF,
                                   json={"confirm": other}).status_code == 401
    fake = TestClient(app)
    fake.cookies.set(auth.cookie_name(), secrets.token_urlsafe(16))
    assert fake.request("DELETE", "/me", headers=CSRF, json={"confirm": other}).status_code == 401
    # signed in as someone else: naming the other account is refused
    r = cl["user"].request("DELETE", "/me", headers=CSRF, json={"confirm": other})
    assert r.status_code == 422
    # the admin's panel delete is the only way to remove another account
    assert cl["user"].request("DELETE", f"/admin/users/{ids['other']}", headers=CSRF).status_code == 403
    assert scalar("SELECT count(*) FROM app_user WHERE id IN (%s, %s)",
                  (ids["other"], ids["user"])) == 2


def test_auto_approval_fills_the_free_slots_then_stops(accounts, monkeypatch):
    names = ["malrec_t_auto1", "malrec_t_auto2"]
    queued, mails = [], []
    monkeypatch.setattr(auth, "_token_request",
                        lambda data: {"access_token": "a", "refresh_token": "r", "expires_in": 3600})
    who = {"name": names[0], "id": 77}

    class Me:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *e): pass
        def me(self): return {"name": who["name"], "id": who["id"]}
    monkeypatch.setattr(auth, "MalClient", Me)
    monkeypatch.setattr(auth.tasks, "enqueue", lambda kind, uid: queued.append((kind, uid)))
    from malrec import notify
    monkeypatch.setattr(notify, "auto_approved", lambda uid, left: mails.append(("auto", left)))
    monkeypatch.setattr(notify, "pending_signup", lambda uid: mails.append(("pending", uid)))
    approved = scalar("SELECT count(*) FROM app_user WHERE status='approved'")
    monkeypatch.setattr(settings(), "auto_approve_limit", approved + 1)      # one free slot

    def callback():
        return _callback()
    try:
        callback()
        import time
        time.sleep(0.3)                                  # mails go out on a thread
        assert one("SELECT status FROM app_user WHERE mal_username=%s", (names[0],))["status"] \
            == "approved"
        assert [k for k, _ in queued] == ["onboard"] and mails == [("auto", 0)]
        who.update(name=names[1], id=78)                 # the slots are gone now
        callback()
        time.sleep(0.3)
        assert one("SELECT status FROM app_user WHERE mal_username=%s", (names[1],))["status"] \
            == "pending"
        assert len(queued) == 1 and mails[-1][0] == "pending"
    finally:
        for n in names:
            execute("DELETE FROM app_user WHERE mal_username=%s", (n,))


def test_admin_sees_each_accounts_latest_action(monkeypatch):
    """Newest of in-app actions and the manual sync, plus 7-day counts."""
    from malrec import admin
    now = dt.datetime.now(dt.UTC)
    rated = {"user_id": 1, "action": "rated", "at": now - dt.timedelta(hours=2),
             "title": "Paprika", "score": 9}

    def fake_query(sql, params=None):
        if "DISTINCT ON" in sql:
            return [rated]
        return [{"user_id": 1, "action": "rated", "n": 12},
                {"user_id": 1, "action": "not_interested", "n": 3}]
    monkeypatch.setattr(admin, "query", fake_query)
    rows = [{"id": 1, "last_manual_sync_at": now - dt.timedelta(hours=5)},
            {"id": 2, "last_manual_sync_at": now - dt.timedelta(minutes=5)},
            {"id": 3, "last_manual_sync_at": None}]
    admin._activity(rows)
    assert rows[0]["last_action"]["kind"] == "rated" and rows[0]["last_action"]["score"] == 9
    assert rows[0]["week"] == {"rated": 12, "not_interested": 3, "sync": 1}
    assert rows[1]["last_action"]["kind"] == "sync"            # only a sync, recently
    assert rows[2]["last_action"] is None and rows[2]["week"] == {}
    assert all("last_manual_sync_at" not in r for r in rows)
