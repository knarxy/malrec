"""OAuth session handling and the plan-to-watch queue.

MAL itself is replaced by a stub so these run offline; what is under test is
our side of the contract - above all that un-queueing can never delete an
entry the user has started or finished.
"""
from __future__ import annotations

import datetime as dt
import secrets
from typing import ClassVar

import pytest
from fastapi.testclient import TestClient

from malrec import auth
from malrec.api import app
from malrec.db import execute, one, scalar
from malrec.ingest.store import get_or_create_user

SCRATCH = "malrec_test_user"
CSRF = {"X-Requested-With": "malrec"}


class FakeMal:
    """Stands in for MalClient; `listed` is the user's MAL list."""
    listed: ClassVar[dict[int, str]] = {}
    calls: ClassVar[list[tuple]] = []

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def my_list_status(self, mal_id):
        st = self.listed.get(mal_id)
        return {"status": st} if st else None

    def set_list_status(self, mal_id, status):
        FakeMal.calls.append(("set", mal_id, status))
        self.listed[mal_id] = status

    def delete_list_entry(self, mal_id):
        FakeMal.calls.append(("delete", mal_id))
        self.listed.pop(mal_id, None)

    def update_list_status(self, mal_id, fields):
        FakeMal.calls.append(("update", mal_id, dict(fields)))
        if "status" in fields:
            self.listed[mal_id] = fields["status"]


@pytest.fixture()
def signed_in(monkeypatch):
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    monkeypatch.setattr(auth, "MalClient", FakeMal)
    # the rebuild after a rating/sync is not under test here
    monkeypatch.setattr(auth.tasks, "enqueue", lambda kind, uid: None)
    FakeMal.listed, FakeMal.calls = {}, []
    uid = get_or_create_user(SCRATCH)
    execute("UPDATE app_user SET status='approved' WHERE id=%s", (uid,))
    token = secrets.token_urlsafe(16)
    execute(
        "INSERT INTO mal_session (token_hash, app_user_id, mal_user_id, access_token,"
        " refresh_token, expires_at) VALUES (%s,%s,%s,'a','r',%s)",
        (auth._hash(token), uid, 1, dt.datetime.now(dt.UTC) + dt.timedelta(days=1)))
    client = TestClient(app)
    client.cookies.set(auth.COOKIE, token)
    yield client, uid
    execute("DELETE FROM app_user WHERE mal_username=%s", (SCRATCH,))


def test_me_reports_the_signed_in_user(signed_in):
    client, _ = signed_in
    assert client.get("/auth/me").json() == {
        "signed_in": True, "username": SCRATCH, "mal_user_id": 1,
        "status": "approved", "is_admin": False, "picture": None}


def test_queue_adds_plan_to_watch_and_mirrors_locally(signed_in):
    client, uid = signed_in
    r = client.post("/me/queue/5114", headers=CSRF)
    assert r.status_code == 200 and r.json()["list_status"] == "plan_to_watch"
    assert FakeMal.calls == [("set", 5114, "plan_to_watch")]
    assert one("SELECT status FROM list_entry WHERE user_id=%s AND mal_id=5114",
               (uid,))["status"] == "plan_to_watch"


def test_queue_never_downgrades_a_show_in_progress(signed_in):
    client, _ = signed_in
    FakeMal.listed = {5114: "watching"}
    r = client.post("/me/queue/5114", headers=CSRF)
    assert r.status_code == 409
    assert FakeMal.calls == [], "a watching entry must not be overwritten"


def test_unqueue_removes_only_plan_to_watch(signed_in):
    client, uid = signed_in
    FakeMal.listed = {5114: "plan_to_watch"}
    r = client.delete("/me/queue/5114", headers=CSRF)
    assert r.status_code == 200 and r.json()["list_status"] is None
    assert FakeMal.calls == [("delete", 5114)]
    assert one("SELECT 1 AS x FROM list_entry WHERE user_id=%s AND mal_id=5114", (uid,)) is None


@pytest.mark.parametrize("status", ["completed", "watching", "on_hold", "dropped"])
def test_unqueue_never_deletes_a_real_entry(signed_in, status):
    client, _ = signed_in
    FakeMal.listed = {5114: status}
    r = client.delete("/me/queue/5114", headers=CSRF)
    assert r.status_code == 409
    assert FakeMal.calls == [], f"a {status} entry was deleted"
    assert FakeMal.listed == {5114: status}


def test_state_changes_need_the_csrf_header(signed_in):
    client, _ = signed_in
    assert client.post("/me/queue/5114").status_code == 403
    assert client.delete("/me/queue/5114").status_code == 403
    assert FakeMal.calls == []


def test_return_to_cannot_leave_the_site():
    for bad in ("//evil.example", "https://evil.example", "/\\evil.example", "", None):
        assert auth._safe_return_to(bad) == "/"
    assert auth._safe_return_to("/?tab=next_up") == "/?tab=next_up"


def test_forged_or_stale_state_is_rejected():
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    r = TestClient(app).get("/auth/callback?code=x&state=forged", follow_redirects=False)
    assert r.status_code == 302 and "auth_error=expired" in r.headers["location"]


def test_login_explains_a_redirect_url_mismatch_instead_of_sending_you_to_it(monkeypatch):
    """MAL answers an unregistered redirect_uri with 401 + WWW-Authenticate,
    which browsers show as a password prompt that can never succeed. The
    login endpoint must catch that and come back with an explanation."""
    import httpx
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")

    class Resp:
        status_code = 401
        text = '{"error":"invalid_client","message":"Client authentication failed"}'
    monkeypatch.setattr(httpx, "get", lambda *a, **k: Resp())
    r = TestClient(app).get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    assert "auth_error=redirect_mismatch" in r.headers["location"]
    assert "myanimelist.net" not in r.headers["location"]


# ---------------------------------------------------------------- rating --

def test_rating_an_unlisted_title_completes_it(signed_in):
    client, uid = signed_in
    r = client.post("/me/rate/5114", headers=CSRF, json={"score": 9})
    assert r.status_code == 200
    assert r.json()["list_status"] == "completed" and r.json()["list_score"] == 9
    kind, mid, fields = FakeMal.calls[0]
    assert (kind, mid, fields["score"], fields["status"]) == ("update", 5114, 9, "completed")
    row = one("SELECT status, score FROM list_entry WHERE user_id=%s AND mal_id=5114", (uid,))
    assert (row["status"], row["score"]) == ("completed", 9)


@pytest.mark.parametrize("status", ["watching", "on_hold", "dropped"])
def test_rating_keeps_the_status_of_a_started_show(signed_in, status):
    client, _ = signed_in
    FakeMal.listed = {5114: status}
    r = client.post("/me/rate/5114", headers=CSRF, json={"score": 6})
    assert r.json()["list_status"] == status
    assert FakeMal.calls == [("update", 5114, {"score": 6})]


def test_rating_validates_the_score(signed_in):
    client, _ = signed_in
    assert client.post("/me/rate/5114", headers=CSRF, json={"score": 11}).status_code == 422
    assert client.post("/me/rate/5114", headers=CSRF, json={"score": 0}).status_code == 422
    assert client.post("/me/rate/5114", json={"score": 7}).status_code == 403
    assert FakeMal.calls == []


# ------------------------------------------------------------------ sync --

def test_manual_sync_has_a_cooldown(signed_in):
    client, _ = signed_in
    first = client.post("/me/sync", headers=CSRF)
    assert first.status_code == 200
    second = client.post("/me/sync", headers=CSRF)
    assert second.status_code == 429
    wait = int(second.headers["Retry-After"])
    assert 0 < wait <= auth.SYNC_COOLDOWN.total_seconds()
    assert client.get("/me/sync").json()["next_allowed_in"] > 0


def test_sync_needs_a_session():
    client = TestClient(app)
    assert client.post("/me/sync", headers=CSRF).status_code == 401


# ----------------------------------------------------------------- prefs --

def test_prefs_round_trip_and_validation(signed_in):
    client, _ = signed_in
    r = client.put("/me/prefs", headers=CSRF, json={"lang": "de",
                                                     "filters": {"eps_max": "26",
                                                                 "media_types": ["tv"]}})
    assert r.status_code == 200
    assert client.get("/me/prefs").json() == {
        "lang": "de", "filters": {"eps_max": 26, "media_types": ["tv"]}}
    # a partial update keeps the rest
    client.put("/me/prefs", headers=CSRF, json={"lang": "en"})
    assert client.get("/me/prefs").json()["filters"]["eps_max"] == 26
    assert client.put("/me/prefs", headers=CSRF, json={"lang": "fr"}).status_code == 422
    assert client.put("/me/prefs", headers=CSRF,
                      json={"filters": {"eps_max": "x"}}).status_code == 422


# ----------------------------------------------------------- rating round --

def test_quiz_offers_unlisted_titles_and_never_repeats(signed_in):
    client, uid = signed_in
    execute("INSERT INTO list_entry (user_id, mal_id, status, score) VALUES (%s, 5114, 'completed', 9)",
            (uid,))
    cards = client.get("/me/quiz?n=5").json()["cards"]
    ids = [c["mal_id"] for c in cards]
    assert ids and 5114 not in ids
    assert client.post(f"/me/quiz/{ids[0]}/unseen", headers=CSRF).status_code == 200
    assert client.post(f"/me/quiz/{ids[1]}/skip", headers=CSRF).status_code == 200
    assert client.post(f"/me/quiz/{ids[2]}/maybe", headers=CSRF).status_code == 422
    again = [c["mal_id"] for c in client.get("/me/quiz?n=5").json()["cards"]]
    assert ids[0] not in again and ids[1] not in again


def test_quiz_rating_is_tagged(signed_in):
    client, uid = signed_in
    r = client.post("/me/rate/5114?source=quiz", headers=CSRF, json={"score": 8})
    assert r.status_code == 200
    assert one("SELECT surface FROM feedback WHERE user_id=%s AND action='rated'",
               (uid,))["surface"] == "quiz"
    assert client.get("/me/quiz").json()["rated_in_round"] == 1
