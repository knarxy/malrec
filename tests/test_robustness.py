"""Failure paths found in the 2026-10-23 audit: one bad date, one half-typed
filter or one bad minute at MAL must not break a user's lists or session, and
the scheduled jobs touch approved accounts only. No database or network."""
from __future__ import annotations

import httpx
import pytest
from fastapi import HTTPException

from malrec import auth, db, refresh
from malrec.clients.mal import _date


def test_partial_dates_widen_and_impossible_ones_are_dropped():
    assert _date("2013") == "2013-01-01"
    assert _date("2013-04") == "2013-04-01"
    assert _date("2020-02-29") == "2020-02-29"
    for bad in ("2020-02-30", "2020-04-31", "0000-00-00", "2020-13", "1999-12-32", "x", ""):
        assert _date(bad) is None, bad
    assert _date(None) is None


def test_saved_filters_keep_only_what_the_list_endpoint_accepts():
    out = auth._clean_filters({"year_min": "95", "year_max": 2024, "eps_min": 0,
                               "eps_max": 20000, "media_types": ["tv"]})
    assert out == {"year_max": 2024, "eps_min": 0, "media_types": ["tv"]}
    with pytest.raises(HTTPException):
        auth._clean_filters({"year_min": "nineteen"})


class _Resp:
    def __init__(self, status: int, body: dict | None = None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


@pytest.fixture()
def token_endpoint(monkeypatch):
    """MAL's token endpoint answering `answer[0]`; the SQL the refresh ran."""
    answer: list = [None]
    ran: list[str] = []

    def post(*a, **kw):
        if isinstance(answer[0], Exception):
            raise answer[0]
        return answer[0]
    monkeypatch.setattr(auth.httpx, "post", post)
    monkeypatch.setattr(auth, "execute", lambda sql, *a, **kw: ran.append(sql))
    return answer, ran


ROW = {"token_hash": "h", "refresh_token": "r", "access_token": "a"}


@pytest.mark.parametrize("status", [400, 401])
def test_a_refused_grant_ends_the_session(token_endpoint, status):
    answer, ran = token_endpoint
    answer[0] = _Resp(status)
    assert auth._refresh(ROW) is None
    assert any(s.startswith("DELETE FROM mal_session") for s in ran)


@pytest.mark.parametrize("failure", [_Resp(429), _Resp(503), _Resp(200, {}),
                                     httpx.ConnectError("down")])
def test_a_busy_or_failing_mal_keeps_the_session(token_endpoint, failure):
    answer, ran = token_endpoint
    answer[0] = failure
    assert auth._refresh(ROW) is None
    assert not any(s.startswith("DELETE") for s in ran)


def test_a_fresh_token_is_stored(token_endpoint):
    answer, ran = token_endpoint
    answer[0] = _Resp(200, {"access_token": "new", "refresh_token": "r2", "expires_in": 3600})
    row = auth._refresh(ROW)
    assert row is not None and row["access_token"] == "new"
    assert any(s.startswith("UPDATE mal_session") for s in ran)


def test_the_nightly_sync_reads_approved_accounts_only(monkeypatch):
    asked: list[str] = []
    monkeypatch.setattr(db, "query", lambda sql, *a, **kw: asked.append(sql) or [])
    assert refresh.sync_all_users() == []
    assert asked and "status = 'approved'" in asked[0]


def test_a_failed_rebuild_is_not_forgotten(monkeypatch):
    """ING-4: a sync that leaves the list as it was still rebuilds when the
    last finished build used another list."""
    now, built = {"fp": "B"}, {"v": "A"}
    monkeypatch.setattr(refresh, "fingerprint", lambda uid: now["fp"])
    monkeypatch.setattr(db, "scalar", lambda sql, *a, **kw: built["v"])
    assert refresh.stale(1, before="B")             # unchanged now, but built from A
    built["v"] = "B"
    assert not refresh.stale(1, before="B")
    assert refresh.stale(1, before="A")             # changed by this sync
    built["v"] = None
    assert not refresh.stale(1, before="B")         # never built: only a change rebuilds


def test_a_filter_that_leaves_nothing_gives_an_empty_list(monkeypatch):
    from malrec import surfaces
    row = {"mal_id": 1, "list_status": None, "media_type": "tv", "num_episodes": 12,
           "season_year": 2020, "mal_genres": ["Drama"]}
    monkeypatch.setattr(surfaces, "query", lambda *a, **kw: [row])
    only_ova = surfaces.Filters(media_types=["ova"])
    for s in surfaces.RERANKED:
        assert surfaces.read_ranked(1, s, filters=only_ova) == []
    monkeypatch.setattr(surfaces, "query", lambda *a, **kw: [{**row, "list_status": "completed"}])
    assert surfaces.read_ranked(1, "safe_bets") == []


@pytest.fixture()
def forum(monkeypatch):
    """expand_discover against one board: a full first listing page, an empty
    second one; `reply` is the status every reply fetch gets."""
    from malrec.ingest import fullfetch
    calls, saved = [], []
    state = {"listing": 200, "reply": 500}

    def raw_get(mal, path, params, tries=6):
        calls.append((path, params.get("offset")))
        if path == "/forum/topics":
            if state["listing"] != 200:
                return state["listing"], {}
            n = 100 if params["offset"] == 0 else 0
            return 200, {"data": [{"id": k, "created_by": {"name": f"u{k}"}} for k in range(n)]}
        return state["reply"], {}
    monkeypatch.setattr(fullfetch, "_raw_get", raw_get)
    monkeypatch.setattr(fullfetch, "_state", lambda stage: {
        "cursor": {"boards": [{"board_id": 1, "offset": 0, "done": False}]}})
    monkeypatch.setattr(fullfetch, "_save", lambda stage, cur, done=False: saved.append(cur))
    monkeypatch.setattr(fullfetch, "_app_user_hashes", set)
    monkeypatch.setattr(fullfetch, "_add_names", lambda names, excl, src: len(names))
    return fullfetch, state, calls, saved


def test_a_failed_reply_fetch_does_not_end_the_board(forum):
    fullfetch, _, calls, saved = forum
    assert fullfetch.expand_discover(None, want=10 ** 6) == 100
    assert ("/forum/topics", 100) in calls          # the second listing page was read
    assert saved[-1]["boards"][0]["done"]


def test_a_failed_listing_keeps_the_board_for_the_next_run(forum):
    fullfetch, state, _, saved = forum
    state["listing"] = 503
    assert fullfetch.expand_discover(None, want=10 ** 6) == 0
    board = saved[-1]["boards"][0]
    assert board["offset"] == 0 and not board["done"]


def test_anilist_sync_stamps_only_ids_it_never_answered_for(monkeypatch):
    from malrec.ingest import jobs

    class Stop(Exception):
        pass

    class FakeAniList:
        unsure: frozenset = frozenset()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def media_by_mal_ids(self, ids):
            return {}                                  # a failed batch

    def query(sql, *a, **kw):
        if "anilist_id IS NOT NULL" in sql:
            return [{"mal_id": 2}]                     # 2 has AniList data already
        return [{"mal_id": 1}, {"mal_id": 2}]
    stamped: list[int] = []

    def mark(ids):
        stamped.extend(ids)
        raise Stop
    monkeypatch.setattr(jobs, "AniListClient", FakeAniList)
    monkeypatch.setattr(jobs, "query", query)
    monkeypatch.setattr(jobs.store, "mark_anilist_attempted", mark)
    with pytest.raises(Stop):
        jobs.sync_anilist()
    assert stamped == [1]


# ------------------------------------------------- MAL answers (ING-2/7) --

def _mal(monkeypatch, answer):
    """A MalClient whose every GET is answered by `answer(request)`."""
    from malrec.clients import mal as malmod
    monkeypatch.setattr(malmod.time, "sleep", lambda s: None)
    m = malmod.MalClient(client_id="test")
    m._min_interval = 0.0
    m._http = httpx.Client(base_url=malmod.BASE, transport=httpx.MockTransport(answer))
    return m


@pytest.mark.parametrize("status", [400, 401, 403])
def test_a_refusal_is_an_error_not_an_empty_answer(monkeypatch, status):
    from malrec.clients.mal import MalApiError
    m = _mal(monkeypatch, lambda req: httpx.Response(status, json={"error": "x"}))
    with pytest.raises(MalApiError) as e:
        m.get("/anime/1")
    assert e.value.status == status


def test_only_a_404_is_gone(monkeypatch):
    m = _mal(monkeypatch, lambda req: httpx.Response(404, json={}))
    assert m.get("/anime/1") == {}


def test_a_list_is_whole_or_an_error(monkeypatch):
    from malrec.clients.mal import MalApiError
    page = {"data": [{"node": {"id": 1}, "list_status": {"status": "completed"}}],
            "paging": {"next": "more"}}

    def answer(req):
        return (httpx.Response(200, json=page) if req.url.params["offset"] == "0"
                else httpx.Response(403, json={}))
    with pytest.raises(MalApiError):                    # page 2 refused: no half list
        _mal(monkeypatch, answer).user_animelist("someone")
    with pytest.raises(MalApiError) as e:               # MAL does not know the name
        _mal(monkeypatch, lambda req: httpx.Response(404, json={})).user_animelist("gone")
    assert e.value.status == 404
    one_page = {"data": page["data"], "paging": {}}
    assert len(_mal(monkeypatch, lambda req: httpx.Response(200, json=one_page))
               .user_animelist("someone")) == 1


def test_details_stop_after_a_run_of_failures(monkeypatch):
    from malrec.clients import mal as malmod
    calls = []

    def answer(req):
        calls.append(req.url.path)
        return httpx.Response(401, json={})             # e.g. a revoked client id
    out = []
    with pytest.raises(RuntimeError, match="in a row"):
        for _, d in _mal(monkeypatch, answer).details_many(list(range(100))):
            out.append(d)
    assert len(calls) == malmod.MAX_FAILURES_IN_A_ROW and out == [None] * (len(calls) - 1)

    def flaky(req):                                     # every other id fails: carries on
        n = int(req.url.path.rsplit("/", 1)[1])
        return httpx.Response(200, json={"id": n}) if n % 2 else httpx.Response(503)
    got = list(_mal(monkeypatch, flaky).details_many(list(range(30))))
    assert len(got) == 30 and sum(d is None for _, d in got) == 15


def test_retry_after_reads_seconds_and_dates():
    import datetime as dt
    from email.utils import format_datetime

    from malrec.clients.mal import RETRY_AFTER_MAX, _retry_after
    assert _retry_after("7") == 7.0
    soon = format_datetime(dt.datetime.now(dt.UTC) + dt.timedelta(seconds=30), usegmt=True)
    assert 25 <= (_retry_after(soon) or 0) <= 31
    assert _retry_after("86400") == RETRY_AFTER_MAX
    assert _retry_after("soon") is None and _retry_after(None) is None


def test_ids_from_an_answer_with_errors_are_not_stamped(monkeypatch):
    """ING-10: missing from an answer that carried GraphQL errors is not
    "AniList does not know it"."""
    from malrec.ingest import ondemand

    class FakeAniList:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def media_by_mal_ids(self, ids):
            self.unsure = set(ids)                     # the whole batch came with errors
            return {}
    stamped: list[int] = []
    monkeypatch.setattr(ondemand, "AniListClient", FakeAniList)
    monkeypatch.setattr(ondemand, "_missing", lambda ids, col, age: list(ids))
    monkeypatch.setattr(ondemand, "query", lambda *a, **kw: [])
    monkeypatch.setattr(ondemand.store, "mark_anilist_attempted", stamped.extend)
    monkeypatch.setattr(ondemand.store, "store_tag_vectors", lambda **kw: 0)
    assert ondemand.ensure_anilist([1, 2, 3]) == 0
    assert stamped == []


def test_a_failing_bulk_fetch_stops_instead_of_draining_the_queue(monkeypatch):
    """ING-8: the remaining lists are dropped, not fetched and thrown away."""
    import time

    from malrec.ingest import fullfetch
    calls: list[str] = []

    def fetch(mal, name):
        calls.append(name)
        time.sleep(0.02)
        if name == "u0":
            raise ValueError("not JSON")
        return 500, []
    users = [{"id": i, "name": f"u{i}", "name_hash": str(i)} for i in range(60)]
    monkeypatch.setattr(fullfetch, "_app_user_hashes", set)
    monkeypatch.setattr(fullfetch, "query", lambda *a, **kw: users)
    monkeypatch.setattr(fullfetch, "_fetch_list", fetch)
    with pytest.raises(ValueError):
        fullfetch.cf_lists(None, workers=3)
    time.sleep(0.2)
    assert len(calls) < 20


def test_users_see_failures_in_general_terms():
    """FE-5: exception texts stay with the admin."""
    from malrec import onboarding, tasks
    assert tasks.public_error("OperationalError: connection to server at 10.0.0.5 failed") \
        == "server"
    assert tasks.public_error("ListShrank: MAL returned 0 entries") == "list_shrank"
    assert tasks.public_error("RuntimeError: MAL request failed after 5 attempts") \
        == "mal_unavailable"
    assert tasks.public_error(None) is None
    assert onboarding.public_error("OnboardError: x's list could not be read") \
        == "x's list could not be read"
    assert onboarding.public_error("KeyError: 'data'") == onboarding.SETUP_FAILED
