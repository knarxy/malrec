"""The background task queue (malrec.tasks) and the bounded model cache."""
from __future__ import annotations

import pytest

from malrec import model, tasks
from malrec.config import settings
from malrec.db import execute, one, scalar
from malrec.ingest.store import get_or_create_user

NAMES = ("malrec_t_task_a", "malrec_t_task_b")


@pytest.fixture()
def users(monkeypatch):
    try:
        scalar("SELECT 1 FROM task LIMIT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    monkeypatch.setattr(settings(), "tasks_inline", False)
    ids = [get_or_create_user(n) for n in NAMES]
    execute("DELETE FROM task WHERE user_id = ANY(%s)", (ids,))
    yield ids
    for n in NAMES:
        execute("DELETE FROM app_user WHERE mal_username=%s", (n,))


def _mine(ids, sql="SELECT id, kind, user_id, state FROM task WHERE user_id = ANY(%s) ORDER BY id"):
    from malrec.db import query
    return query(sql, (ids,))


def test_a_waiting_task_absorbs_repeats(users):
    a, _ = users
    for _ in range(5):
        tasks.enqueue("rebuild", a)
    assert len(_mine(users)) == 1
    tasks.enqueue("sync", a)
    assert sorted(t["kind"] for t in _mine(users)) == ["rebuild", "sync"]


def test_one_running_task_per_user(users, monkeypatch):
    a, b = users
    tasks.enqueue("rebuild", a)
    tasks.enqueue("sync", a)
    tasks.enqueue("rebuild", b)
    # only claim our own test rows, whatever else is queued on this database
    monkeypatch.setattr(tasks, "CLAIM_SQL", tasks.CLAIM_SQL.replace(
        "WHERE t.state = 'queued'", f"WHERE t.state = 'queued' AND t.user_id IN ({a}, {b})"))
    first, second = tasks.claim(), tasks.claim()
    assert {first["user_id"], second["user_id"]} == {a, b}, "a's second task waits for its first"
    assert tasks.claim() is None
    tasks.finish(first["id"])
    third = tasks.claim()
    assert third is not None and third["user_id"] == a


def test_state_follows_the_latest_task(users):
    a, _ = users
    assert tasks.state(a)["refresh"] == "idle"
    tasks.enqueue("sync", a)
    assert tasks.state(a)["refresh"] == "syncing"
    tid = one("SELECT id FROM task WHERE user_id=%s", (a,))["id"]
    tasks.finish(tid, "MalApiError: 503")
    st = tasks.state(a)
    assert st["refresh"] == "failed" and "503" in st["refresh_error"]
    tasks.enqueue("rebuild", a)
    execute("UPDATE task SET created_at = now() - interval '1 hour' WHERE user_id=%s"
            " AND state='queued'", (a,))
    assert tasks.state(a)["refresh"] == "timeout"


def test_recover_requeues_what_a_dead_worker_left(users):
    a, b = users
    tasks.enqueue("rebuild", a)
    tasks.enqueue("rebuild", b)
    execute("UPDATE task SET state='running', attempts=1 WHERE user_id = ANY(%s)", (users,))
    tasks.enqueue("rebuild", b)                       # b already has a queued twin
    tasks.recover()
    by_user = {(t["user_id"], t["state"]) for t in _mine(users)}
    assert (a, "queued") in by_user
    assert (b, "failed") in by_user and (b, "queued") in by_user


def test_inline_mode_runs_without_a_worker(monkeypatch):
    ran = []
    monkeypatch.setattr(settings(), "tasks_inline", True)
    monkeypatch.setattr(tasks, "run_task", lambda kind, uid: ran.append((kind, uid)))
    tasks.enqueue("rebuild", 7)
    import time
    for _ in range(50):
        if ran:
            break
        time.sleep(0.02)
    assert ran == [("rebuild", 7)]
    with pytest.raises(ValueError):
        tasks.enqueue("nonsense", 7)


def test_model_cache_is_bounded_and_follows_the_list(monkeypatch):
    version = {"n": 1}
    loads = []
    monkeypatch.setattr(model, "one", lambda sql, p: {"run": 1, "pop": 1, "n": version["n"],
                                                     "at": None})
    monkeypatch.setattr(model, "load_or_train", lambda uid: (loads.append(uid) or f"m{uid}", 1))
    monkeypatch.setattr(settings(), "scorer_cache_size", 3)
    model._scorers.clear()
    for uid in (1, 2, 3, 1, 4):            # 1 is used again, so 2 is the oldest
        model.cached_model(uid)
    assert list(model._scorers) == [3, 1, 4]
    assert loads == [1, 2, 3, 4]
    version["n"] = 2                       # the list changed (e.g. the worker rebuilt)
    model.cached_model(4)
    assert loads[-1] == 4 and len(loads) == 5
    model._scorers.clear()
