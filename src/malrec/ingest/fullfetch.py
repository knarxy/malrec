"""One-time full fetch of everything MAL and AniList can contribute.

Authorised as a single catalogue-wide pass. Two rules make that compatible
with fair use:

  * every response is written to Postgres the moment it arrives, and every
    item is stamped, so no request is ever repeated - not after a crash, not
    on a re-run, not ever;
  * requests stay under the rates measured to be safe (MAL ~1.5/s, AniList
    ~28/min), with backoff on any sign of load shedding.

Stages, in dependency order:

    MAL      mal_catalogue   every anime, via /anime/ranking?bypopularity   ~65 req
             cf_discover     usernames from forum topic listings             ~100 req
             cf_lists        those users' public lists                       ~5-6k req
             mal_details     rec graph + relations + stats, real audience    ~several k
    AniList  anilist_catalogue  tags, staff, stats, relations, 50 per req    ~600 req

The two hosts are rate-limited independently, so `--part mal` and
`--part anilist` run as separate processes in parallel.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from psycopg.types.json import Jsonb

from ..clients.mal import MalClient, _date, normalise_node
from ..db import conn, execute, one, query, refresh_franchises, scalar
from . import store

log = logging.getLogger(__name__)

CF_DISCOVER_TARGET = 6000        # usernames; ~88% turn out to have public lists
MAL_DETAILS_MIN_MEMBERS = 5000   # below this nothing is ever recommended anyway
FLUSH_EVERY = 50


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def name_hash(name: str) -> str:
    return hashlib.sha256(name.strip().lower().encode()).hexdigest()


# ------------------------------------------------------------ stage state --

def _state(stage: str) -> dict:
    row = one("SELECT cursor, done FROM fetch_state WHERE stage=%s", (stage,))
    return {"cursor": row["cursor"], "done": row["done"]} if row else {"cursor": {}, "done": False}


def _save(stage: str, cursor: dict, done: bool = False) -> None:
    execute(
        "INSERT INTO fetch_state (stage, cursor, done, updated_at) VALUES (%s,%s,%s,now())"
        " ON CONFLICT (stage) DO UPDATE SET cursor=EXCLUDED.cursor, done=EXCLUDED.done,"
        " updated_at=now()",
        (stage, Jsonb(cursor), done),
    )


# ------------------------------------------------------------- raw access --

def _raw_get(mal: MalClient, path: str, params: dict, tries: int = 6) -> tuple[int, dict]:
    """Like MalClient.get, but reports the final status instead of hiding it.

    The list fetch has to tell "this list is private" (403, permanent, stamp
    it) apart from "MAL is shedding load" (307/5xx, retry later), and the
    normal client deliberately collapses both into an empty dict.
    """
    for attempt in range(tries):
        mal._throttle()
        try:
            r = mal._http.get(path, params=params)
        except (httpx.TimeoutException, httpx.TransportError):
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            return 200, r.json()
        if r.status_code in (403, 404):
            return r.status_code, {}
        # 307 to error.json, 429 and 5xx are all transient load shedding
        time.sleep(min(60, 3 * 2 ** attempt))
    return 0, {}


# ---------------------------------------------------------- MAL catalogue --

def mal_catalogue(mal: MalClient) -> dict:
    """Walk /anime/ranking?ranking_type=bypopularity to the end.

    Popularity order reaches every anime anyone has listed, 500 at a time
    with full metadata, so the whole catalogue is ~65 requests.
    """
    st = _state("mal_catalogue")
    if st["done"]:
        return {"skipped": "already complete"}
    offset = int(st["cursor"].get("offset", 0))
    total = int(st["cursor"].get("total", 0))
    while True:
        status, d = _raw_get(mal, "/anime/ranking", {
            "ranking_type": "bypopularity", "limit": 500, "offset": offset,
            "nsfw": "true", "fields": "id,title,main_picture,alternative_titles,start_date,"
            "end_date,synopsis,mean,rank,popularity,num_list_users,num_scoring_users,nsfw,"
            "genres,media_type,status,num_episodes,start_season,source,"
            "average_episode_duration,rating,studios"})
        if status != 200:
            log.warning("catalogue page at offset %d failed (%s); will resume here", offset, status)
            _save("mal_catalogue", {"offset": offset, "total": total})
            return {"total": total, "stopped_at": offset}
        rows = d.get("data", [])
        store.upsert_anime([normalise_node(e["node"]) | {"mal_fetched_at": _now()} for e in rows])
        total += len(rows)
        offset += 500
        _save("mal_catalogue", {"offset": offset, "total": total})
        if offset % 5000 == 0:
            log.info("catalogue: %d anime stored", total)
        if len(rows) < 500 or "next" not in d.get("paging", {}):
            _save("mal_catalogue", {"offset": offset, "total": total}, done=True)
            log.info("catalogue complete: %d anime", total)
            return {"total": total}


# ----------------------------------------------------------- CF discovery --

def _app_user_hashes() -> set[str]:
    return {name_hash(r["mal_username"]) for r in query("SELECT mal_username FROM app_user")}


def cf_discover(mal: MalClient, target: int = CF_DISCOVER_TARGET) -> dict:
    """Collect usernames from forum topic listings.

    Each listing names the author and the latest poster of up to 100 topics,
    ~60 new users per request. App users are recorded as 'excluded' so their
    own lists can never leak into the data used to evaluate them.
    """
    st = _state("cf_discover")
    if st["done"]:
        return {"skipped": "already complete"}
    have = scalar("SELECT count(*) FROM cf_user") or 0
    if have >= target:
        _save("cf_discover", st["cursor"], done=True)
        return {"users": have}

    boards = st["cursor"].get("boards")
    if not boards:
        _, b = _raw_get(mal, "/forum/boards", {})
        boards = []
        for cat in b.get("categories", []):
            for board in cat.get("boards", []):
                boards.append({"board_id": board["id"], "offset": 0, "done": False})
                for sub in board.get("subboards", []) or []:
                    boards.append({"board_id": board["id"], "subboard_id": sub["id"],
                                   "offset": 0, "done": False})
    excluded = _app_user_hashes()

    while have < target and any(not b["done"] for b in boards):
        for b in boards:
            if b["done"] or have >= target:
                continue
            params = {"board_id": b["board_id"], "limit": 100, "offset": b["offset"]}
            if b.get("subboard_id"):
                params["subboard_id"] = b["subboard_id"]
            status, d = _raw_get(mal, "/forum/topics", params)
            topics = d.get("data", []) if status == 200 else []
            names = set()
            for t in topics:
                for k in ("created_by", "last_post_created_by"):
                    n = (t.get(k) or {}).get("name")
                    if n:
                        names.add(n)
            if names:
                rows = [(name_hash(n), None if name_hash(n) in excluded else n,
                         "excluded" if name_hash(n) in excluded else "pending") for n in names]
                with conn() as c, c.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO cf_user (name_hash, name, state) VALUES (%s,%s,%s)"
                        " ON CONFLICT (name_hash) DO NOTHING", rows)
                    c.commit()
            b["offset"] += 100
            # deep pages of quiet boards stop yielding anyone new
            if len(topics) < 100 or b["offset"] >= 3000:
                b["done"] = True
            have = scalar("SELECT count(*) FROM cf_user") or 0
            _save("cf_discover", {"boards": boards})
        log.info("discovery: %d users known", have)

    _save("cf_discover", {"boards": boards}, done=True)
    return {"users": have}


# ---------------------------------------------------------------- CF lists --

# Requests start at exactly the configured rate (the client's limiter hands out
# slots under a lock); a few workers only let slow responses overlap instead of
# each one delaying the next start.
WORKERS = 3


def _fetch_list(mal: MalClient, name: str) -> tuple[int, list[dict]]:
    entries: list[dict] = []
    offset, status = 0, 200
    while True:
        status, d = _raw_get(mal, f"/users/{name}/animelist", {
            "fields": "list_status", "limit": 1000, "offset": offset, "nsfw": "true"})
        if status != 200:
            break
        entries.extend(d.get("data", []))
        if "next" not in d.get("paging", {}):
            break
        offset += 1000
    return status, entries


def _store_list(u: dict, entries: list[dict]) -> bool:
    """One user per transaction; the username is dropped once stored."""
    rows = []
    for e in entries:
        ls = e.get("list_status") or {}
        # MAL returns partial dates ("2015", "2015-04"); widen them
        rows.append((u["id"], e["node"]["id"], ls.get("score") or 0,
                     ls.get("status") or "unknown", _date(ls.get("finish_date")),
                     ls.get("updated_at")))
    try:
        with conn() as c, c.cursor() as cur:
            cur.executemany(
                "INSERT INTO cf_rating (user_id, mal_id, score, status, finished_at,"
                " updated_at) VALUES (%s,%s,%s,%s,%s,%s)"
                " ON CONFLICT (user_id, mal_id) DO NOTHING", rows)
            cur.execute(
                "UPDATE cf_user SET state='done', name=NULL, fetched_at=now(),"
                " n_entries=%s, n_scored=%s WHERE id=%s",
                (len(rows), sum(1 for r in rows if r[2] > 0), u["id"]))
            c.commit()
    except Exception as e:  # noqa: BLE001
        # One malformed list must not end a multi-hour run. The name is kept
        # so the user can be retried once the cause is fixed.
        log.error("could not store list for cf_user %s: %s", u["id"], e)
        execute("UPDATE cf_user SET state='error' WHERE id=%s", (u["id"],))
        return False
    return True


def cf_lists(mal: MalClient) -> dict:
    """Fetch every pending user's public list."""
    excluded = _app_user_hashes()
    pending = query("SELECT id, name, name_hash FROM cf_user WHERE state='pending' ORDER BY id")
    for u in pending:
        if u["name_hash"] in excluded:
            execute("UPDATE cf_user SET state='excluded', name=NULL WHERE id=%s", (u["id"],))
    pending = [u for u in pending if u["name_hash"] not in excluded]
    log.info("lists: %d pending", len(pending))

    done = private = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_fetch_list, mal, u["name"]): u for u in pending}
        for i, fut in enumerate(as_completed(futures)):
            u = futures[fut]
            status, entries = fut.result()
            if status in (403, 404):
                execute("UPDATE cf_user SET state='private', name=NULL, fetched_at=now()"
                        " WHERE id=%s", (u["id"],))
                private += 1
            elif status == 200 and _store_list(u, entries):
                done += 1
            # anything else is transient: the user stays pending for a later run
            if (i + 1) % 250 == 0:
                log.info("lists: %d/%d processed (%d stored, %d private)",
                         i + 1, len(pending), done, private)
    return {"stored": done, "private": private}


# ------------------------------------------------------------- MAL details --

def mal_details(mal: MalClient, min_members: int = MAL_DETAILS_MIN_MEMBERS,
                targets: list[int] | None = None) -> dict:
    """Recommendation graph, relations and status statistics for every anime
    with a real audience that has not been fetched yet."""
    from .jobs import _collect

    if targets is None:
        targets = [r["mal_id"] for r in query(
            "SELECT mal_id FROM anime WHERE graph_fetched_at IS NULL"
            " AND coalesce(mal_num_list_users, 0) >= %s ORDER BY mal_popularity NULLS LAST",
            (min_members,))]
    log.info("details: %d anime pending (~%.0f min)", len(targets), len(targets) / 1.5 / 60)

    rec_edges: list = []
    rels: list = []
    nodes: list = []
    stats: list = []
    done_ids: list[int] = []
    fetched = 0

    def flush() -> None:
        nonlocal rec_edges, rels, nodes, stats, done_ids
        if not done_ids:
            return
        store.upsert_anime([x for x in nodes if x.get("title")])
        store.upsert_rec_edges(rec_edges)
        store.upsert_relations(rels)
        store.store_mal_stats(stats)
        store.mark_graph_fetched(done_ids)
        rec_edges, rels, nodes, stats, done_ids = [], [], [], [], []

    def fetch(aid: int) -> tuple[int, int, dict]:
        status, d = _raw_get(mal, f"/anime/{aid}", {
            "fields": "id,title,recommendations,related_anime,statistics,num_list_users"})
        return aid, status, d

    pool = ThreadPoolExecutor(max_workers=WORKERS)
    try:
        for i, fut in enumerate(as_completed([pool.submit(fetch, a) for a in targets])):
            aid, status, d = fut.result()
            if status == 200 and d:
                fetched += 1
                _collect(aid, d, rec_edges, rels, nodes)
                st = ((d.get("statistics") or {}).get("status")) or {}
                dist = {k: int(v) for k, v in st.items() if str(v).isdigit()}
                started = sum(dist.get(k, 0) for k in ("watching", "completed", "on_hold", "dropped"))
                drop = dist.get("dropped", 0) / started if started >= 50 else None
                stats.append((aid, dist or None, drop, d))
                done_ids.append(aid)
            elif status == 404:
                done_ids.append(aid)          # gone for good
            if len(done_ids) >= FLUSH_EVERY:
                flush()
            if (i + 1) % 500 == 0:
                log.info("details: %d/%d", i + 1, len(targets))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        flush()
    if fetched:
        refresh_franchises()
    return {"fetched": fetched}


# ---------------------------------------------------------------- AniList --

def anilist_catalogue() -> dict:
    """AniList for every anime in the catalogue that lacks it."""
    from .ondemand import ensure_anilist
    ids = [r["mal_id"] for r in query(
        "SELECT mal_id FROM anime WHERE al_fetched_at IS NULL"
        " ORDER BY mal_popularity NULLS LAST")]
    log.info("anilist: %d anime pending (~%.0f min)", len(ids), math.ceil(len(ids) / 50) / 28)
    return {"fetched": ensure_anilist(ids)}


# --------------------------------------------------------------- backfill --
#
# Both stages above select what was never fetched. Anime enriched on demand
# before the full fetch existed were therefore skipped, although their stored
# payloads predate the fields it added (status/score distributions, staff) -
# and those are mostly the popular titles. This re-requests exactly those, once.

def backfill(mal: MalClient, min_members: int = MAL_DETAILS_MIN_MEMBERS) -> dict:
    from .ondemand import ensure_anilist
    out: dict = {}
    al_ids = [r["mal_id"] for r in query("""
        SELECT a.mal_id FROM anime a
         WHERE a.al_fetched_at IS NOT NULL AND a.al_score_dist IS NULL
           AND (a.al_average_score IS NOT NULL OR a.al_favourites IS NOT NULL
                OR EXISTS (SELECT 1 FROM anime_tag t WHERE t.mal_id = a.mal_id))
         ORDER BY a.mal_popularity NULLS LAST""")]
    log.info("backfill: %d anime lack AniList statistics", len(al_ids))
    out["anilist"] = ensure_anilist(al_ids, force=True)
    mal_ids = [r["mal_id"] for r in query(
        "SELECT mal_id FROM anime WHERE graph_fetched_at IS NOT NULL AND mal_status_dist IS NULL"
        " AND coalesce(mal_num_list_users, 0) >= %s ORDER BY mal_popularity NULLS LAST",
        (min_members,))]
    log.info("backfill: %d anime lack MAL statistics", len(mal_ids))
    out["mal"] = mal_details(mal, targets=mal_ids)
    return out


# ---------------------------------------------------------------- refresh --
#
# Usernames are dropped once a list is stored, so a sampled user can never be
# re-read. The sample is kept current by rotation instead: add a capped batch
# of new users, then retire the same number of the oldest lists.
#
# New users come from the *replies* in the newest anime discussion topics, not
# just topic starters and last posters: episode threads are where casual
# viewers post, which pulls the sample away from forum regulars.

REFRESH_NEW_USERS = 800
REFRESH_RETIRE_DAYS = 365


def discover_from_posts(mal: MalClient, want: int, topics_per_board: int = 40) -> int:
    _, b = _raw_get(mal, "/forum/boards", {})
    boards = [bd for cat in b.get("categories", []) for bd in cat.get("boards", [])
              if "anime" in (bd.get("title") or "").lower()]
    excluded = _app_user_hashes()
    added = 0
    for bd in boards:
        if added >= want:
            break
        status, d = _raw_get(mal, "/forum/topics", {"board_id": bd["id"], "limit": 100})
        topics = (d.get("data") or [])[:topics_per_board] if status == 200 else []
        for t in topics:
            if added >= want:
                break
            status, p = _raw_get(mal, f"/forum/topic/{t['id']}", {"limit": 100})
            posts = ((p.get("data") or {}).get("posts") or []) if status == 200 else []
            names = {(x.get("created_by") or {}).get("name") for x in posts} - {None, ""}
            rows = [(name_hash(n), None if name_hash(n) in excluded else n,
                     "excluded" if name_hash(n) in excluded else "pending") for n in names]
            if not rows:
                continue
            with conn() as c, c.cursor() as cur:
                for row in rows:
                    cur.execute(
                        "INSERT INTO cf_user (name_hash, name, state, source)"
                        " VALUES (%s,%s,%s,'mal_topic_posts') ON CONFLICT (name_hash) DO NOTHING"
                        " RETURNING id", row)
                    if cur.fetchone() and row[2] == "pending":
                        added += 1
                c.commit()
    log.info("refresh: %d new users from topic replies", added)
    return added


def cf_refresh(mal: MalClient, new_users: int = REFRESH_NEW_USERS,
               retire_days: int = REFRESH_RETIRE_DAYS) -> dict:
    """One rotation: capped discovery, fetch the new lists, retire as many of
    the oldest lists (older than `retire_days`) as were added. About
    `new_users` + ~50 requests - a quarter of an hour at 1 req/s."""
    found = discover_from_posts(mal, new_users)
    lists = cf_lists(mal)
    stored = lists.get("stored", 0)
    retired = query("""
        UPDATE cf_user SET state = 'retired' WHERE id IN (
            SELECT id FROM cf_user WHERE state = 'done'
               AND fetched_at < now() - %s::interval
             ORDER BY fetched_at LIMIT %s)
        RETURNING id""", (f"{int(retire_days)} days", stored))
    if retired:
        execute("DELETE FROM cf_rating WHERE user_id = ANY(%s)", ([r["id"] for r in retired],))
    out = {"discovered": found, **lists, "retired": len(retired)}
    log.info("cf refresh: %s", out)
    return out


# ----------------------------------------------------------------- runner --

def run(part: str) -> dict:
    out: dict = {}
    if part in ("mal", "all"):
        with MalClient() as mal:
            out["mal_catalogue"] = mal_catalogue(mal)
            out["cf_discover"] = cf_discover(mal)
            out["cf_lists"] = cf_lists(mal)
            out["mal_details"] = mal_details(mal)
    if part in ("anilist", "all"):
        out["anilist_catalogue"] = anilist_catalogue()
    if part == "backfill":
        with MalClient() as mal:
            out["backfill"] = backfill(mal)
    log.info("full fetch (%s) finished: %s", part, out)
    return out

