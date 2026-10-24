"""MyAnimeList API v2 client.

Only the client-ID header is used. That is enough for anime search, details,
rankings, seasonal listings and - importantly - reading any *public* user list,
so no OAuth flow, token store or refresh loop is needed.
"""
from __future__ import annotations

import datetime as dt
import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)

BASE = "https://api.myanimelist.net/v2"

# Everything useful the detail endpoint will return for a list/ranking node.
NODE_FIELDS = (
    "id,title,main_picture,alternative_titles,start_date,end_date,synopsis,mean,rank,"
    "popularity,num_list_users,num_scoring_users,nsfw,genres,media_type,status,"
    "num_episodes,start_season,source,average_episode_duration,rating,studios"
)
# Detail-only extras: the recommendation graph and franchise relations.
DETAIL_FIELDS = NODE_FIELDS + ",recommendations,related_anime,statistics"


class MalApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# details_many gives up after this many failed ids in a row: MAL is down or
# refusing the client, and every further id would cost another half minute
# of retries while the only worker thread waits (audit 2026-10-23, ING-7)
MAX_FAILURES_IN_A_ROW = 10
RETRY_AFTER_MAX = 120.0


def _retry_after(value: str | None) -> float | None:
    """Retry-After in seconds - a number or an HTTP date - capped at
    RETRY_AFTER_MAX; None when absent or unreadable."""
    if not value:
        return None
    try:
        secs = float(value)
    except ValueError:
        from email.utils import parsedate_to_datetime
        try:
            secs = (parsedate_to_datetime(value) - dt.datetime.now(dt.UTC)).total_seconds()
        except (TypeError, ValueError):
            return None
    return min(max(secs, 0.0), RETRY_AFTER_MAX)


class MalClient:
    """Client-ID access by default; pass `token` to act as a signed-in user
    (needed for @me, private lists and writing list status)."""

    def __init__(self, client_id: str | None = None, rps: float | None = None,
                 token: str | None = None):
        self.client_id = client_id or settings().mal_client_id
        if not self.client_id and not token:
            raise RuntimeError("MAL_CLIENT_ID is not set")
        self._min_interval = 1.0 / (rps or settings().mal_rps)
        self._last = 0.0
        self._lock = threading.Lock()
        headers = ({"Authorization": f"Bearer {token}"} if token
                   else {"X-MAL-CLIENT-ID": self.client_id})
        self._http = httpx.Client(
            base_url=BASE,
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self) -> None:
        """Reserve the next request slot, `_min_interval` after the previous one.

        Slots are handed out under a lock, so several threads can share one
        client: request *starts* stay exactly at the configured rate while
        slow responses overlap. With a single thread a 1.5 s response to a
        1,000-entry list would otherwise drag the effective rate well below it.
        """
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._last + self._min_interval)
            self._last = slot
        if slot > now:
            time.sleep(slot - now)

    def get(self, path: str, params: dict | None = None, tries: int = 5) -> dict:
        """The JSON of a 200; {} for a 404 (genuinely gone). Any other 4xx
        raises MalApiError: a refused token or client id (401/403), a private
        list, a bad request are answers, not absences - read as {} they made
        a list "empty", an anime "gone for good" (stamped, never fetched
        again) and a catalogue sync "0 updated, ok" (audit 2026-10-23,
        ING-2). 429, 5xx and MAL's 307 load shedding are retried, then
        RuntimeError."""
        for attempt in range(tries):
            self._throttle()
            try:
                r = self._http.get(path, params=params)
                if r.status_code == 429:
                    back = _retry_after(r.headers.get("Retry-After")) or 5 * (attempt + 1)
                    log.warning("MAL 429, backing off %.0fs", back)
                    time.sleep(back)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(2 * (attempt + 1))
                    continue
                if r.status_code == 404:
                    return {}                     # genuinely gone
                # MAL sheds load by 307-redirecting to error.json instead of
                # returning 429. It looks permanent but is not: the same id
                # serves fine seconds later, so this must be retried, never
                # recorded as "entry unavailable".
                if 300 <= r.status_code < 400:
                    back = 2 * (attempt + 1)
                    log.warning("MAL %s -> %s (throttled), retry in %ds",
                                path, r.status_code, back)
                    time.sleep(back)
                    continue
                if 400 <= r.status_code < 500:
                    # will not fix itself on a retry
                    log.warning("MAL %s -> %s", path, r.status_code)
                    raise MalApiError(r.status_code, f"MAL {path} -> {r.status_code}")
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError:
                time.sleep(2 * (attempt + 1))
            except (httpx.TimeoutException, httpx.TransportError) as e:
                log.warning("MAL transport error (%s), retry %d", type(e).__name__, attempt + 1)
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"MAL request failed after {tries} attempts: {path} {params}")

    # ------------------------------------------------------------------ list

    def user_animelist(self, username: str) -> list[dict]:
        """Full list in as few calls as possible: limit maxes out at 1000 and
        `fields` accepts anime-node fields, so one page usually suffices.
        Every page must arrive: a page that fails raises (MalApiError, also
        for a user MAL does not know - 404 -, or RuntimeError), since a part
        of a list must never stand in for the whole (store.upsert_list
        deletes what it does not contain)."""
        out: list[dict] = []
        offset = 0
        while True:
            d = self.get(
                f"/users/{username}/animelist",
                {"limit": 1000, "offset": offset, "nsfw": "true",
                 "fields": f"list_status,{NODE_FIELDS}"},
            )
            if "data" not in d:
                raise MalApiError(404, f"MAL has no user {username!r} (renamed or deleted?)")
            out.extend(d["data"])
            if "next" not in d.get("paging", {}):
                break
            offset += 1000
        return out

    # --------------------------------------------------------------- catalog

    def ranking(self, ranking_type: str, limit: int = 500, offset: int = 0) -> list[dict]:
        d = self.get("/anime/ranking", {
            "ranking_type": ranking_type, "limit": limit, "offset": offset,
            "nsfw": "true", "fields": NODE_FIELDS,
        })
        return d.get("data", [])

    def seasonal(self, year: int, season: str, limit: int = 500) -> list[dict]:
        d = self.get(f"/anime/season/{year}/{season}", {
            "limit": limit, "sort": "anime_num_list_users", "nsfw": "true",
            "fields": NODE_FIELDS,
        })
        return d.get("data", [])

    def details(self, mal_id: int) -> dict:
        return self.get(f"/anime/{mal_id}", {"fields": DETAIL_FIELDS})

    def details_many(self, mal_ids: list[int]) -> Iterator[tuple[int, dict | None]]:
        """One request per id; there is no batch endpoint.

        Yields the payload on success, {} when the anime genuinely does not
        exist (404), and None when the request failed for a reason that may
        resolve later. The caller must not mark a None as fetched. After
        MAX_FAILURES_IN_A_ROW Nones in a row it raises RuntimeError instead
        of trying the rest.
        """
        failed = 0
        for i, aid in enumerate(mal_ids):
            try:
                d = self.details(aid)
            except (RuntimeError, MalApiError, httpx.HTTPError) as e:
                log.error("giving up on %s for now: %s", aid, e)
                failed += 1
                if failed >= MAX_FAILURES_IN_A_ROW:
                    raise RuntimeError(f"MAL failed {failed} anime in a row (last: {e});"
                                       " stopping until a later run") from e
                yield aid, None
            else:
                failed = 0
                yield aid, d
            if i and i % 250 == 0:
                log.info("mal details %d/%d", i, len(mal_ids))

    # ------------------------------------------------ user-scoped (needs token)

    def _send(self, method: str, path: str, data: dict | None = None) -> dict:
        """Single write-style call that reports failure instead of hiding it.

        get() collapses errors into an empty dict, which is right for reads
        but would make a failed list update look like a success.
        """
        for attempt in range(4):
            self._throttle()
            try:
                r = self._http.request(method, path, data=data)
            except (httpx.TimeoutException, httpx.TransportError):
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code in (429, 500, 502, 503, 504) or 300 <= r.status_code < 400:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code >= 400:
                raise MalApiError(r.status_code, r.text[:300])
            return r.json() if r.content else {}
        raise MalApiError(0, f"MAL {method} {path} failed after retries")

    def me(self) -> dict:
        return self._send("GET", "/users/@me")

    def my_list_status(self, mal_id: int) -> dict | None:
        """The signed-in user's entry for this anime, or None if not listed."""
        d = self._send("GET", f"/anime/{mal_id}?fields=my_list_status")
        return d.get("my_list_status") or None

    def set_list_status(self, mal_id: int, status: str) -> dict:
        return self._send("PATCH", f"/anime/{mal_id}/my_list_status", {"status": status})

    def update_list_status(self, mal_id: int, fields: dict) -> dict:
        """PATCH any of status / score / num_watched_episodes."""
        return self._send("PATCH", f"/anime/{mal_id}/my_list_status",
                          {k: str(v) for k, v in fields.items()})

    def delete_list_entry(self, mal_id: int) -> None:
        try:
            self._send("DELETE", f"/anime/{mal_id}/my_list_status")
        except MalApiError as e:
            if e.status != 404:          # already gone is the outcome we wanted
                raise

    def search(self, q: str, limit: int = 20) -> list[dict]:
        d = self.get("/anime", {"q": q, "limit": min(limit, 100), "fields": NODE_FIELDS})
        return d.get("data", [])


def normalise_node(node: dict[str, Any]) -> dict[str, Any]:
    """MAL node JSON -> flat row matching the `anime` table."""
    alt = node.get("alternative_titles") or {}
    season = node.get("start_season") or {}
    pic = node.get("main_picture") or {}
    return {
        "mal_id": node["id"],
        "title": node.get("title") or str(node["id"]),
        "title_en": (alt.get("en") or None),
        "title_ja": (alt.get("ja") or None),
        "synopsis": node.get("synopsis"),
        "media_type": node.get("media_type"),
        "status": node.get("status"),
        "source": node.get("source"),
        "rating": node.get("rating"),
        "nsfw": node.get("nsfw"),
        "num_episodes": node.get("num_episodes"),
        "avg_episode_seconds": node.get("average_episode_duration"),
        "start_date": _date(node.get("start_date")),
        "end_date": _date(node.get("end_date")),
        "season_year": season.get("year"),
        "season": season.get("season"),
        "picture_medium": pic.get("medium"),
        "picture_large": pic.get("large"),
        "mal_mean": node.get("mean"),
        "mal_rank": node.get("rank"),
        "mal_popularity": node.get("popularity"),
        "mal_num_list_users": node.get("num_list_users"),
        "mal_num_scoring_users": node.get("num_scoring_users"),
        "mal_genres": [g["name"] for g in node.get("genres") or []],
        "mal_studios": [s["name"] for s in node.get("studios") or []],
    }


def _date(v: str | None):
    """MAL returns partial dates like '2013' or '2013-04'; widen to a real date.
    A date that does not exist (a list entry finished on 2020-02-30, year 0)
    gives None: Postgres rejects it, and one such entry failed the whole sync."""
    if not v:
        return None
    parts = v.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return dt.date(y, m, d).isoformat()
    except (ValueError, IndexError):
        return None
