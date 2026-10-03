"""AniList GraphQL client.

AniList is used for three things MAL cannot provide:
  * weighted tags - ~22 per anime with a 0-100 relevance rank, versus MAL's
    handful of flat genres. Measurably the single best content signal available.
  * a second, independent recommendation graph.
  * a second consensus score (averageScore / favourites).

`idMal` on every Media node means no fuzzy title matching is ever needed.
GraphQL batches 50 anime per request, which more than offsets the 30/min limit.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

import httpx

from ..config import settings

log = logging.getLogger(__name__)

ENDPOINT = "https://graphql.anilist.co"
# AniList rejects requests without a User-Agent with a bare 403.
USER_AGENT = "malrec/0.1 (personal recommendation project)"

PAGE_QUERY = """
query ($ids: [Int], $page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo { hasNextPage }
    media(idMal_in: $ids, type: ANIME) {
      id idMal
      title { romaji english }
      averageScore meanScore popularity favourites
      episodes duration format source seasonYear season countryOfOrigin isAdult
      genres
      tags { name rank category isAdult }
      studios(isMain: true) { nodes { name } }
      staff(perPage: 8, sort: [RELEVANCE]) { edges { role node { id name { full } } } }
      stats {
        scoreDistribution { score amount }
        statusDistribution { status amount }
      }
      recommendations(sort: RATING_DESC, perPage: 25) {
        nodes { rating mediaRecommendation { idMal title { romaji } } }
      }
      relations { edges { relationType node { idMal type } } }
    }
  }
}
"""

# AniList relation names -> the vocabulary used in the `relation` table.
RELATION_MAP = {
    "SEQUEL": "sequel",
    "PREQUEL": "prequel",
    "SIDE_STORY": "side_story",
    "PARENT": "parent_story",
    "ALTERNATIVE": "alternative_version",
    "SPIN_OFF": "spin_off",
    "SUMMARY": "summary",
    "ADAPTATION": "adaptation",
    "CHARACTER": "character",
    "OTHER": "other",
}


class AniListClient:
    def __init__(self, rpm: int | None = None):
        self._min_interval = 60.0 / (rpm or settings().anilist_rpm)
        self._last = 0.0
        self._http = httpx.Client(
            timeout=httpx.Timeout(45.0, connect=10.0),
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": USER_AGENT},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _adopt_limit(self, headers: httpx.Headers) -> None:
        """Pace at exactly the limit the server reports.

        AniList documents 90 req/min but currently runs degraded at 30, and
        says so in X-RateLimit-Limit on every response. Deriving the interval
        from that header means we never run slower than allowed, and pick up a
        restored limit without a config change. Requests are spaced evenly
        rather than sent in bursts, which also keeps the (unspecified) burst
        limiter happy.
        """
        try:
            limit = int(headers.get("X-RateLimit-Limit", 0))
        except ValueError:
            return
        if limit > 0:
            interval = 60.0 / limit
            if abs(interval - self._min_interval) > 1e-6:
                log.info("AniList limit is %d/min; pacing at one request per %.2fs",
                         limit, interval)
                self._min_interval = interval

    def gql(self, query: str, variables: dict, tries: int = 5) -> dict:
        for attempt in range(tries):
            self._throttle()
            try:
                r = self._http.post(ENDPOINT, json={"query": query, "variables": variables})
                if r.status_code == 429:
                    # documented: a one-minute timeout, with Retry-After set
                    back = float(r.headers.get("Retry-After", 0)) or 60
                    log.warning("AniList 429, sleeping %.0fs", back)
                    time.sleep(back + 1)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(3 * (attempt + 1))
                    continue
                r.raise_for_status()
                self._adopt_limit(r.headers)
                # Even pacing should never exhaust the window; if something else
                # shares the quota and it does, wait for the documented reset.
                if int(r.headers.get("X-RateLimit-Remaining", 99)) <= 0:
                    reset = float(r.headers.get("X-RateLimit-Reset", 0) or 0)
                    time.sleep(max(1.0, reset - time.time()) if reset else 60.0)
                body = r.json()
                if body.get("errors"):
                    log.warning("AniList GraphQL errors: %s", body["errors"][:1])
                return body
            except (httpx.TimeoutException, httpx.TransportError) as e:
                log.warning("AniList transport error (%s), retry %d", type(e).__name__, attempt + 1)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("AniList request failed after retries")

    def media_by_mal_ids(self, mal_ids: Iterable[int]) -> dict[int, dict]:
        """Batch-fetch by MAL id. Returns {mal_id: media}; ids AniList does not
        know about are simply absent."""
        ids = list(dict.fromkeys(int(i) for i in mal_ids))
        size = settings().anilist_batch
        out: dict[int, dict] = {}
        for i in range(0, len(ids), size):
            chunk = ids[i:i + size]
            body = self.gql(PAGE_QUERY, {"ids": chunk, "page": 1, "perPage": size})
            page = ((body.get("data") or {}).get("Page") or {})
            for m in page.get("media") or []:
                if m.get("idMal"):
                    out[m["idMal"]] = m
            if i and i % (size * 10) == 0:
                log.info("anilist %d/%d", i, len(ids))
        return out


def normalise_media(m: dict[str, Any]) -> dict[str, Any]:
    """AniList media -> the AniList columns of the `anime` table."""
    stats = m.get("stats") or {}
    status = {d["status"]: d["amount"] for d in stats.get("statusDistribution") or []
              if d.get("status")}
    started = sum(status.get(k, 0) for k in ("CURRENT", "COMPLETED", "DROPPED", "PAUSED"))
    return {
        "mal_id": m["idMal"],
        "anilist_id": m.get("id"),
        "al_average_score": m.get("averageScore"),
        "al_mean_score": m.get("meanScore"),
        "al_popularity": m.get("popularity"),
        "al_favourites": m.get("favourites"),
        "al_genres": m.get("genres") or [],
        "al_status_dist": status or None,
        "al_score_dist": {str(d["score"]): d["amount"]
                          for d in stats.get("scoreDistribution") or []} or None,
        # share of people who started it and gave up; needs a real audience
        # before it means anything
        "al_drop_rate": (status.get("DROPPED", 0) / started) if started >= 50 else None,
        "raw": m,
    }


def extract_staff(m: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    """(mal_id, staff_id, name, role) for the key creative staff."""
    out = []
    for e in (m.get("staff") or {}).get("edges") or []:
        node = e.get("node") or {}
        if node.get("id") and e.get("role"):
            # "Director (eps 1-12)" and "Director" are the same job for our purposes
            role = e["role"].split("(")[0].strip()
            out.append((m["idMal"], node["id"], (node.get("name") or {}).get("full") or "", role))
    return out


def extract_tags(m: dict[str, Any], min_rank: int = 40) -> list[tuple[int, str, int, str]]:
    """(mal_id, tag, rank, category), dropping weak/spoiler-ish low-rank tags."""
    out = []
    for t in m.get("tags") or []:
        if (t.get("rank") or 0) >= min_rank and not t.get("isAdult"):
            out.append((m["idMal"], t["name"], int(t["rank"]), t.get("category")))
    return out
