"""Bulk upsert helpers. Everything goes through COPY-style executemany with
ON CONFLICT so an ingest can be re-run at any time without duplicating rows."""
from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence

from psycopg.types.json import Jsonb

from ..clients.mal import _date
from ..db import conn

log = logging.getLogger(__name__)

ANIME_COLS = [
    "mal_id", "title", "title_en", "title_ja", "synopsis", "media_type", "status", "source",
    "rating", "nsfw", "num_episodes", "avg_episode_seconds", "start_date", "end_date",
    "season_year", "season", "picture_medium", "picture_large", "mal_mean", "mal_rank",
    "mal_popularity", "mal_num_list_users", "mal_num_scoring_users", "mal_genres", "mal_studios",
]


ARRAY_COLS = {"mal_genres", "mal_studios"}


def upsert_anime(rows: Sequence[dict]) -> int:
    """Insert or refresh MAL-sourced catalog columns.

    Two things this must never do:
      * wipe a fully populated row. The recommendation graph yields stub nodes
        carrying only an id and a title, so every column is COALESCEd against
        the existing value rather than overwritten with NULL.
      * touch AniList columns or tag_vec, so a catalog refresh cannot undo an
        enrichment pass.
    """
    if not rows:
        return 0
    # de-duplicate within the batch, preferring the richest version of each row
    best: dict[int, dict] = {}
    for r in rows:
        prev = best.get(r["mal_id"])
        if prev is None or sum(v is not None for v in r.values()) > sum(
                v is not None for v in prev.values()):
            best[r["mal_id"]] = r
    rows = list(best.values())

    cols = ANIME_COLS
    placeholders = ",".join(["%s"] * (len(cols) + 1))          # +1 for mal_fetched_at
    # For arrays an empty value means "this row carried no data", not "the
    # anime has no genres", so NULLIF keeps a stub from clearing a real list.
    updates = ",".join(
        (f"{c}=COALESCE(NULLIF(EXCLUDED.{c}, '{{}}'), anime.{c})" if c in ARRAY_COLS
         else f"{c}=COALESCE(EXCLUDED.{c}, anime.{c})")
        for c in cols if c != "mal_id"
    )
    sql = (
        f"INSERT INTO anime ({','.join(cols)}, mal_fetched_at) VALUES ({placeholders}) "
        f"ON CONFLICT (mal_id) DO UPDATE SET {updates}, "
        f"mal_fetched_at=COALESCE(EXCLUDED.mal_fetched_at, anime.mal_fetched_at)"
    )

    def cell(r: dict, c: str):
        v = r.get(c)
        # NOT NULL array columns need a value on INSERT; the COALESCE above
        # keeps an existing non-empty array from being clobbered on UPDATE.
        if v is None and c in ARRAY_COLS:
            return []
        return v

    data = [tuple(cell(r, c) for c in cols) + (r.get("mal_fetched_at"),) for r in rows]
    with conn() as c, c.cursor() as cur:
        cur.executemany(sql, data)
        c.commit()
    return len(data)


def upsert_anilist(rows: Sequence[dict]) -> int:
    """AniList columns only; requires the anime row to already exist."""
    if not rows:
        return 0
    sql = """
        UPDATE anime SET anilist_id=%s, al_average_score=%s, al_mean_score=%s,
               al_popularity=%s, al_favourites=%s, al_genres=%s,
               al_status_dist=%s, al_score_dist=%s, al_drop_rate=%s,
               raw_al=%s, al_fetched_at=now()
         WHERE mal_id=%s
    """
    data = [(r["anilist_id"], r["al_average_score"], r["al_mean_score"], r["al_popularity"],
             r["al_favourites"], r["al_genres"],
             Jsonb(r["al_status_dist"]) if r.get("al_status_dist") is not None else None,
             Jsonb(r["al_score_dist"]) if r.get("al_score_dist") is not None else None,
             r.get("al_drop_rate"),
             Jsonb(r["raw"]) if r.get("raw") is not None else None,
             r["mal_id"]) for r in rows]
    with conn() as c, c.cursor() as cur:
        cur.executemany(sql, data)
        c.commit()
    return len(data)


def replace_tags(pairs: Iterable[tuple[int, str, int, str]]) -> int:
    pairs = list(pairs)
    if not pairs:
        return 0
    ids = sorted({p[0] for p in pairs})
    with conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM anime_tag WHERE mal_id = ANY(%s)", (ids,))
        cur.executemany(
            "INSERT INTO anime_tag (mal_id, tag, rank, category) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT (mal_id, tag) DO UPDATE SET rank=EXCLUDED.rank",
            pairs,
        )
        c.commit()
    return len(pairs)


def upsert_rec_edges(edges: Iterable[tuple[int, int, str, int, float]]) -> int:
    """(src, dst, provider, votes, weight), stored in both directions."""
    edges = list(edges)
    if not edges:
        return 0
    both = []
    for src, dst, provider, votes, weight in edges:
        both.append((src, dst, provider, votes, weight))
        both.append((dst, src, provider, votes, weight))
    with conn() as c, c.cursor() as cur:
        cur.executemany(
            "INSERT INTO rec_edge (src,dst,provider,votes,weight) VALUES (%s,%s,%s,%s,%s) "
            "ON CONFLICT (src,dst,provider) DO UPDATE SET "
            "votes=GREATEST(rec_edge.votes, EXCLUDED.votes), "
            "weight=GREATEST(rec_edge.weight, EXCLUDED.weight)",
            both,
        )
        c.commit()
    return len(both)


def upsert_relations(rels: Iterable[tuple[int, int, str, str]]) -> int:
    """(src, dst, relation_type, provider). Stored with the inverse edge too so
    the prerequisite CTE can walk in either direction."""
    rels = list(rels)
    if not rels:
        return 0
    # MAL's relation vocabulary is directional and comes in pairs. Anything
    # without a named inverse (alternative_version, spin_off, other, ...) is
    # genuinely symmetric and keeps its own name.
    inverse = {
        "sequel": "prequel", "prequel": "sequel",
        "parent_story": "side_story", "side_story": "parent_story",
        "summary": "full_story", "full_story": "summary",
    }
    both = []
    for src, dst, rt, provider in rels:
        both.append((src, dst, rt, provider))
        both.append((dst, src, inverse.get(rt, rt), provider))
    with conn() as c, c.cursor() as cur:
        cur.executemany(
            "INSERT INTO relation (src,dst,relation_type,provider) VALUES (%s,%s,%s,%s) "
            "ON CONFLICT DO NOTHING",
            both,
        )
        c.commit()
    return len(both)


def upsert_list(user_id: int, entries: Sequence[dict]) -> int:
    if not entries:
        return 0
    sql = """
        INSERT INTO list_entry (user_id, mal_id, status, score, episodes_watched,
                                is_rewatching, started_at, finished_at, updated_at)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (user_id, mal_id) DO UPDATE SET
            status=EXCLUDED.status, score=EXCLUDED.score,
            episodes_watched=EXCLUDED.episodes_watched,
            is_rewatching=EXCLUDED.is_rewatching, started_at=EXCLUDED.started_at,
            finished_at=EXCLUDED.finished_at, updated_at=EXCLUDED.updated_at
    """
    data = []
    for e in entries:
        ls = e["list_status"]
        data.append((user_id, e["node"]["id"], ls["status"], ls.get("score") or 0,
                     ls.get("num_episodes_watched") or 0, ls.get("is_rewatching") or False,
                     _date(ls.get("start_date")), _date(ls.get("finish_date")),
                     ls.get("updated_at")))
    with conn() as c, c.cursor() as cur:
        # A removed entry should disappear rather than linger as a stale filter.
        cur.execute("DELETE FROM list_entry WHERE user_id=%s AND mal_id <> ALL(%s)",
                    (user_id, [d[1] for d in data]))
        cur.executemany(sql, data)
        c.commit()
    return len(data)


def get_or_create_user(username: str) -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO app_user (mal_username) VALUES (%s) "
            "ON CONFLICT (mal_username) DO UPDATE SET mal_username=EXCLUDED.mal_username "
            "RETURNING id",
            (username,),
        )
        uid = cur.fetchone()["id"]
        c.commit()
    return uid


def mark_graph_fetched(mal_ids: Sequence[int]) -> int:
    """Stamp anime whose detail payload has been pulled, so an interrupted
    sync_graph resumes instead of refetching."""
    if not mal_ids:
        return 0
    with conn() as c, c.cursor() as cur:
        cur.execute("UPDATE anime SET graph_fetched_at=now() WHERE mal_id = ANY(%s)",
                    (list(mal_ids),))
        c.commit()
    return len(mal_ids)


def mark_anilist_attempted(mal_ids: Sequence[int]) -> int:
    """AniList genuinely has no entry for some MAL ids. Stamp them anyway so
    `only_missing` runs stop retrying the same dead ids on every refresh."""
    if not mal_ids:
        return 0
    with conn() as c, c.cursor() as cur:
        cur.execute("UPDATE anime SET al_fetched_at=now() WHERE mal_id = ANY(%s)",
                    (list(mal_ids),))
        c.commit()
    return len(mal_ids)


def mark_synced(user_id: int) -> None:
    with conn() as c, c.cursor() as cur:
        cur.execute("UPDATE app_user SET last_sync_at=now() WHERE id=%s", (user_id,))
        c.commit()


def store_tag_vectors(dim: int = 256) -> int:
    """Project each anime's weighted tags into a fixed-width vector via hashing,
    then L2-normalise so `<=>` is a true cosine distance. Hashing avoids having
    to pin a tag vocabulary in the schema as AniList adds tags over time."""
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT mal_id, tag, rank FROM anime_tag ORDER BY mal_id")
        rows = cur.fetchall()
        by_anime: dict[int, list[tuple[str, int]]] = {}
        for r in rows:
            by_anime.setdefault(r["mal_id"], []).append((r["tag"], r["rank"]))

        data = []
        for mal_id, tags in by_anime.items():
            v = [0.0] * dim
            for tag, rank in tags:
                h = hash_tag(tag)
                sign = 1.0 if (h >> 31) & 1 else -1.0
                v[h % dim] += sign * (rank / 100.0)
            norm = math.sqrt(sum(x * x for x in v))
            if norm == 0:
                continue
            data.append(("[" + ",".join(f"{x / norm:.6f}" for x in v) + "]", mal_id))
        cur.executemany("UPDATE anime SET tag_vec=%s::vector WHERE mal_id=%s", data)
        c.commit()
    return len(data)


def hash_tag(tag: str) -> int:
    """Stable across processes, unlike builtin hash() with PYTHONHASHSEED."""
    import zlib
    return zlib.crc32(tag.encode("utf-8"))


def replace_staff(rows: Iterable[tuple[int, int, str, str]]) -> int:
    """(mal_id, staff_id, name, role). Replaces the staff of every anime present."""
    rows = list(rows)
    if not rows:
        return 0
    ids = sorted({r[0] for r in rows})
    with conn() as c, c.cursor() as cur:
        cur.execute("DELETE FROM anime_staff WHERE mal_id = ANY(%s)", (ids,))
        cur.executemany(
            "INSERT INTO anime_staff (mal_id, staff_id, name, role) VALUES (%s,%s,%s,%s)"
            " ON CONFLICT DO NOTHING", rows)
        c.commit()
    return len(rows)


def store_mal_stats(rows: Iterable[tuple]) -> int:
    """(mal_id, status_distribution, drop_rate, raw_detail_payload)."""
    rows = list(rows)
    if not rows:
        return 0
    with conn() as c, c.cursor() as cur:
        cur.executemany(
            "UPDATE anime SET mal_status_dist=%s, mal_drop_rate=%s, raw_mal=%s WHERE mal_id=%s",
            [(Jsonb(dist) if dist else None, drop, Jsonb(raw) if raw else None, mid)
             for mid, dist, drop, raw in rows])
        c.commit()
    return len(rows)


PICTURE_HOST = "cdn.myanimelist.net"


def set_picture(user_id: int, url: str | None) -> None:
    """Store the MAL profile picture, but only an https URL on MAL's image CDN
    (the only image host besides AniList the app's CSP allows); anything else
    clears it and the app shows the initial instead."""
    from urllib.parse import urlparse
    u = urlparse(url or "")
    ok = u.scheme == "https" and u.netloc == PICTURE_HOST and len(url or "") <= 500
    with conn() as c:
        c.execute("UPDATE app_user SET picture_url=%s WHERE id=%s", (url if ok else None, user_id))
