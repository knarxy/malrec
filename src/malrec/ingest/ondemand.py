"""Lazy, demand-driven fetching.

The rule this module enforces: **never fetch an anime nobody asked for.**

Per-item endpoints (MAL `/anime/{id}`, AniList media) are only called for anime
that are about to be used - the user's own list, or the shortlist the ranker is
about to score. Walking the whole catalogue "just in case" would mean thousands
of requests for rows nobody ever sees, which is neither fair use nor useful.

Bulk endpoints are treated differently and deliberately so: `/anime/ranking`
returns 500 fully-populated rows per request and exists precisely to hand out
lists of anime, so ~30 of those calls give broad candidate metadata at a
fraction of the cost of the equivalent per-item fetches. That is a one-off, not
a crawl.

Everything here is idempotent and incremental: an id already present and fresh
costs nothing.
"""
from __future__ import annotations

import datetime as dt
import logging

from ..clients.anilist import AniListClient
from ..clients.mal import MalClient
from ..config import settings
from ..db import query, refresh_franchises
from . import store

log = logging.getLogger(__name__)

# How long an enrichment stays fresh before it is worth refetching. Scores and
# popularity drift slowly; tags and relations barely at all.
STALE_AFTER = dt.timedelta(days=30)


def _missing(ids: list[int], column: str, max_age: dt.timedelta | None) -> list[int]:
    if not ids:
        return []
    sql = f"SELECT mal_id FROM anime WHERE mal_id = ANY(%s) AND {column} IS NOT NULL"
    params: list = [list(set(ids))]
    if max_age is not None:
        sql += " AND %s - " + column + " < %s"
        params += [dt.datetime.now(dt.UTC), max_age]
    have = {r["mal_id"] for r in query(sql, tuple(params))}
    return [i for i in dict.fromkeys(ids) if i not in have]


def ensure_anilist(mal_ids: list[int], max_age: dt.timedelta | None = None,
                   force: bool = False) -> int:
    """Fetch AniList data for exactly the ids that lack it.

    Batched 50 per GraphQL request, so a 500-anime shortlist is 10 requests.
    Results are written batch by batch, and ids AniList does not recognise are
    stamped too so they are not retried forever.
    """
    todo = list(dict.fromkeys(mal_ids)) if force else _missing(mal_ids, "al_fetched_at", max_age)
    if not todo:
        return 0
    log.info("enriching %d anime from AniList (%d requested)", len(todo), len(mal_ids))

    from .jobs import _store_anilist_batch  # local import avoids a cycle

    batch = settings().anilist_batch
    fetched = 0
    with AniListClient() as al:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            got = al.media_by_mal_ids(chunk)
            if got:
                _store_anilist_batch(got)
                fetched += len(got)
            store.mark_anilist_attempted([c for c in chunk if c not in got])
    store.store_tag_vectors()
    # New relation edges only affect franchise collapse once components are
    # recomputed, and enriching a shortlist is exactly when new ones arrive.
    if fetched:
        refresh_franchises()
    return fetched


def ensure_graph(mal_ids: list[int], max_age: dt.timedelta | None = None) -> int:
    """Fetch MAL detail payloads (recommendations + relations) for exactly the
    ids that lack them. One request each, so callers should pass small sets -
    typically just the entries newly added to a user's list."""
    todo = _missing(mal_ids, "graph_fetched_at", max_age)
    if not todo:
        return 0
    log.info("fetching MAL details for %d anime (%d requested)", len(todo), len(mal_ids))

    from .jobs import GRAPH_FLUSH_EVERY, _collect

    rec_edges: list = []
    rels: list = []
    nodes: list = []
    done: list[int] = []
    fetched = 0

    def flush() -> None:
        nonlocal rec_edges, rels, nodes, done
        if not done:
            return
        store.upsert_anime([x for x in nodes if x.get("title")])
        store.upsert_rec_edges(rec_edges)
        store.upsert_relations(rels)
        store.mark_graph_fetched(done)
        rec_edges, rels, nodes, done = [], [], [], []

    try:
        with MalClient() as mal:
            for aid, d in mal.details_many(todo):
                if d is None:
                    continue                      # retry on a later call
                if d:
                    fetched += 1
                    _collect(aid, d, rec_edges, rels, nodes)
                done.append(aid)
                if len(done) >= GRAPH_FLUSH_EVERY:
                    flush()
    finally:
        flush()
    if fetched:
        refresh_franchises()
    return fetched


def ensure_for_user(user_id: int) -> dict:
    """Everything a user's own list needs, and nothing else.

    Called after a list sync. Only entries that are new since last time cost a
    request, so the steady-state cost of a re-sync is close to zero.
    """
    listed = [r["mal_id"] for r in query(
        "SELECT mal_id FROM list_entry WHERE user_id = %s", (user_id,))]
    rated = [r["mal_id"] for r in query(
        "SELECT mal_id FROM list_entry WHERE user_id = %s AND score > 0", (user_id,))]
    graph = ensure_graph(listed)
    tags = ensure_anilist(rated)
    return {"listed": len(listed), "graph_fetched": graph, "anilist_fetched": tags}


def ensure_for_candidates(mal_ids: list[int]) -> dict:
    """Enrich a shortlist just before the model scores it.

    Only AniList is pulled here: its tags are the feature that matters and it
    batches 50 per request. MAL recommendation edges for these anime are
    already present, because edges are stored in both directions when the
    user's own list is fetched.
    """
    return {"anilist_fetched": ensure_anilist(mal_ids)}
