"""The four ingest jobs, in the order a cold start runs them.

    sync_user_list  -> who we are recommending for
    sync_catalog    -> the candidate universe (cheap: ~18 MAL calls)
    sync_graph      -> recommendation + franchise edges (the expensive one)
    sync_anilist    -> weighted tags and second consensus score
"""
from __future__ import annotations

import datetime as dt
import logging
import math

from ..clients.anilist import (
    RELATION_MAP,
    AniListClient,
    extract_staff,
    extract_tags,
    normalise_media,
)
from ..clients.mal import MalClient, normalise_node
from ..config import settings
from ..db import log_ingest, query, refresh_franchises, scalar
from . import store

log = logging.getLogger(__name__)


def _now():
    return dt.datetime.now(dt.UTC)


def sync_user_list(username: str | None = None, enrich: bool = True,
                   token: str | None = None) -> dict:
    """With `token` the list is read as the signed-in user (@me), which also
    works for lists that are private."""
    username = username or settings().malrec_user
    with MalClient(token=token) as mal:
        entries = mal.user_animelist("@me" if token else username)
        me = None
        if token:
            try:
                me = mal.me()        # one request; keeps the profile picture current
            except Exception:  # noqa: BLE001 - the picture is cosmetic
                log.warning("could not read the profile of %s", username)
    uid = store.get_or_create_user(username)
    if me is not None:
        store.set_picture(uid, me.get("picture"))
    nodes = [normalise_node(e["node"]) | {"mal_fetched_at": _now()} for e in entries]
    store.upsert_anime(nodes)
    n = store.upsert_list(uid, entries)
    store.mark_synced(uid)
    scored = sum(1 for e in entries if (e["list_status"].get("score") or 0) > 0)
    detail = {"user": username, "entries": n, "scored": scored}
    if enrich:
        # only entries new since the last sync cost a request
        from .ondemand import ensure_for_user
        detail |= ensure_for_user(uid)
    log_ingest("sync_user_list", "ok", detail)
    log.info("synced %s: %d entries, %d scored", username, n, scored)
    return detail


def sync_catalog() -> dict:
    """Ranking endpoints return up to 500 rows *with full fields*, so the whole
    candidate universe costs about 18 requests."""
    cfg = settings()
    seen: dict[int, dict] = {}
    with MalClient() as mal:
        for rt in cfg.ranking_types:
            for page in range(cfg.ranking_pages):
                data = mal.ranking(rt, limit=500, offset=page * 500)
                for e in data:
                    seen[e["node"]["id"]] = normalise_node(e["node"]) | {"mal_fetched_at": _now()}
                if len(data) < 500:
                    break
            log.info("catalog after %s: %d", rt, len(seen))

        # current and upcoming seasons, which rankings under-cover
        y = _now().year
        for year in (y - 1, y, y + 1):
            for season in ("winter", "spring", "summer", "fall"):
                try:
                    for e in mal.seasonal(year, season):
                        seen.setdefault(e["node"]["id"],
                                        normalise_node(e["node"]) | {"mal_fetched_at": _now()})
                except RuntimeError:
                    continue

    store.upsert_anime(list(seen.values()))
    detail = {"anime": len(seen)}
    log_ingest("sync_catalog", "ok", detail)
    log.info("catalog: %d anime", len(seen))
    return detail


def _graph_targets(limit: int | None, only_missing: bool, scope: str) -> list[int]:
    """Which anime are worth a MAL detail call (one request each, ~1.5/s).

    scope='listed' (the default) fetches only what the user has on their list.
    That is enough for the recommendation graph: rec_edge rows are written in
    both directions, so pulling the 250 listed anime already yields every edge
    between a listed anime and the things the community recommends alongside
    it - which is exactly what the affinity features read when scoring a
    candidate. Fetching the whole catalogue would only add candidate-to-
    candidate edges that nothing currently uses, at 30x the request count.

    scope='all' exists for when a future feature needs those edges (e.g. graph
    walks two hops out, or candidate-to-candidate diversity).
    """
    if scope == "listed":
        sql = """
            SELECT DISTINCT le.mal_id
              FROM list_entry le JOIN anime a ON a.mal_id = le.mal_id
        """
        sql += " WHERE a.graph_fetched_at IS NULL" if only_missing else ""
        rows = query(sql)
    elif scope == "all":
        sql = """
            SELECT a.mal_id, bool_or(le.mal_id IS NOT NULL) AS listed
              FROM anime a
              LEFT JOIN list_entry le ON le.mal_id = a.mal_id
             WHERE (le.mal_id IS NOT NULL OR coalesce(a.mal_num_scoring_users, 0) >= %s)
        """
        if only_missing:
            sql += " AND a.graph_fetched_at IS NULL"
        sql += """
             GROUP BY a.mal_id, a.mal_popularity
             ORDER BY listed DESC, a.mal_popularity ASC NULLS LAST
        """
        rows = query(sql, (settings().min_scoring_users,))
    else:
        raise ValueError(f"scope must be 'listed' or 'all', got {scope!r}")
    ids = [r["mal_id"] for r in rows]
    return ids[:limit] if limit else ids


# Flush to Postgres every N anime. Small enough that a crash or a rate-limit
# stall costs seconds of refetching, not the whole run.
GRAPH_FLUSH_EVERY = 50


def _collect(aid: int, d: dict, rec_edges: list, rels: list, nodes: list) -> None:
    """Pull recommendation and relation edges out of one detail payload."""
    nodes.append(normalise_node(d) | {"mal_fetched_at": _now()})
    recs = d.get("recommendations") or []
    total = sum(r.get("num_recommendations", 0) for r in recs) or 1
    for r in recs:
        v = r.get("num_recommendations", 0)
        if v <= 0:
            continue
        # sqrt(v) rewards absolute agreement, v/total normalises away how
        # heavily any one anime is recommended overall
        rec_edges.append((aid, r["node"]["id"], "mal", v, math.sqrt(v) * (v / total)))
        nodes.append({"mal_id": r["node"]["id"], "title": r["node"]["title"]})
    for e in d.get("related_anime") or []:
        rels.append((aid, e["node"]["id"], (e.get("relation_type") or "other").lower(), "mal"))
        nodes.append({"mal_id": e["node"]["id"], "title": e["node"]["title"]})


def sync_graph(limit: int | None = None, only_missing: bool = True,
               scope: str = "listed") -> dict:
    """Fetch per-anime details to populate rec_edge and relation.

    Defaults to the user's own list (~250 requests, about 3 minutes). See
    _graph_targets for why the whole catalogue is not needed. Pushing past
    ~1.5 rps makes MAL 307-redirect requests to error.json, which is load
    shedding rather than a real "not found".

    Results are committed every GRAPH_FLUSH_EVERY anime and each one is stamped
    with graph_fetched_at as it lands, so an interrupted run resumes exactly
    where it stopped. Transient failures are deliberately left unstamped so the
    next run picks them up again.
    """
    targets = _graph_targets(limit, only_missing, scope)
    if not targets:
        log.info("graph: nothing pending")
        return {"fetched": 0, "pending": 0}
    log.info("graph: %d anime to fetch (~%.0f min)", len(targets),
             len(targets) / settings().mal_rps / 60)

    rec_edges: list[tuple] = []
    rels: list[tuple] = []
    nodes: list[dict] = []
    done_ids: list[int] = []
    n = 0
    failed = 0

    def flush() -> None:
        nonlocal rec_edges, rels, nodes, done_ids
        if not done_ids:
            return
        store.upsert_anime([x for x in nodes if x.get("title")])
        store.upsert_rec_edges(rec_edges)
        store.upsert_relations(rels)
        store.mark_graph_fetched(done_ids)
        rec_edges, rels, nodes, done_ids = [], [], [], []

    try:
        with MalClient() as mal:
            for aid, d in mal.details_many(targets):
                if d is None:
                    # transient failure - leave unstamped so the next run retries
                    failed += 1
                elif not d:
                    # a real 404; stamp it so we never ask again
                    done_ids.append(aid)
                else:
                    n += 1
                    _collect(aid, d, rec_edges, rels, nodes)
                    done_ids.append(aid)

                # NB: this check must not sit behind a `continue`, or a long run
                # of skipped ids never flushes and the work is lost on exit.
                if len(done_ids) >= GRAPH_FLUSH_EVERY:
                    flush()
                    log.info("graph: %d/%d stored (%d failed)", n, len(targets), failed)
    finally:
        # whatever was fetched before an interrupt stays on disk
        flush()

    passes = refresh_franchises()
    detail = {"fetched": n, "failed_retry_later": failed, "franchise_passes": passes,
              "rec_edges": scalar("SELECT count(*) FROM rec_edge"),
              "relations": scalar("SELECT count(*) FROM relation")}
    log_ingest("sync_graph", "ok", detail)
    log.info("graph: %s", detail)
    return detail


def sync_anilist(limit: int | None = None, only_missing: bool = True) -> dict:
    """Weighted tags + second consensus + second recommendation graph.
    Batched 50 per GraphQL call, so this is fast despite the 30/min limit."""
    sql = """
        SELECT a.mal_id
          FROM anime a
          LEFT JOIN list_entry le ON le.mal_id = a.mal_id
         WHERE (coalesce(a.mal_num_scoring_users,0) >= %s OR le.mal_id IS NOT NULL)
    """
    if only_missing:
        sql += " AND a.al_fetched_at IS NULL"
    sql += """
         GROUP BY a.mal_id, a.mal_popularity, le.mal_id
         ORDER BY (le.mal_id IS NOT NULL) DESC, a.mal_popularity ASC NULLS LAST
    """
    ids = [r["mal_id"] for r in query(sql, (settings().min_scoring_users,))]
    if limit:
        ids = ids[:limit]
    if not ids:
        log_ingest("sync_anilist", "ok", {"fetched": 0})
        return {"fetched": 0}

    # Persist batch by batch: AniList allows 50 anime per request but only ~30
    # requests a minute, so a long run must not be all-or-nothing.
    media: dict[int, dict] = {}
    batch = settings().anilist_batch
    with AniListClient() as al:
        for i in range(0, len(ids), batch):
            chunk = ids[i:i + batch]
            got = al.media_by_mal_ids(chunk)
            if got:
                _store_anilist_batch(got)
                media.update(got)
            # ids AniList does not know still count as attempted, otherwise
            # every later run retries them forever
            store.mark_anilist_attempted([c for c in chunk if c not in got])
            log.info("anilist: %d/%d stored", min(i + batch, len(ids)), len(ids))

    tags = [t for m in media.values() for t in extract_tags(m)]
    edges, rels = [], []
    for mal_id, m in media.items():
        nodes = (m.get("recommendations") or {}).get("nodes") or []
        total = sum(max(n.get("rating") or 0, 0) for n in nodes) or 1
        for nd in nodes:
            mr = nd.get("mediaRecommendation") or {}
            v = nd.get("rating") or 0
            if not mr.get("idMal") or v <= 0:
                continue
            edges.append((mal_id, mr["idMal"], "anilist", v, math.sqrt(v) * (v / total)))
        for e in (m.get("relations") or {}).get("edges") or []:
            nd = e.get("node") or {}
            if nd.get("type") == "ANIME" and nd.get("idMal"):
                rt = RELATION_MAP.get(e.get("relationType") or "", "other")
                rels.append((mal_id, nd["idMal"], rt, "anilist"))

    vecs = store.store_tag_vectors()
    passes = refresh_franchises()
    detail = {"fetched": len(media), "requested": len(ids), "tags": len(tags),
              "al_edges": len(edges), "tag_vectors": vecs, "franchise_passes": passes}
    log_ingest("sync_anilist", "ok", detail)
    log.info("anilist: %s", detail)
    return detail


def _store_anilist_batch(media: dict[int, dict]) -> None:
    """Write one AniList batch: scores, tags, edges and relations."""
    store.upsert_anilist([normalise_media(m) for m in media.values()])
    store.replace_tags([t for m in media.values() for t in extract_tags(m)])
    store.replace_staff([s for m in media.values() for s in extract_staff(m)])

    edges, rels = [], []
    for mal_id, m in media.items():
        nodes = (m.get("recommendations") or {}).get("nodes") or []
        total = sum(max(n.get("rating") or 0, 0) for n in nodes) or 1
        for nd in nodes:
            mr = nd.get("mediaRecommendation") or {}
            v = nd.get("rating") or 0
            if mr.get("idMal") and v > 0:
                edges.append((mal_id, mr["idMal"], "anilist", v, math.sqrt(v) * (v / total)))
        for e in (m.get("relations") or {}).get("edges") or []:
            nd = e.get("node") or {}
            if nd.get("type") == "ANIME" and nd.get("idMal"):
                rels.append((mal_id, nd["idMal"], RELATION_MAP.get(e.get("relationType") or "",
                                                                   "other"), "anilist"))
    # AniList references anime the MAL catalog pass may never have seen; an
    # edge pointing at an unknown anime is dead weight.
    if edges or rels:
        wanted = {e[1] for e in edges} | {r[1] for r in rels}
        known = {r["mal_id"] for r in query(
            "SELECT mal_id FROM anime WHERE mal_id = ANY(%s)", (sorted(wanted),))}
        store.upsert_rec_edges([e for e in edges if e[1] in known])
        store.upsert_relations([r for r in rels if r[1] in known])


def bootstrap(username: str | None = None, with_catalog: bool = True) -> dict:
    """Cold start for one user, fetching only what that user needs.

    1. their list                       1 request
    2. broad candidate metadata        ~30 requests, bulk ranking endpoints
    3. MAL details for their list     ~250 requests, once
    4. AniList for their rated anime   ~3 batched requests

    Candidates are enriched later, on demand, when a surface is actually built
    - so anime nobody is ever shown are never fetched.
    """
    from .ondemand import ensure_for_user

    out: dict = {"user_list": sync_user_list(username, enrich=False)}
    if with_catalog:
        out["catalog"] = sync_catalog()
    uid = store.get_or_create_user(username or settings().malrec_user)
    out["on_demand"] = ensure_for_user(uid)
    return out


UPCOMING_MIN_MEMBERS = 1000


def sync_upcoming(min_members: int = UPCOMING_MIN_MEMBERS) -> dict:
    """Keep "coming soon" current: re-read the rankings and season lists
    (~30 requests; new announcements and status changes arrive with them),
    then fetch relations for upcoming titles that lack them, so they join
    their franchise. Titles with fewer than `min_members` list users are left
    alone - announced sequels of anything people care about pass that
    within days."""
    from .ondemand import ensure_graph
    cat = sync_catalog()
    ids = [r["mal_id"] for r in query(
        "SELECT mal_id FROM anime WHERE status = 'not_yet_aired' AND graph_fetched_at IS NULL"
        " AND coalesce(mal_num_list_users, 0) >= %s ORDER BY mal_num_list_users DESC",
        (min_members,))]
    out = {"catalog": cat["anime"], "related": ensure_graph(ids)}
    log_ingest("sync_upcoming", "ok", out)
    return out
