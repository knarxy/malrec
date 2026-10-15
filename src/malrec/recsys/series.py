"""Which titles belong together: series and franchises, from MAL's relations.

A **series** is one story: its seasons and films (sequel / prequel), other
versions of it, and the side stories and recaps that hang off it. A
**franchise** is a series plus its spin-offs and alternative settings
(Gintama and Ginpachi-sensei, Fate/stay night and Fate/Apocrypha).

Both used to be plain connected components, which let one crossover join two
unrelated shows: "Nisekoimonogatari" is listed as a side story of both
Bakemonogatari and Nisekoi, so the two were one franchise - only one of them
could appear in a list, and "Not for me" on one removed the other. Here only
story chains merge outright; a title that hangs off a series joins it only
when it hangs off exactly one. A short that links two series is attached to
one of them and bridges nothing.

Series also weight a user's ratings (`weights`): thirteen seasons of one
show rated 10 are strong evidence of liking that show, but not thirteen
pieces of evidence about its genres. Each entry counts 1 / n ** power, n the
user's entries in that series (config.series_weight_power; 0 = off, as
measured; the genre profile uses PROFILE_SERIES_POWER).
"""
from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable

# MAL often lists several relation types for one pair of titles, in either
# direction (Ginpachi-sensei is both a spin-off and a side story of Gintama;
# 652 alternative settings are also listed as alternative versions). Each
# pair gets one kind, the first of these that any of its types has:
#   chain    the story continues (sequel / prequel)
#   world    a different story in the same world (spin-off, other setting)
#   retold   the same story told again (alternative version)
#   attach   hangs off a story (side story, recap)
KINDS = (("chain", frozenset({"sequel", "prequel"})),
         ("world", frozenset({"spin_off", "alternative_setting"})),
         ("retold", frozenset({"alternative_version"})),
         ("attach", frozenset({"side_story", "parent_story", "summary", "full_story"})))
SERIES_MERGE = frozenset({"chain", "retold"})      # one story
SERIES_HANG = frozenset({"attach"})
FRANCHISE_HANG = frozenset({"attach", "world"})    # plus its spin-offs


def kinds(relations: Iterable[tuple[int, int, str]]) -> dict[tuple[int, int], str]:
    """(lower id, higher id) -> the pair's kind; 'other' / 'character' links
    and self-links give none."""
    types: dict[tuple[int, int], set[str]] = {}
    for s, d, t in relations:
        if s != d:
            types.setdefault((min(s, d), max(s, d)), set()).add(t)
    out = {}
    for pair, ts in types.items():
        for kind, members in KINDS:
            if ts & members:
                out[pair] = kind
                break
    return out


class _Sets:
    """Union-find; a set's id is its smallest member, so ids are stable."""

    def __init__(self, nodes: Iterable[int]):
        self.parent = {n: n for n in nodes}

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def group(nodes: Iterable[int], pairs: dict[tuple[int, int], str],
          hang: frozenset[str]) -> dict[int, int]:
    """mal_id -> group id. One story (SERIES_MERGE pairs) merges outright.
    Pairs of a `hang` kind then merge only upward: a group joins a larger
    one it hangs off, smallest groups first. A group hanging off one larger
    group joins it; a lone title hanging off several (a crossover short)
    joins the largest and its other links are dropped; a longer series
    hanging off several stays on its own. A dropped link is gone for good,
    so no later merge can turn it into a bridge between two series."""
    sets = _Sets(set(nodes) | {x for pair in pairs for x in pair})
    for (a, b), kind in pairs.items():
        if kind in SERIES_MERGE:
            sets.union(a, b)
    links = {pair for pair, kind in pairs.items() if kind in hang}
    by_node: dict[int, set[tuple[int, int]]] = {}
    for pair in links:
        for x in pair:
            by_node.setdefault(x, set()).add(pair)
    members: dict[int, list[int]] = {}
    for n in sets.parent:
        members.setdefault(sets.find(n), []).append(n)

    def key(g: int) -> tuple[int, int]:
        return (len(members[g]), -g)    # larger, then lower id, ranks higher

    def drop(pair: tuple[int, int]) -> None:
        links.discard(pair)
        for x in pair:
            by_node[x].discard(pair)

    changed = True
    while changed and links:
        changed = False
        for pair in [p for p in links if sets.find(p[0]) == sets.find(p[1])]:
            drop(pair)                  # inside one group now: spent
        active = sorted({sets.find(x) for pair in links for x in pair}, key=key)
        for g in active:
            g = sets.find(g)
            up: dict[int, set[tuple[int, int]]] = {}
            for n in members[g]:
                for pair in by_node.get(n, ()):
                    other = sets.find(pair[1] if pair[0] == n else pair[0])
                    if other != g and key(other) > key(g):
                        up.setdefault(other, set()).add(pair)
            if not up:
                continue
            if len(up) == 1 or len(members[g]) == 1:
                target = max(up, key=key)
                for ls in up.values():
                    for pair in ls:
                        drop(pair)
                sets.union(g, target)
                root = sets.find(g)
                gone = target if root == g else g
                members[root] = members[root] + members.pop(gone)
                changed = True
            else:
                for ls in up.values():
                    for pair in ls:
                        drop(pair)
    return {n: sets.find(n) for n in sets.parent}


def groups(nodes: Iterable[int], relations: Iterable[tuple[int, int, str]]
           ) -> tuple[dict[int, int], dict[int, int]]:
    """(series, franchise) for every title."""
    pairs = kinds(relations)
    nodes = list(nodes)
    return group(nodes, pairs, SERIES_HANG), group(nodes, pairs, FRANCHISE_HANG)


def refresh() -> tuple[int, int]:
    """Recompute both tables from `relation`. Returns (series, franchises)
    with more than one title."""
    from ..db import conn, query
    nodes = [r["mal_id"] for r in query("SELECT mal_id FROM anime")]
    rel = [(r["src"], r["dst"], r["relation_type"])
           for r in query("SELECT src, dst, relation_type FROM relation")]
    series, franchise = groups(nodes, rel)
    with conn() as c, c.cursor() as cur:
        cur.execute("TRUNCATE series, franchise")
        with cur.copy("COPY series (mal_id, series_id) FROM STDIN") as cp:
            for m, s in series.items():
                cp.write_row((m, s))
        with cur.copy("COPY franchise (mal_id, franchise_id) FROM STDIN") as cp:
            for m, f in franchise.items():
                cp.write_row((m, f))
        c.commit()
    _cache["at"] = 0.0

    def multi(g: dict[int, int]) -> int:
        return sum(1 for v in Counter(g.values()).values() if v > 1)
    return multi(series), multi(franchise)


_cache: dict = {"at": 0.0, "series": {}}


def series_of() -> dict[int, int]:
    """mal_id -> series id, cached for an hour (titles without relations are
    missing: each is its own series)."""
    if not _cache["series"] or time.time() - _cache["at"] > 3600:
        from ..db import query
        _cache["series"] = {r["mal_id"]: r["series_id"] for r in query(
            "SELECT mal_id, series_id FROM series WHERE mal_id <> series_id"
            " OR series_id IN (SELECT series_id FROM series GROUP BY 1 HAVING count(*) > 1)")}
        _cache["at"] = time.time()
    return _cache["series"]


def shares(ids: Iterable[int], series: dict[int, int], power: float) -> dict[int, float]:
    """Each title's weight as evidence of taste: 1 / n ** power, n the
    number of `ids` in its series."""
    ids = list(ids)
    n = Counter(series.get(m, m) for m in ids)
    return {m: n[series.get(m, m)] ** -power for m in ids}


def weights(ids: Iterable[int]) -> dict[int, float]:
    """`shares` at the configured power; empty when it is off, so callers
    use .get(m, 1.0)."""
    from ..config import settings
    power = settings().series_weight_power
    if power <= 0:
        return {}
    return shares(ids, series_of(), power)


GENRE_MIN_SERIES = 4
# The profile weights series at its own, fixed power: it describes taste, so
# one long show gets one show's say. series_weight_power stays 0 for the
# ranking, where weighting cost accuracy on long lists (RESEARCH.md).
PROFILE_SERIES_POWER = 1.0


def genre_lifts(rated: list[dict], series: dict[int, int] | None = None,
                power: float = PROFILE_SERIES_POWER) -> list[dict]:
    """The profile's "genres you rate above your own average": per genre,
    the series-weighted mean score minus the user's, best first. A genre
    needs GENRE_MIN_SERIES different series behind it - one long show is a
    taste for that show, not for its genres (a dozen Gintama seasons rated
    10 would otherwise make Gag Humor, Samurai and Parody someone's top
    genres). `rated`: mal_id, score, mal_genres."""
    series = series_of() if series is None else series
    w = shares((r["mal_id"] for r in rated), series, power)
    tot = sum(w.values())
    if not rated or tot <= 0:
        return []
    mean = sum(w[r["mal_id"]] * r["score"] for r in rated) / tot
    acc: dict[str, list] = {}
    for r in rated:
        for g in r["mal_genres"] or []:
            a = acc.setdefault(g, [0.0, 0.0, 0, set()])
            a[0] += w[r["mal_id"]] * r["score"]
            a[1] += w[r["mal_id"]]
            a[2] += 1
            a[3].add(series.get(r["mal_id"], r["mal_id"]))
    out = [{"genre": g, "n": n, "lift": round(sw / ws - mean, 2)}
           for g, (sw, ws, n, ss) in acc.items() if len(ss) >= GENRE_MIN_SERIES]
    return sorted(out, key=lambda x: (-x["lift"], x["genre"]))
