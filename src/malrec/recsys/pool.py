"""What a discovery tab may show one user, in memory.

Safe Bets, Discover and Hidden Gems draw from eligible_candidates (SQL), less
the titles Up Next shows (surfaces.continuation_ids). The benchmark and the
learned orders' training have no stored user to run that for: they take the
catalogue rules from ltr.surface_pool and the parts that depend on the
user's list from here -

  prerequisites  a title whose main-format prequel chain is not all
                 completed (prerequisites() / prereqs_satisfied(),
                 db/migrations/002_logic.sql and 012_prereq_main.sql)
  continuations  the direct main-format sequels of a completed title rated
                 at least max(round(mean) - 1, 6): what Up Next takes
                 (surfaces.NEXT_UP_SQL), so the discovery tabs leave them out

Without these the benchmark scored the tabs on a pool production never
shows: a fifth of the titles a user went on to like sat behind an unmet
prerequisite and an eighth were continuations, the easiest hits there are
(audit 2026-10-23, P1-2). Approximations: the mean is the plain one (SQL
user_mean() weights by recency), and every qualifying sequel counts, where
Up Next keeps one per franchise and at most 60.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

CHAIN_DEPTH = 12            # prerequisites() follows prequels this deep


@dataclass(frozen=True)
class Relations:
    chain: dict[int, frozenset[int]]      # title -> the main-format prequels it needs completed
    sequels: dict[int, frozenset[int]]    # title -> its direct main-format sequels

    def exclusions(self, entries: list[dict]) -> set[int]:
        """The titles a list of these entries (mal_id, score, status) keeps
        out of the discovery tabs: unmet prerequisites and continuations."""
        completed = {e["mal_id"] for e in entries if e["status"] == "completed"}
        unmet = {m for m, need in self.chain.items() if not need <= completed}
        scores = [e["score"] for e in entries if e["score"] > 0]
        least = max(round(float(np.mean(scores))) - 1, 6) if scores else 6
        liked = {e["mal_id"] for e in entries
                 if e["status"] == "completed" and e["score"] >= least}
        return unmet | ({s for p in liked for s in self.sequels.get(p, ())} - unmet)


def relations(main: set[int], ids) -> Relations:
    """The prequel chains of `ids` and every direct sequel, restricted to
    the `main`-format titles, from the relation table."""
    from ..db import query
    pre: dict[int, set[int]] = defaultdict(set)
    seq: dict[int, set[int]] = defaultdict(set)
    for r in query("SELECT src, dst, relation_type FROM relation"
                   " WHERE relation_type IN ('prequel', 'sequel')"):
        (pre if r["relation_type"] == "prequel" else seq)[r["src"]].add(r["dst"])
    return Relations(prequel_chains(pre, main, ids),
                     {p: frozenset(s & main) for p, s in seq.items() if s & main})


def prequel_chains(pre: dict[int, set[int]], main: set[int], ids) -> dict[int, frozenset[int]]:
    """Per title of `ids`, the `main` titles among its prequels up to
    CHAIN_DEPTH steps back, walked through any format (an OVA between two
    seasons still gates the second); titles without one are left out."""
    out: dict[int, frozenset[int]] = {}
    for m in ids:
        seen: set[int] = set()
        frontier, depth = {m}, 0
        while frontier and depth < CHAIN_DEPTH:
            frontier = {p for x in frontier for p in pre.get(x, ())} - seen
            seen |= frontier
            depth += 1
        need = frozenset((seen - {m}) & main)
        if need:
            out[m] = need
    return out
