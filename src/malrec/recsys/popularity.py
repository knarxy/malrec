"""How well known a title is, fairly to its age.

MAL's member count is a total: a show that started three months ago has had
three months to collect it, one from 2010 sixteen years. Ranked raw, nearly
every new show counts as a little-known "hidden gem" (2026-10-20: 37 % of
Hidden Gems' top 20 started 2025 or later). Dividing by age would do the
opposite wrong - old titles stopped growing long ago - so a title is
instead scaled up by the share of its eventual audience a title of its age
has typically reached (GROWTH, experiments/exp_growth.py): three years in
it is 75 %, after about ten years all, so nothing is ever scaled down and
the oldest titles keep their count.

`gems_popularity` (config) picks the Hidden Gems pool rule: 'rank' (MAL's
raw popularity rank past GEMS_FROM, shipped until 2026-10) or 'grown': the
age-scaled count below what raw rank GEMS_FROM has, and below what
YOUNG_GEMS_FROM has for a title under a year old - a season's talking point is no hidden gem even when
it will not reach the all-time top 1,200 (results_gems_2026-10-21.txt).
"""
from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import numpy as np

GEMS_FROM = 1200            # Hidden Gems: titles past this popularity rank
YOUNG_GEMS_FROM = 2000      # ... and past this one while under YOUNG_MONTHS old
YOUNG_MONTHS = 12
# The curve is twice too steep in the first three months (completion dates:
# a season counts once it ends), so a title days old is scaled at most fivefold
SHARE_FLOOR = 0.2
# (months since start, median share of the eventual audience reached), from
# the sampled lists' entry dates, experiments/exp_growth.py (2026-10-20)
GROWTH: tuple[tuple[float, float], ...] = (
    (0.0, 0.0), (1.0, 0.101), (2.0, 0.125), (3.0, 0.347), (4.0, 0.403), (6.0, 0.462),
    (9.0, 0.521), (12.0, 0.563), (18.0, 0.624), (24.0, 0.672), (36.0, 0.747), (48.0, 0.805),
    (60.0, 0.855), (72.0, 0.894), (96.0, 0.958), (121.0, 1.0))


def share_at(months: float) -> float:
    xs, ys = zip(*GROWTH, strict=True)
    return max(SHARE_FLOOR, min(1.0, float(np.interp(months, xs, ys))))


def months_since(start: dt.date | None, today: dt.date) -> float | None:
    return None if start is None else max(0.0, (today - start).days / 30.44)


def grown(members: int | None, start: dt.date | None, today: dt.date) -> float:
    """The member count a title of this age is on course for."""
    m = float(members or 0)
    age = months_since(start, today)
    return m if age is None else m / share_at(age)


def grown_ranks(rows: Iterable[tuple[int, int | None, dt.date | None]],
                today: dt.date) -> dict[int, int]:
    """mal_id -> rank by grown member count (1 = best known), over `rows`
    of (mal_id, members, start date) - the whole catalogue, like MAL's rank."""
    g = sorted(((grown(n, s, today), m) for m, n, s in rows), reverse=True)
    return {m: k + 1 for k, (_, m) in enumerate(g)}


def bars(members: Iterable[int | None]) -> tuple[float, float]:
    """The member counts MAL's raw ranks GEMS_FROM and YOUNG_GEMS_FROM have
    today, over the whole catalogue: what a title's grown count has to stay
    under. A count, not a rank among grown counts - ranked, the scaled-up
    new titles pushed ~100 older ones past #1,200 into the pool."""
    n = sorted((m or 0 for m in members), reverse=True)

    def at(k: int) -> float:
        return float(n[k - 1]) if len(n) >= k else 0.0
    return at(GEMS_FROM), at(YOUNG_GEMS_FROM)


def is_gem(row: dict, rule: str, cuts: tuple[float, float] | None, today: dt.date) -> bool:
    """Whether a title (mal_popularity, members, start_date) belongs in the
    Hidden Gems pool under `rule`; `cuts` from bars()."""
    if rule != "grown":
        return (row.get("mal_popularity") or 99_999) > GEMS_FROM
    old, young = cuts or (0.0, 0.0)
    age = months_since(row.get("start_date"), today)
    cut = young if age is not None and age < YOUNG_MONTHS else old
    return grown(row.get("members"), row.get("start_date"), today) < cut


_cache: dict = {"day": None, "ids": frozenset()}


def gem_ids(rule: str) -> frozenset[int]:
    """Every catalogue title in the Hidden Gems pool under `rule` today
    (computed once a day: the bars need the whole catalogue)."""
    today = dt.datetime.now(dt.UTC).date()
    if _cache["day"] != (today, rule):
        from ..db import query
        rows = query("SELECT mal_id, mal_popularity, mal_num_list_users AS members, start_date"
                     "  FROM anime")
        cuts = bars(r["members"] for r in rows)
        _cache["ids"] = frozenset(r["mal_id"] for r in rows if is_gem(r, rule, cuts, today))
        _cache["day"] = (today, rule)
    return _cache["ids"]
