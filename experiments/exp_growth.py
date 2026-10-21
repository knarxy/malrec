"""How a title's audience grows after it starts, from the sampled lists.

"Under the radar" and the Hidden Gems pool read MAL's member count, which
a three-month-old show has had three months to collect and a 2010 show
sixteen years. Dividing by age would punish old titles, whose audience
stopped growing long ago; ranking within a release year would call a 1995
show with 20,000 members a hit. What is needed is the share of its eventual
audience a title has reached at a given age: young titles are scaled up by
it, old ones (share ~1) keep their count.

Estimated from titles that started 2012 - mid 2021 (all now five years or
more old), counting only list entries of sampled users who were already
active three months before the title started - otherwise users who joined
MAL later would make every title look like it grew late. An entry's date is
the earlier of its finish date and its last update, an upper bound on when
it was added, so the early shares are if anything too low.

Checked against reality: titles that started 2025-2026 gained MAL scorers
between the benchmark snapshot and today; the curve of scored entries
predicts that gain from their age alone.

  python -m experiments.exp_growth
"""
from __future__ import annotations

import datetime as dt
import time
from collections import defaultdict

import numpy as np

from experiments import bench
from malrec.db import query

MONTHS = (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 60)
CAP = 120

ENTRY_SQL = """
WITH ua AS (SELECT user_id, min(least(finished_at::timestamptz, updated_at)) AS first_at
              FROM cf_rating GROUP BY 1),
     t AS (SELECT mal_id, start_date, media_type FROM anime
            WHERE start_date BETWEEN %(lo)s AND %(hi)s AND media_type IN ('tv', 'movie', 'ona'))
SELECT r.mal_id, (r.score > 0) AS scored,
       least(%(cap)s, floor(greatest(0, extract(epoch FROM
             least(r.finished_at::timestamptz, r.updated_at) - t.start_date::timestamptz)
             / 86400 / 30.44)))::int AS m,
       count(*) AS n
  FROM cf_rating r JOIN t USING (mal_id) JOIN ua USING (user_id)
 WHERE ua.first_at <= t.start_date - interval '90 days'
 GROUP BY 1, 2, 3
"""


def curves(rows, min_entries: int = 300):
    """Per title the cumulative share of its entries by month; members (all
    entries) and scored entries separately."""
    hist: dict[tuple[int, bool], np.ndarray] = defaultdict(lambda: np.zeros(CAP + 1))
    for r in rows:
        hist[(r["mal_id"], False)][r["m"]] += r["n"]
        if r["scored"]:
            hist[(r["mal_id"], True)][r["m"]] += r["n"]
    out: dict[bool, dict[int, np.ndarray]] = {False: {}, True: {}}
    for (m, scored), h in hist.items():
        if h.sum() >= (min_entries if not scored else min_entries // 2):
            out[scored][m] = np.cumsum(h) / h.sum()
    return out


def table(name: str, cs: dict[int, np.ndarray]):
    if not cs:
        return
    a = np.array(list(cs.values()))     # column k-1 holds the share at ages under k months
    print(f"   {name:<26}" + "".join(f"{np.median(a[:, k - 1]):6.2f}" for k in MONTHS)
          + f"   ({len(a)} titles; q25/q75 at 3 m: {np.percentile(a[:, 2], 25):.2f}/"
            f"{np.percentile(a[:, 2], 75):.2f}, at 12 m: {np.percentile(a[:, 11], 25):.2f}/"
            f"{np.percentile(a[:, 11], 75):.2f})")


def main() -> tuple[tuple[float, float], ...]:
    t0 = time.time()
    rows = query(ENTRY_SQL, {"lo": dt.date(2012, 1, 1), "hi": dt.date(2021, 6, 30), "cap": CAP})
    info = {r["mal_id"]: r for r in query(
        "SELECT mal_id, media_type, mal_num_list_users FROM anime WHERE start_date IS NOT NULL")}
    print(f"{len(rows)} (title, month) cells [{time.time() - t0:.0f}s]")
    cs = curves(rows)
    print("\n== share of the eventual audience reached after k months (median over titles)")
    print(f"   {'':<26}" + "".join(f"{k:>5}m" for k in MONTHS))
    table("members, all", cs[False])
    table("scored, all", cs[True])
    for mt in ("tv", "movie", "ona"):
        table(f"members, {mt}", {m: c for m, c in cs[False].items() if info[m]["media_type"] == mt})
    pops = sorted((info[m]["mal_num_list_users"] or 0, m) for m in cs[False])
    third = len(pops) // 3
    for label, part in (("members, least known third", pops[:third]),
                        ("members, middle third", pops[third:2 * third]),
                        ("members, best known third", pops[2 * third:])):
        table(label, {m: cs[False][m] for _, m in part})

    # ---- check: the scored curve's predicted 10-day gain against MAL's
    med = np.median(np.array(list(cs[True].values())), axis=0)
    members = np.median(np.array(list(cs[False].values())), axis=0)
    growth = ((0.0, 0.0), *((float(k + 1), round(float(members[k]), 3))
                            for k in range(CAP + 1)))

    def share(months: float) -> float:          # cell m holds ages [m, m+1)
        return float(np.interp(months, np.arange(CAP + 2), np.concatenate([[0.0], med])))
    _, static = bench.load_snapshot()
    snap = dt.datetime.fromisoformat(static["created"])
    now = dt.datetime.now(dt.UTC)
    gap = (now - snap).days / 30.44
    young = query("""SELECT mal_id, start_date, mal_num_scoring_users FROM anime
                      WHERE start_date >= '2025-01-01' AND start_date <= %s
                        AND media_type IN ('tv', 'movie', 'ona')""", (snap.date(),))
    by: dict[str, list] = defaultdict(list)
    for r in young:
        then = (static["store"].rows.get(r["mal_id"]) or {}).get("mal_num_scoring_users")
        if not then or then < 500 or not r["mal_num_scoring_users"]:
            continue
        age = (snap.date() - r["start_date"]).days / 30.44
        seen = r["mal_num_scoring_users"] / then - 1
        expect = share(age + gap) / max(share(age), 1e-3) - 1
        b = "0-3 m" if age < 3 else "3-6 m" if age < 6 else "6-12 m" if age < 12 else "12-21 m"
        by[b].append((seen, expect))
    print(f"\n== MAL scorers gained in {gap * 30.44:.0f} days since the snapshot: seen vs curve")
    for b in ("0-3 m", "3-6 m", "6-12 m", "12-21 m"):
        v = np.array(by[b]) if by[b] else np.zeros((0, 2))
        if len(v):
            print(f"   started {b:<8} {len(v):4d} titles   seen median {np.median(v[:, 0]):+.3f}"
                  f"   curve median {np.median(v[:, 1]):+.3f}")
    print("\n   GROWTH = " + repr(tuple(x for x in growth if x[0] in (0.0, *map(float, MONTHS),
                                                             72.0, 96.0, 121.0))))
    print(f"[{time.time() - t0:.0f}s]")
    return growth


if __name__ == "__main__":
    main()
