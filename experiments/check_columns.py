"""One-off check: the column loader (service.load_cf_columns) gives the global
fit exactly what the row loader it replaced did, at a fraction of the memory.

The old load_cf_ratings, inlined, for 300 random sampled users, against
CfColumns.ratings() for the same users; then the peak memory of loading
everything as columns.
"""
from __future__ import annotations

import resource
import time

import numpy as np

from malrec.db import query
from malrec.recsys.service import EPOCH, load_cf_columns


def old_ratings(users: list[int]) -> dict[int, list[tuple]]:
    rows = query("""
        SELECT r.user_id, r.mal_id, r.score,
               coalesce(r.finished_at::timestamptz, r.updated_at) AS at
          FROM cf_rating r JOIN cf_user u ON u.id = r.user_id
         WHERE u.state = 'done' AND u.n_scored >= 5 AND r.score > 0
           AND r.user_id = ANY(%s)
         ORDER BY r.user_id, at, r.mal_id""", (users,))
    out: dict[int, list[tuple]] = {}
    for r in rows:
        out.setdefault(r["user_id"], []).append((r["mal_id"], float(r["score"]), r["at"] or EPOCH))
    return out


def main():
    t0 = time.time()
    cols = load_cf_columns()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"columns: {len(cols.user):,} entries, {len(np.unique(cols.user)):,} users, "
          f"{sum(a.nbytes for a in (cols.user, cols.mal_id, cols.score, cols.at)) / 2**20:.0f} MB "
          f"in arrays, process peak {peak:.0f} MB [{time.time() - t0:.0f}s]")
    users = np.random.default_rng(0).choice(cols.scored_users(), 300, replace=False).tolist()
    new, old = cols.ratings(users), old_ratings(users)
    assert set(new) == set(old), "different users"
    worst = 0.0
    for u in users:
        a, b = new[u], old[u]
        assert len(a) == len(b), (u, len(a), len(b))
        for (m1, s1, t1), (m2, s2, t2) in zip(a, b):
            assert (m1, s1) == (m2, s2), (u, m1, m2)
            worst = max(worst, abs((t1 - t2).total_seconds()))
    print(f"300 users identical in titles, scores and order; dates within {worst * 1e6:.1f} µs")
    assert worst < 1e-3
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
