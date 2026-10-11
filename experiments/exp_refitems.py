"""Which of the reference profile's held-out titles does a population model
get worse - and are they the season that was airing when the new lists were
fetched?

The reference profile's newest 60 ratings, predicted from the older ones by
the production path (temporal_holdout's fit), once per stored model. Prints
the titles whose squared error changed most, with their season, raters in
each model and the share of those raters' entries still "watching", then
the change grouped by start year.

  python -m experiments.exp_refitems 11 14 [--user NAME] [--h 60]
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from malrec.config import settings
from malrec.db import query, require_one
from malrec.eval import spearman
from malrec.recsys.hybrid import fit_user
from malrec.recsys.service import (_half_life, blend_mode, blend_opts, item_store, load_global,
                                   user_examples)


def main(a: int, b: int, user: str, h: int):
    uid = require_one("SELECT id FROM app_user WHERE mal_username=%s", (user,))["id"]
    rated = query("""SELECT mal_id, score, coalesce(finished_at::timestamptz, updated_at, now()) AS at
                       FROM list_entry WHERE user_id=%s AND score>0 ORDER BY at""", (uid,))
    test = rated[-h:]
    cutoff = test[0]["at"]
    store = item_store()
    hl = _half_life(uid)
    scored, implicit = user_examples(uid, cutoff, hl)
    items = [r["mal_id"] for r in test]
    y = np.array([float(r["score"]) for r in test])
    preds, raters = {}, {}
    for gid in (a, b):
        gm = load_global(gid)
        um = fit_user(gm.pop, gm.stacker, store, scored, implicit, mode=blend_mode(len(rated)),
                      blend_of="personal", **{**blend_opts(), "size_n": len(rated)})
        preds[gid] = np.asarray(um.predict(items), dtype=float)
        raters[gid] = {m: int(gm.pop.n_raters[gm.pop.index[m]]) if m in gm.pop.index else 0
                       for m in items}
        ok = ~np.isnan(preds[gid])
        print(f"model {gid}: rho {spearman(preds[gid][ok], y[ok]):.4f} on {ok.sum()} titles")
    watching = {r["mal_id"]: float(r["w"]) for r in query(
        """SELECT mal_id, avg((status='watching')::int) w FROM cf_rating
            WHERE mal_id = ANY(%s) AND score > 0 GROUP BY 1""", (items,))}
    print(f"cutoff {cutoff:%Y-%m-%d}, newest rating {test[-1]['at']:%Y-%m-%d}\n")
    d = (preds[b] - y) ** 2 - (preds[a] - y) ** 2
    order = np.argsort(-np.abs(np.nan_to_num(d)))
    print(f"{'title':<42}{'start':>8}{'you':>5}{'m' + str(a):>7}{'m' + str(b):>7}"
          f"{'raters':>13}{'watching':>9}{'d err²':>8}")
    for i in order[:20]:
        m = items[i]
        row = store.rows.get(m) or {}
        start = f"{row.get('season_year') or '?'}{'*' if row.get('status') == 'currently_airing' else ''}"
        print(f"{(row.get('title') or str(m))[:41]:<42}{str(start)[:7]:>8}{y[i]:5.0f}"
              f"{preds[a][i]:7.2f}{preds[b][i]:7.2f}{raters[a][m]:6d}/{raters[b][m]:<6d}"
              f"{watching.get(m, 0):9.2f}{d[i]:+8.2f}")
    by = defaultdict(list)
    for i, m in enumerate(items):
        row = store.rows.get(m) or {}
        by[str(row.get("season_year") or "?")].append(d[i])
    print("\nchange in squared error by start year (+ = model", b, "worse)")
    for k in sorted(by):
        v = np.array(by[k])
        print(f"  {k}: {len(v):3d} titles, mean {np.nanmean(v):+.3f}, sum {np.nansum(v):+.2f}")
    print("\nsettings: recency half-life", hl, "| ease", settings().relevance_ease)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=int)
    ap.add_argument("b", type=int)
    ap.add_argument("--user", default=None)
    ap.add_argument("--h", type=int, default=60)
    x = ap.parse_args()
    main(x.a, x.b, x.user or settings().malrec_user, x.h)
