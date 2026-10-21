"""The profile page's sentences, on sampled lists of every size.

Before the wording and thresholds of malrec.taste were fixed, every section
was computed for sampled MyAnimeList lists (cf_user / cf_rating) and the app's
own users: which headline each section gets, how often a section has nothing
clear to say, what the numbers behind the sentences look like, and whether
any list breaks it (no ratings, one score for everything, titles without a
year or tags). Sampled lists carry no episode counts, so their hours count
completed titles in full.

  python -m experiments.exp_profile [--per-bucket 300] [--show-app]
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import time
import traceback
from collections import Counter, defaultdict

import numpy as np

from malrec import taste
from malrec.db import query
from malrec.recsys.series import series_of

BUCKETS = ((1, 19), (20, 49), (50, 99), (100, 199), (200, 399), (400, 799), (800, 100_000))
SECTIONS = ("scale", "genres", "themes", "disagree", "eras", "studios")


def label(b):
    return f"{b[0]}-{b[1]}" if b[1] < 100_000 else f"{b[0]}+"


def load_titles():
    titles = {r["mal_id"]: r for r in query(taste.TITLE_SQL.replace(
        "WHERE mal_id = ANY(%s)", ""))}
    tags = defaultdict(list)
    for r in query("SELECT mal_id, tag, category FROM anime_tag WHERE rank >= %s",
                   (taste.TAG_RANK_MIN,)):
        if taste.theme_ok(r["category"]):
            tags[r["mal_id"]].append(r["tag"])
    return titles, tags


def sampled(per_bucket: int, seed: int) -> list[tuple[str, int, str]]:
    rng = random.Random(seed)
    out = []
    for b in BUCKETS:
        ids = [r["id"] for r in query(
            "SELECT id FROM cf_user WHERE state = 'done' AND n_entries BETWEEN %s AND %s",
            b)]
        rng.shuffle(ids)
        out += [("cf", u, label(b)) for u in ids[:per_bucket]]
    return out


def entries_of(kind: str, uid: int) -> list[dict]:
    if kind == "cf":
        return query("SELECT mal_id, status, score, NULL::int AS episodes_watched,"
                     "       coalesce(finished_at::timestamptz, updated_at) AS at"
                     "  FROM cf_rating WHERE user_id = %s", (uid,))
    return query("SELECT mal_id, status, score, episodes_watched,"
                 "       coalesce(finished_at::timestamptz, updated_at) AS at"
                 "  FROM list_entry WHERE user_id = %s", (uid,))


def bucket_of(n: int) -> str:
    return next(label(b) for b in BUCKETS if b[0] <= n <= b[1]) if n else "0"


def main(per_bucket: int, seed: int, show_app: bool = False):
    t0 = time.time()
    titles, tags = load_titles()
    series, genres = series_of(), taste.genre_names()
    users = sampled(per_bucket, seed)
    apps = query("SELECT id, mal_username FROM app_user WHERE status = 'approved'")
    users += [("app", r["id"], "") for r in apps]
    print(f"{len(titles)} titles, {len(tags)} with tags; {len(users)} lists"
          f" ({len(apps)} app users) [{time.time() - t0:.0f}s]", flush=True)

    keys: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    examples: dict[str, list] = defaultdict(list)
    nums: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    named: dict[str, Counter] = defaultdict(Counter)    # what headlines name, per section
    errors = []
    today = dt.date(2026, 10, 20)
    for kind, uid, b in users:
        es = entries_of(kind, uid)
        b = b or bucket_of(len(es))
        try:
            out = taste.breakdown(es, titles, tags, series, genres, today)
        except Exception:  # noqa: BLE001 - every failure is a finding here
            errors.append((kind, uid, traceback.format_exc(limit=3)))
            continue
        for s in SECTIONS:
            h = out[s].get("headline")
            k = h["key"] if h else f"{s}.(hidden)"
            keys[b][s][k] += 1
            keys["all"][s][k] += 1
            if h and len(examples[k]) < 3:
                examples[k].append({x: y for x, y in h.items() if x != "key"})
            for x in ("up", "down", "a", "b"):
                if h and isinstance(h.get(x), str):
                    named[f"{s}.{x if x in ('up', 'down') else 'up'}"][h[x]] += 1
        st, sc, tst = out["stats"], out["scale"], out["taste"]
        n = nums[b]
        n["rated_series"].append(tst["rated_series"])
        n["clear_genres"].append(len(out["genres"]["clear"]))
        n["clear_themes"].append(len(out["themes"]["clear"]))
        n["clear_studios"].append(len(out["studios"]["clear"]))
        n["era_rows"].append(len(out["eras"]["rows"]))
        n["hours"].append(st["hours"])
        if "diff" in sc:
            n["scale_diff"].append(sc["diff"])
        if kind == "app" and show_app:          # never in a saved result: names a person's taste
            print(f"\n-- app user {uid}: {st['entries']} entries, {tst['rated_series']} rated series")
            for s in SECTIONS:
                print(f"   {s:<9} {out[s].get('headline')}")

    print(f"\n{len(users) - len(errors)} computed, {len(errors)} errors [{time.time() - t0:.0f}s]")
    for e in errors[:5]:
        print(e[0], e[1], e[2])
    order = [label(b) for b in BUCKETS] + ["all"]
    for s in SECTIONS:
        allk = sorted({k for b in order for k in keys[b][s]})
        print(f"\n== {s} (share of lists per size)")
        print(f"   {'headline':<26}" + "".join(f"{b:>9}" for b in order))
        for k in allk:
            cells = []
            for b in order:
                tot = sum(keys[b][s].values())
                cells.append(f"{keys[b][s][k] / tot:9.2f}" if tot else f"{'':>9}")
            print(f"   {k:<26}" + "".join(cells))
    print("\n== numbers (median / 90th percentile per size)")
    for m in ("rated_series", "clear_genres", "clear_themes", "clear_studios", "era_rows",
              "hours", "scale_diff"):
        cells = []
        for b in order[:-1]:
            v = nums[b][m]
            cells.append(f"{np.median(v):6.1f}/{np.percentile(v, 90):<6.1f}" if v else f"{'':>13}")
        print(f"   {m:<14}" + "".join(f"{c:>14}" for c in cells))
    print("\n== names in headlines (count)")
    for k in sorted(named):
        print(f"   {k:<14} " + ", ".join(f"{n} {c}" for n, c in named[k].most_common(45)))
    seen = {k for s in SECTIONS for k in keys["all"][s]}
    print("\n== headlines never produced:",
          ", ".join(h for h in taste.HEADLINES if h not in seen and not h.startswith("pred."))
          or "none")
    print("== produced but unknown to the app:",
          ", ".join(sorted(k for k in seen if k not in taste.HEADLINES and "(hidden)" not in k))
          or "none")
    print("\n== examples per headline")
    for k in sorted(examples):
        print(f"   {k:<26} {examples[k]}")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--show-app", action="store_true", help="print each app user's headlines")
    a = ap.parse_args()
    main(a.per_bucket, a.seed, a.show_app)
