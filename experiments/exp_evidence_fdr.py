"""How many of the "clear" attributes behind the evidence lines are chance,
and which rule keeps the real ones (audit 2026-10-23, P1-12). Read-only.

taste.attribute_groups tests every genre, theme, studio, format, era, length
and source of a list against the user's own mean (|z| >= 2, themes 2.5) - 40
to 60 tests on a long list, none corrected for the others. Each "clear" one
becomes a line on many cards ("you rate Studio X higher").

For sampled lists (cf_rating, >= taste.MIN_RATED_SERIES rated series):
  real   the list's groups as they are
  null   the same list with its series' scores shuffled among its series
         (PERMS times): every "clear" there is chance by construction, so
         null / real estimates the share of chance among the real ones (FDR)
  split  the rated series halved at random: of what is clear on one half,
         the share whose lift on the other half points the same way, and
         that lift (in the claimed direction)
Rules (each over one user's whole family of tests, the groups with enough
series; every rule but the fixed z ones keeps today's z bars as well, so
none says more than today):
  now        |z| >= 2, themes 2.5
  z2.5, z3   one bar for every kind
  holm       Holm at .05
  bh.05/.10/.20  Benjamini-Hochberg at that FDR
  ...t       the same on p from Student's t (n - 1 degrees of freedom)
             instead of the normal: a group has 5 to 8 series, where the
             normal's tails are far too thin

  docker compose --profile lab run --rm -T lab python -m experiments.exp_evidence_fdr [--profile]

--profile: the profile page's family instead (taste.breakdown: genres,
themes, eras and the first studio), and the app accounts' claims on it.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.stats import t as student_t

from malrec import taste
from malrec.db import query
from malrec.recsys.series import series_of

USERS = 2000
PERMS = 5
SEED = 7
BUCKETS = ((15, 49), (50, 149), (150, 10 ** 9))
RULES = ("now", "z2.5", "z3", "holm", "bh.05", "bh.10", "bh.20", "holm.t", "bh.05t",
         "bh.10t", "bh.20t")


def tests(by_series: dict[int, float], members: dict[str, dict[str, set[int]]]) -> list[dict]:
    """Every group with enough series to be called clear: kind, name, lift,
    z (taste._groups' formula, unrounded), p (two-sided, normal) and pt
    (two-sided, Student's t with n - 1 degrees of freedom)."""
    if len(by_series) < taste.MIN_RATED_SERIES:
        return []
    mu = sum(by_series.values()) / len(by_series)
    out = []
    for kind, groups in members.items():
        min_k = taste.THEME_MIN_SERIES if kind == "theme" else taste.MIN_SERIES
        for name, ss in groups.items():
            vals = [by_series[s] for s in ss if s in by_series]
            n = len(vals)
            if n < min_k:
                continue
            m = sum(vals) / n
            sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1))
            z = (m - mu) / (max(sd, taste.SD_FLOOR) / math.sqrt(n))
            out.append({"kind": kind, "name": name, "lift": m - mu, "z": z,
                        "p": math.erfc(abs(z) / math.sqrt(2)),
                        "pt": float(2 * student_t.sf(abs(z), n - 1))})
    return out


def today(t: dict) -> bool:
    return abs(t["z"]) >= (taste.THEME_Z if t["kind"] == "theme" else taste.Z)


def clear(ts: list[dict], rule: str) -> list[dict]:
    if rule == "now":
        return [t for t in ts if today(t)]
    if rule.startswith("z"):
        return [t for t in ts if abs(t["z"]) >= float(rule[1:])]
    pk = "pt" if rule.endswith("t") else "p"
    rule = rule.removesuffix("t").removesuffix(".")
    m = len(ts)
    order = sorted(range(m), key=lambda k: ts[k][pk])
    keep: set[int] = set()
    if rule == "holm":
        for i, k in enumerate(order):
            if ts[k][pk] > 0.05 / (m - i):
                break
            keep.add(k)
    else:                                       # Benjamini-Hochberg step-up
        q = float(rule[2:])
        last = max((i for i, k in enumerate(order) if ts[k][pk] <= q * (i + 1) / m),
                   default=-1)
        keep = set(order[:last + 1])
    return [ts[k] for k in sorted(keep) if today(ts[k])]


PROFILE = {"on": False}           # --profile: the profile page's family


def members_of(rated: list[dict], vals: dict[int, dict], sid: dict[int, int]
               ) -> dict[str, dict[str, set[int]]]:
    kinds = ("genre", "theme", "era", "studio") if PROFILE["on"] else taste.ATTRIBUTES
    out: dict[str, dict[str, set[int]]] = {k: defaultdict(set) for k in kinds}
    for e in rated:
        for kind in kinds:
            vs = vals[e["mal_id"]][kind]
            for v in (vs[:1] if PROFILE["on"] and kind == "studio" else vs):
                out[kind][v].add(sid[e["mal_id"]])
    return out


def app_profile(series: dict[int, int], genre_names: set[str]) -> None:
    """What each rule leaves on the approved accounts' profile pages: the
    clear groups (all four cards together) per account. Totals only."""
    users = [r["id"] for r in query("SELECT id FROM app_user WHERE status = 'approved'")]
    tot = {rule: [0, 0] for rule in RULES}            # claims, accounts with any
    for uid in users:
        es = [e for e in query(taste.ENTRY_SQL, (uid,)) if (e["score"] or 0) > 0]
        titles, tags = taste.catalogue([e["mal_id"] for e in es])
        es = [e for e in es if e["mal_id"] in titles]
        sid = {e["mal_id"]: series.get(e["mal_id"], e["mal_id"]) for e in es}
        per: dict[int, list[int]] = defaultdict(list)
        for e in es:
            per[sid[e["mal_id"]]].append(e["score"])
        by_series = {s: sum(v) / len(v) for s, v in per.items()}
        vals = {m: taste.attribute_values(t, tags.get(m, ()), genre_names)
                for m, t in titles.items()}
        ts = tests(by_series, members_of(es, vals, sid))
        for rule in RULES:
            n = len(clear(ts, rule))
            tot[rule][0] += n
            tot[rule][1] += n > 0
    print(f"\napp accounts ({len(users)}), profile page: clear groups per account,"
          " accounts with any")
    for rule in RULES:
        print(f"   {rule:<7}{tot[rule][0] / max(len(users), 1):7.2f}"
              f"{tot[rule][1] / max(len(users), 1):7.2f}")


def app_cards(series: dict[int, int], genre_names: set[str]) -> None:
    """What each rule leaves on the app's cards: over the approved accounts'
    stored Safe Bets, Discover and Hidden Gems lists, the share of cards
    with at least one evidence line and the lines per card (taste.evidence
    shows up to EVIDENCE_SHOWN). Totals only."""
    users = [r["id"] for r in query("SELECT id FROM app_user WHERE status = 'approved'")]
    tot = {rule: [0, 0, 0] for rule in RULES}         # cards, with a line, lines
    for uid in users:
        es = [e for e in query(taste.ENTRY_SQL, (uid,)) if (e["score"] or 0) > 0]
        recs = [r["mal_id"] for r in query(
            "SELECT mal_id FROM recommendation WHERE user_id = %s"
            " AND surface IN ('safe_bets', 'discover', 'hidden_gems')", (uid,))]
        titles, tags = taste.catalogue([e["mal_id"] for e in es] + recs)
        es = [e for e in es if e["mal_id"] in titles]
        sid = {e["mal_id"]: series.get(e["mal_id"], e["mal_id"]) for e in es}
        per: dict[int, list[int]] = defaultdict(list)
        for e in es:
            per[sid[e["mal_id"]]].append(e["score"])
        by_series = {s: sum(v) / len(v) for s, v in per.items()}
        vals = {m: taste.attribute_values(t, tags.get(m, ()), genre_names)
                for m, t in titles.items()}
        ts = tests(by_series, members_of(es, vals, sid))
        for rule in RULES:
            ok = {(t["kind"], t["name"]) for t in clear(ts, rule)}
            for m in recs:
                if m not in vals:
                    continue
                n = sum((k, v) in ok for k, vs in vals[m].items() for v in vs)
                tot[rule][0] += 1
                tot[rule][1] += n > 0
                tot[rule][2] += min(n, taste.EVIDENCE_SHOWN)
    print(f"\napp accounts ({len(users)}), their stored lists: cards with an evidence line,"
          " lines per card")
    for rule in RULES:
        c, w, n = tot[rule]
        print(f"   {rule:<7}{w / max(c, 1):7.2f}{n / max(c, 1):7.2f}   ({c} cards)")


def main() -> int:
    t0 = time.time()
    rng = random.Random(SEED)
    users = [r["user_id"] for r in query(
        "SELECT r.user_id FROM cf_rating r JOIN cf_user u ON u.id = r.user_id"
        " WHERE u.state = 'done' AND r.score > 0 GROUP BY 1 HAVING count(*) >= 30")]
    rng.shuffle(users)
    users = users[:USERS]
    rows = query("SELECT user_id, mal_id, score, status FROM cf_rating"
                 " WHERE user_id = ANY(%s) AND score > 0", (users,))
    lists: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        lists[r["user_id"]].append(r)
    titles, tags = taste.catalogue({r["mal_id"] for r in rows})
    series = series_of()
    genre_names = {g.lower() for g in taste.genre_names()}
    vals = {m: taste.attribute_values(t, tags.get(m, ()), genre_names) for m, t in titles.items()}
    print(f"{len(lists)} lists, {len(rows)} ratings, {len(titles)} titles [{time.time() - t0:.0f}s]",
          flush=True)

    # per rule and bucket: real / null clear counts, tests, users; split
    agg = {(rule, b): defaultdict(float) for rule in RULES for b in range(len(BUCKETS) + 1)}
    nrng = np.random.default_rng(SEED)
    for entries in lists.values():
        es = [e for e in entries if e["mal_id"] in titles]
        sid = {e["mal_id"]: series.get(e["mal_id"], e["mal_id"]) for e in es}
        per: dict[int, list[int]] = defaultdict(list)
        for e in es:
            per[sid[e["mal_id"]]].append(e["score"])
        by_series = {s: sum(v) / len(v) for s, v in per.items()}
        if len(by_series) < taste.MIN_RATED_SERIES:
            continue
        b = next(k for k, (lo, hi) in enumerate(BUCKETS) if lo <= len(by_series) <= hi)
        mem = members_of(es, vals, sid)
        real = tests(by_series, mem)
        keys, scores = list(by_series), np.array(list(by_series.values()))
        nulls = [tests(dict(zip(keys, nrng.permutation(scores).tolist())), mem)
                 for _ in range(PERMS)]
        half = set(nrng.permutation(keys)[:len(keys) // 2].tolist())
        a = {s: v for s, v in by_series.items() if s in half}
        c = {s: v for s, v in by_series.items() if s not in half}
        ta = tests(a, mem) if len(a) >= taste.MIN_RATED_SERIES else []
        mu_c = sum(c.values()) / len(c) if c else 0.0
        for rule in RULES:
            for bb in (b, len(BUCKETS)):
                g = agg[(rule, bb)]
                g["users"] += 1
                g["tests"] += len(real)
                g["real"] += len(clear(real, rule))
                g["null"] += sum(len(clear(n, rule)) for n in nulls) / PERMS
                for t in clear(ta, rule):
                    other = [c[s] for s in mem[t["kind"]][t["name"]] if s in c]
                    if len(other) < 2:
                        continue
                    lift = (sum(other) / len(other) - mu_c) * (1 if t["lift"] > 0 else -1)
                    g["rep_n"] += 1
                    g["rep_same"] += lift > 0
                    g["rep_lift"] += lift

    (app_profile if PROFILE["on"] else app_cards)(series, genre_names)
    names = [f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" for lo, hi in BUCKETS] + ["all"]
    print(f"\nper user: tests, clear (real), clear by chance (null), est. FDR = null / real,"
          f" real - null; split: replicated share, lift on the other half [{time.time() - t0:.0f}s]")
    for bb, name in enumerate(names):
        n = agg[("now", bb)]["users"]
        print(f"\n== rated series {name} ({n:.0f} lists, "
              f"{agg[('now', bb)]['tests'] / max(n, 1):.1f} tests each)")
        print(f"   {'rule':<7}{'clear':>7}{'chance':>8}{'FDR':>7}{'real':>7}{'repl':>7}{'lift':>7}")
        for rule in RULES:
            g = agg[(rule, bb)]
            real, null = g["real"] / max(n, 1), g["null"] / max(n, 1)
            fdr = null / real if real > 0 else float("nan")
            rep = g["rep_same"] / g["rep_n"] if g["rep_n"] else float("nan")
            lift = g["rep_lift"] / g["rep_n"] if g["rep_n"] else float("nan")
            print(f"   {rule:<7}{real:7.2f}{null:8.2f}{fdr:7.2f}{real - null:7.2f}"
                  f"{rep:7.2f}{lift:7.2f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    PROFILE["on"] = ap.parse_args().profile
    sys.exit(main())
