"""Do the "Why this?" panel's personal reasons say what the user's ratings say?

The panel lists the personal model's largest terms for a title ("− Witch",
"− a TV series"): ridge contributions, where a tag can come out negative
although the user rates titles with it above their average, because other
features already carry that. For every app user's top 20 in Safe Bets,
Discover and Hidden Gems this compares each such line with the user's own
series-level ratings of that attribute (malrec.taste.attribute_groups):

  backed        the attribute clearly differs from the user's average, same sign
  weak          same sign, but not clearly (few series, or within the noise)
  contradicted  the user rates it the other way
  unmapped      not an attribute (consensus terms, taste profile, ...)

and counts what the evidence-based alternative (malrec.taste.evidence) would
show per card instead. Aggregates only: nothing names a user or a title.

  python -m experiments.exp_evidence
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from malrec import taste
from malrec.db import query
from malrec.model import load_latest
from malrec.rank import _SOURCE_LABELS, _TYPE_LABELS
from malrec.recsys.scorer import _NEGATED, HybridScorer
from malrec.recsys.series import series_of

TABS = ("safe_bets", "discover", "hidden_gems")
UN_NEGATED = {v: k for k, v in _NEGATED.items()}
TYPES = {v: k for k, v in _TYPE_LABELS.items()}
SOURCES = {v: k for k, v in _SOURCE_LABELS.items()}


def attribute(label: str, genres: set[str]) -> tuple[str, str] | None:
    """(kind, name) of a driver label, or None when it is no attribute."""
    label = UN_NEGATED.get(label, label)
    if label in TYPES:
        return "format", TYPES[label]
    if label in SOURCES:
        return "source", SOURCES[label]
    if label.startswith("animated by "):
        return "studio", label[len("animated by "):]
    if label[:4].isdigit() and (label.endswith("s anime") or label[4:5] == "–"):
        return "era", label[:4]
    if label.endswith(" episodes"):
        return "length", label[:-len(" episodes")]
    if label in genres:
        return "genre", label
    if " " not in label or label[:1].isupper():
        return "theme", label
    return None


def main():
    t0 = time.time()
    series = series_of()
    gnames = taste.genre_names()
    gset = set(gnames)
    lines: Counter = Counter()
    by_kind: dict[str, Counter] = {}
    cards = Counter()
    per_card_new: list[int] = []
    per_card_old: list[int] = []
    conflict_old = conflict_new = 0
    users = query("SELECT id FROM app_user WHERE status = 'approved' ORDER BY id")
    for u in users:
        uid = u["id"]
        entries = query("SELECT mal_id, status, score, coalesce(finished_at::timestamptz,"
                        " updated_at) AS at FROM list_entry WHERE user_id = %s", (uid,))
        recs = query("SELECT surface, mal_id FROM recommendation WHERE user_id = %s"
                     " AND surface = ANY(%s) AND rank <= 20", (uid, list(TABS)))
        titles, tags = taste.catalogue([e["mal_id"] for e in entries] + [r["mal_id"] for r in recs])
        groups = taste.attribute_groups(entries, titles, tags, series, gnames)
        model, _ = load_latest(uid)
        if not isinstance(model, HybridScorer):
            continue
        for r in recs:
            m = r["mal_id"]
            b = model.breakdown(m)
            if b is None or b["sides"]["personal"] is None or m not in titles:
                continue
            cards["cards"] += 1
            above = b["shown"] > b["your_mean"]
            pers = b["sides"]["personal"]
            old = [(d["label"], 1) for d in pers["up"]] + [(d["label"], -1) for d in pers["down"]]
            per_card_old.append(len(old))
            if old and above and all(s < 0 for _, s in old):
                conflict_old += 1               # a recommendation explained by minuses only
            for label, sign in old:
                a = attribute(label, gset)
                if a is None:
                    verdict = "unmapped"
                else:
                    g = groups.get(a[0], {}).get(a[1])
                    if g is None or g["lift"] == 0:
                        verdict = "weak"
                    elif (g["lift"] > 0) != (sign > 0):
                        verdict = "contradicted"
                    else:
                        verdict = "backed" if g["clear"] else "weak"
                lines[verdict] += 1
                by_kind.setdefault(a[0] if a else "-", Counter())[verdict] += 1
            ev = taste.evidence(titles[m], tags.get(m, []), groups, gnames)
            per_card_new.append(len(ev))
            if ev and above and all(x["lift"] < 0 for x in ev):
                conflict_new += 1
        print(f"  user done [{time.time() - t0:.0f}s]", flush=True)

    tot = sum(lines.values())
    print(f"\n== personal lines today: {tot} on {cards['cards']} cards "
          f"({np.mean(per_card_old):.1f} per card)")
    for k in ("backed", "weak", "contradicted", "unmapped"):
        print(f"   {k:<13}{lines[k]:6d}  {lines[k] / max(tot, 1):6.1%}")
    print("   by attribute:  " + "; ".join(
        f"{kind} " + ", ".join(f"{v} {c}" for v, c in cnt.most_common())
        for kind, cnt in sorted(by_kind.items())))
    print(f"   cards above the user's mean explained by minuses only: {conflict_old}")
    a = np.array(per_card_new)
    print(f"\n== evidence lines instead: {a.mean():.2f} per card; "
          f"none {np.mean(a == 0):.1%}, one {np.mean(a == 1):.1%}, two+ {np.mean(a >= 2):.1%}")
    print(f"   cards above the user's mean explained by minuses only: {conflict_new}")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
