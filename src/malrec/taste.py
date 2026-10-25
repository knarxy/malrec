"""The profile page: what a user's list says about their taste.

Every section is a headline sentence (a key plus its arguments; the app words
it) over the numbers that back it. Only what the list can support is said:
each series counts once however many seasons were rated (recsys/series.py),
and a genre, theme, studio or era is named only when enough different series
back it and its difference from the user's average is clear - at least
`Z` standard errors. With ~100 rated series only a handful pass, which is the
point: "you rate Psychological higher" should still hold after the next
twenty ratings. The thresholds were set on a few thousand sampled lists of
every size (experiments/exp_profile.py).

`breakdown` is pure (entries plus catalogue lookups in, a dict out), so the
experiment and the API run the same code; `for_user` loads one app user.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from . import rank

MIN_RATED_SERIES = 15       # fewer: the taste sections ask for more ratings
MIN_SERIES = 5              # a genre / studio / era needs this many series
Z = 2.0                     # and a difference this many standard errors wide
THEME_MIN_SERIES = 8        # themes: hundreds of tags are tested, so stricter
THEME_Z = 2.5
SD_FLOOR = 0.5              # a group whose series all got one score is not certain
# The evidence lines on cards (attribute_groups) test every genre, theme,
# studio, format, era, length and source of a list - 150 a list on average,
# 200 on a long one - and kept every one past the z bar above: with the
# series' scores shuffled, 7 a list still passed, a fifth of all claims and
# half of them on lists under 50 rated series (audit 2026-10-23, P1-12;
# experiments/exp_evidence_fdr.py, 2,000 sampled lists). Benjamini-Hochberg
# at this rate over all of a list's tests, on p from Student's t, leaves 0.6
# by chance (2 % of the claims; 8 % at 50-149 rated series, 27 % below 50)
# and keeps as many real ones; cards with a line 86 % -> 77 %.
EVIDENCE_FDR = 0.2
MIN_COMPARED = 10           # titles with a MAL score before the scale is compared
SCALE_SLIGHT, SCALE_STRONG = 0.25, 0.75
POPULAR = 20_000            # disagreements: titles at least this many rated on MAL
DISAGREE = 1.5              # and at least this far from MAL's mean
SHOW = 3                    # titles per side of the disagreements
ONE_SCORE = 0.6             # share of one score that makes the scale "one score"
LISTED = 5                  # "most watched" genres shown
THEME_CATEGORIES = ("Theme", "Setting", "Cast-Traits")   # not demographic, cast mix, technical
ERA_SPAN = 5
EPISODE_SECONDS = 1440      # when MAL has no duration
# Labels a profile page does not announce as someone's taste: MAL's prize
# badge and adult genres, AniList tags about sex, gender and orientation, and
# themes no sentence should say someone "rates high" (found on 2,800 sampled
# lists, experiments/exp_profile.py).
NOT_TASTE = frozenset(x.lower() for x in (
    "Award Winning", "Hentai", "Erotica", "Boys Love", "Girls Love",
    "Nudity", "Heterosexual", "Bisexual", "Transgender", "Agender", "Aromantic", "Asexual",
    "LGBTQ+ Themes", "Femboy", "Crossdressing", "Gender Bending", "Yuri", "Boys' Love",
    "Teens' Love", "Incest", "Inseki", "Polyamorous", "Cheating", "Succubus", "Psychosexual",
    "Ero Guro", "Suicide", "Torture", "Cannibalism", "Slavery", "Bullying", "Body Image",
    "Drugs", "Terrorism", "Religion", "Cult", "Disability", "Homeless"))
SHOW_UP, SHOW_DOWN = 6, 4   # clear groups listed per side
# Every headline a section can get; the app has a sentence for each
# (i18n.tsx, taste.<key>), which tests/test_taste.py checks.
HEADLINES = (
    "scale.none", "scale.few", "scale.one_score", "scale.much_generous", "scale.generous",
    "scale.aligned", "scale.strict", "scale.much_strict",
    "genres.few", "genres.none", "genres.up_one", "genres.up_two", "genres.up_down",
    "genres.down", "genres.watched_lower", "genres.up_watched_lower",
    "themes.up_one", "themes.up_two", "themes.up_down", "themes.down",
    "disagree.both", "disagree.above", "disagree.below", "disagree.none",
    "eras.favourite", "eras.lower", "eras.none", "eras.ages_only",
    "studios.up_one", "studios.up_two", "studios.down",
    "pred.close", "pred.ok", "pred.hard")


def era_of(year: int) -> int:
    return (year // ERA_SPAN) * ERA_SPAN


def _groups(by_series: dict[int, float], members: dict[str, set[int]], mu: float,
            min_series: int, z_min: float, with_p: bool = False) -> list[dict]:
    """Groups (a genre, tag, studio, era) with their series-level mean, lift
    over `mu` and its z; `clear` when it passes min_series and z_min.
    `with_p`: also the two-sided p of that z on Student's t with n - 1
    degrees of freedom (a group has a handful of series: the normal's tails
    are far too thin for it)."""
    from scipy.stats import t as student_t
    out = []
    for k, ss in members.items():
        vals = [by_series[s] for s in ss if s in by_series]
        n = len(vals)
        if n < min_series:
            continue
        m = sum(vals) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1)) if n > 1 else 0.0
        z = (m - mu) / (max(sd, SD_FLOOR) / math.sqrt(n))
        g = {"name": k, "series": n, "mean": round(m, 2), "lift": round(m - mu, 2),
             "z": round(z, 1), "clear": abs(z) >= z_min}
        if with_p:
            g["p"] = float(2 * student_t.sf(abs(z), n - 1)) if n > 1 else 1.0
        out.append(g)
    return sorted(out, key=lambda g: -g["lift"])


def benjamini_hochberg(p: list[float], q: float) -> list[bool]:
    """Which of these p-values Benjamini-Hochberg's step-up procedure rejects
    at false discovery rate q."""
    order = sorted(range(len(p)), key=lambda k: p[k])
    last = max((i for i, k in enumerate(order) if p[k] <= q * (i + 1) / len(p)), default=-1)
    keep = set(order[:last + 1])
    return [k in keep for k in range(len(p))]


def _pick(clear: list[dict]) -> tuple[list[dict], list[dict]]:
    """Clear groups above and below the average, best evidenced first: a
    headline names the group it is surest of, not the niche one with five
    series and the largest lift."""
    up = sorted((g for g in clear if g["lift"] > 0), key=lambda g: -g["z"])
    down = sorted((g for g in clear if g["lift"] < 0), key=lambda g: g["z"])
    return up, down


def _shown(up: list[dict], down: list[dict]) -> list[dict]:
    """What a section lists: the best evidenced of each side, by lift."""
    return (sorted(up[:SHOW_UP], key=lambda g: -g["lift"])
            + sorted(down[:SHOW_DOWN], key=lambda g: -g["lift"]))


def _hours(e: dict, t: dict) -> float:
    eps = e.get("episodes_watched")
    if eps is None:                     # sampled lists carry no episode count
        eps = (t.get("num_episodes") or 0) if e["status"] == "completed" else 0
    return eps * (t.get("avg_episode_seconds") or EPISODE_SECONDS) / 3600


def breakdown(entries: list[dict], titles: dict[int, dict], tags: dict[int, list[str]],
              series: dict[int, int], genre_names: Iterable[str] = (),
              today: dt.date | None = None) -> dict:
    """`entries`: mal_id, status, score, at (finished, else last updated) and
    optionally episodes_watched. `titles`: mal_id -> the anime row (title,
    title_en, media_type, num_episodes, avg_episode_seconds, season_year,
    mal_mean, mal_num_scoring_users, mal_genres, mal_studios, picture_medium).
    `tags`: mal_id -> theme tags. `series`: mal_id -> series id."""
    today = today or dt.datetime.now(dt.UTC).date()
    genre_names = {g.lower() for g in genre_names}
    entries = [e for e in entries if e["mal_id"] in titles]
    sid = {e["mal_id"]: series.get(e["mal_id"], e["mal_id"]) for e in entries}
    rated = [e for e in entries if (e.get("score") or 0) > 0]
    started = [e for e in entries if e["status"] != "plan_to_watch"]
    status = Counter(e["status"] for e in entries)
    hours = sum(_hours(e, titles[e["mal_id"]]) for e in started)
    out: dict = {"stats": {
        "entries": len(entries), "series": len(set(sid.values())), "rated": len(rated),
        "mean": round(sum(e["score"] for e in rated) / len(rated), 2) if rated else None,
        "completed": status["completed"], "watching": status["watching"],
        "dropped": status["dropped"], "on_hold": status["on_hold"],
        "plan_to_watch": status["plan_to_watch"], "hours": round(hours)}}

    # ---- the scale: the user's scores against MAL's on the same titles
    hist = [sum(1 for e in rated if e["score"] == s) for s in range(1, 11)]
    pairs = [(e, e["score"] - titles[e["mal_id"]]["mal_mean"]) for e in rated
             if titles[e["mal_id"]].get("mal_mean")]
    scale: dict = {"hist": hist, "compared": len(pairs)}
    if not rated:
        scale["headline"] = {"key": "scale.none"}
    elif max(hist) / len(rated) >= ONE_SCORE and len(rated) >= MIN_COMPARED:
        scale["headline"] = {"key": "scale.one_score", "score": hist.index(max(hist)) + 1,
                             "share": round(max(hist) / len(rated), 2)}
    elif len(pairs) < MIN_COMPARED:
        scale["headline"] = {"key": "scale.few"}
    if pairs:
        d = sum(x for _, x in pairs) / len(pairs)
        scale |= {"diff": round(d, 2),
                  "above": round(sum(1 for _, x in pairs if x >= 0.5) / len(pairs), 2),
                  "below": round(sum(1 for _, x in pairs if x <= -0.5) / len(pairs), 2)}
        if "headline" not in scale:
            key = ("much_generous" if d >= SCALE_STRONG else "generous" if d >= SCALE_SLIGHT
                   else "much_strict" if d <= -SCALE_STRONG else "strict" if d <= -SCALE_SLIGHT
                   else "aligned")
            scale["headline"] = {"key": f"scale.{key}", "diff": round(d, 2)}
    out["scale"] = scale

    # ---- series-level scores: each series is the mean of its rated entries
    per: dict[int, list[int]] = defaultdict(list)
    for e in rated:
        per[sid[e["mal_id"]]].append(e["score"])
    by_series = {s: sum(v) / len(v) for s, v in per.items()}
    mu = sum(by_series.values()) / len(by_series) if by_series else 0.0
    enough = len(by_series) >= MIN_RATED_SERIES
    out["taste"] = {"rated_series": len(by_series), "series_mean": round(mu, 2),
                    "enough": enough, "min_series": MIN_RATED_SERIES}

    def members(key) -> dict[str, set[int]]:
        g: dict[str, set[int]] = defaultdict(set)
        for e in rated:
            for k in key(e["mal_id"]):
                g[k].add(sid[e["mal_id"]])
        return g

    # ---- genres
    listed = Counter(g for e in started for g in (titles[e["mal_id"]].get("mal_genres") or [])
                     if g.lower() not in NOT_TASTE)
    top_listed = [{"name": g, "n": n} for g, n in listed.most_common(LISTED)]
    genres: dict = {"listed": top_listed}
    if not enough:
        genres["headline"] = {"key": "genres.few", "missing": MIN_RATED_SERIES - len(by_series)}
        genres["clear"] = []
    else:
        gs = _groups(by_series, members(lambda m: [g for g in titles[m].get("mal_genres") or []
                                                   if g.lower() not in NOT_TASTE]),
                     mu, MIN_SERIES, Z)
        up, down = _pick([g for g in gs if g["clear"]])
        genres["clear"] = _shown(up, down)
        heavy = {g["name"] for g in top_listed}
        lower_watched = [g for g in down if g["name"] in heavy]
        if lower_watched and up:
            genres["headline"] = {"key": "genres.up_watched_lower", "up": up[0]["name"],
                                  "down": lower_watched[0]["name"]}
        elif lower_watched:
            genres["headline"] = {"key": "genres.watched_lower", "down": lower_watched[0]["name"]}
        elif up and down:
            genres["headline"] = {"key": "genres.up_down", "up": up[0]["name"],
                                  "down": down[0]["name"]}
        elif len(up) >= 2:
            genres["headline"] = {"key": "genres.up_two", "a": up[0]["name"], "b": up[1]["name"]}
        elif up:
            genres["headline"] = {"key": "genres.up_one", "up": up[0]["name"]}
        elif down:
            genres["headline"] = {"key": "genres.down", "down": down[0]["name"]}
        else:
            genres["headline"] = {"key": "genres.none"}
    out["genres"] = genres

    # ---- themes (AniList tags)
    themes: dict = {"clear": []}
    if enough:
        ts = _groups(by_series, members(lambda m: [t for t in tags.get(m, ())
                                                   if t.lower() not in genre_names
                                                   and t.lower() not in NOT_TASTE]),
                     mu, THEME_MIN_SERIES, THEME_Z)
        up, down = _pick([t for t in ts if t["clear"]])
        themes["clear"] = _shown(up, down)
        if up and down:
            themes["headline"] = {"key": "themes.up_down", "up": up[0]["name"],
                                  "down": down[0]["name"]}
        elif len(up) >= 2:
            themes["headline"] = {"key": "themes.up_two", "a": up[0]["name"], "b": up[1]["name"]}
        elif up:
            themes["headline"] = {"key": "themes.up_one", "up": up[0]["name"]}
        elif down:
            themes["headline"] = {"key": "themes.down", "down": down[0]["name"]}
    out["themes"] = themes              # no headline: the page leaves the card out

    # ---- disagreements with MAL on widely rated titles
    def card(e, d):
        t = titles[e["mal_id"]]
        return {"mal_id": e["mal_id"], "title": t.get("title_en") or t.get("title"),
                "picture": t.get("picture_medium"), "you": e["score"],
                "mal": round(t["mal_mean"], 2), "diff": round(d, 2)}
    wide = [(e, d) for e, d in pairs
            if (titles[e["mal_id"]].get("mal_num_scoring_users") or 0) >= POPULAR]
    above = [card(e, d) for e, d in sorted(wide, key=lambda p: -p[1]) if d >= DISAGREE][:SHOW]
    below = [card(e, d) for e, d in sorted(wide, key=lambda p: p[1]) if d <= -DISAGREE][:SHOW]
    dis: dict = {"above": above, "below": below, "compared": len(wide)}
    if above and below:
        dis["headline"] = {"key": "disagree.both"}
    elif above:
        dis["headline"] = {"key": "disagree.above"}
    elif below:
        dis["headline"] = {"key": "disagree.below"}
    elif len(wide) >= MIN_COMPARED:
        dis["headline"] = {"key": "disagree.none", "n": len(wide)}
    out["disagree"] = dis

    # ---- eras: release years, and how old titles were when watched
    year = {m: t.get("season_year") for m, t in titles.items()}
    eras: dict = {"rows": []}
    if enough:
        es = _groups(by_series, members(lambda m: [era_of(y)] if (y := year.get(m)) else []),
                     mu, MIN_SERIES, Z)
        def to(g):                      # the running span ends this year
            return min(g["name"] + ERA_SPAN - 1, today.year)
        eras["rows"] = sorted(({**g, "from": g["name"], "to": to(g)} for g in es),
                              key=lambda g: g["from"])
        up, down = _pick([g for g in es if g["clear"]])
        if up:
            eras["headline"] = {"key": "eras.favourite", "from": up[0]["name"], "to": to(up[0])}
        elif down:
            eras["headline"] = {"key": "eras.lower", "from": down[0]["name"], "to": to(down[0])}
        elif eras["rows"]:
            eras["headline"] = {"key": "eras.none"}
    mix = rank.age_mix(entries, year.get, today.year)
    if mix is not None:
        share = [float(x) for x in mix.share]
        eras["ages"] = [round(x, 3) for x in share]
        eras["age_edges"] = list(rank.AGE_EDGES)
        eras["new"] = round(share[0], 2)                  # under two years old
        eras["old"] = round(sum(share[3:]), 2)            # ten years or older
        eras.setdefault("headline", {"key": "eras.ages_only", "new": eras["new"],
                                     "old": eras["old"]})
    out["eras"] = eras

    # ---- studios (the first listed, the main one)
    studios: dict = {"clear": []}
    if enough:
        ss = _groups(by_series, members(lambda m: (titles[m].get("mal_studios") or [])[:1]),
                     mu, MIN_SERIES, Z)
        up, down = _pick([g for g in ss if g["clear"]])
        studios["clear"] = _shown(up, down)
        if len(up) >= 2:
            studios["headline"] = {"key": "studios.up_two", "a": up[0]["name"], "b": up[1]["name"]}
        elif up:
            studios["headline"] = {"key": "studios.up_one", "up": up[0]["name"]}
        elif down:
            studios["headline"] = {"key": "studios.down", "down": down[0]["name"]}
    out["studios"] = studios
    return out


# ---------------------------------------------- evidence for explanations --

ATTRIBUTES = ("genre", "theme", "studio", "format", "era", "length", "source")


def length_of(t: dict) -> str:
    """The episode bucket the model uses (features._cat_tokens)."""
    e = t.get("num_episodes") or 0
    return "1-13" if e <= 13 else "14-26" if e <= 26 else "27-60" if e <= 60 else "60+"


def attribute_values(t: dict, tags: Iterable[str], genre_names: set[str]) -> dict[str, list[str]]:
    """A title's value(s) of every attribute; `genre_names` lower-cased."""
    return {
        "genre": [g for g in t.get("mal_genres") or [] if g.lower() not in NOT_TASTE],
        "theme": [x for x in tags if x.lower() not in genre_names and x.lower() not in NOT_TASTE],
        "studio": t.get("mal_studios") or [],
        "format": [t["media_type"]] if t.get("media_type") else [],
        "era": [str(era_of(y))] if (y := t.get("season_year")) else [],
        "length": [length_of(t)],
        "source": [t["source"]] if t.get("source") else [],
    }


def attribute_groups(entries: list[dict], titles: dict[int, dict], tags: dict[int, list[str]],
                     series: dict[int, int], genre_names: Iterable[str] = ()) -> dict:
    """The user's series-level mean per value of every attribute a title has
    (its genres, themes, studios, format, era, length, source), with the
    profile page's rule for `clear` - and, since a long list runs a couple
    of hundred of these tests, Benjamini-Hochberg over all of them at
    EVIDENCE_FDR. Every group with two or more series is kept, so an
    explanation can also be checked against a weak one."""
    genre_names = {g.lower() for g in genre_names}
    entries = [e for e in entries if e["mal_id"] in titles]
    sid = {e["mal_id"]: series.get(e["mal_id"], e["mal_id"]) for e in entries}
    rated = [e for e in entries if (e.get("score") or 0) > 0]
    per: dict[int, list[int]] = defaultdict(list)
    for e in rated:
        per[sid[e["mal_id"]]].append(e["score"])
    by_series = {s: sum(v) / len(v) for s, v in per.items()}
    mu = sum(by_series.values()) / len(by_series) if by_series else 0.0
    vals = {e["mal_id"]: attribute_values(titles[e["mal_id"]], tags.get(e["mal_id"], ()),
                                          genre_names) for e in rated}
    out: dict = {"mu": round(mu, 3), "rated_series": len(by_series)}
    tested: list[dict] = []          # every group that could be called clear
    for kind in ATTRIBUTES:
        min_k, z = ((THEME_MIN_SERIES, THEME_Z) if kind == "theme" else (MIN_SERIES, Z))
        g: dict[str, set[int]] = defaultdict(set)
        for e in rated:
            for k in vals[e["mal_id"]][kind]:
                g[k].add(sid[e["mal_id"]])
        groups = _groups(by_series, g, mu, 2, z, with_p=True)
        for x in groups:
            testable = x["series"] >= min_k and len(by_series) >= MIN_RATED_SERIES
            x["clear"] = x["clear"] and testable
            if testable:
                tested.append(x)
        out[kind] = {x["name"]: x for x in groups}
    if tested:
        for x, ok in zip(tested, benjamini_hochberg([x["p"] for x in tested], EVIDENCE_FDR)):
            x["clear"] = x["clear"] and ok
    return out


def evidence(title: dict, tags: list[str], groups: dict, genre_names: Iterable[str] = ()
             ) -> list[dict]:
    """What the user's own ratings clearly say about this title's attributes,
    best evidenced first: {kind, name, lift, series, z}."""
    out = []
    for kind, names in attribute_values(title, tags, {g.lower() for g in genre_names}).items():
        for n in names:
            g = groups.get(kind, {}).get(n)
            if g and g["clear"]:
                out.append({"kind": kind, "name": n, "lift": g["lift"], "series": g["series"],
                            "z": g["z"]})
    return sorted(out, key=lambda x: -abs(x["z"]))


EVIDENCE_SHOWN = 4             # lines in the "Why this?" panel
GROUPS_TTL = 600               # s a user's attribute groups are reused for
_groups_cache: dict[int, tuple[float, dict]] = {}


def evidence_label(kind: str, name: str) -> str:
    """The panel's label for an attribute value, in the drivers' wording
    (the app translates it like those)."""
    if kind == "format":
        return rank._TYPE_LABELS.get(name, name)
    if kind == "source":
        return rank._SOURCE_LABELS.get(name, name)
    if kind == "era":
        return f"{name}–{int(name) + ERA_SPAN - 1}"
    if kind == "length":
        return f"{name} episodes"
    return name


def _user_groups(user_id: int, fresh: bool = False) -> dict:
    """The user's attribute groups, reused for GROUPS_TTL seconds (a list
    build recomputes them: it follows a sync)."""
    import time

    from .db import query
    from .recsys.series import series_of
    hit = _groups_cache.get(user_id)
    if fresh or hit is None or time.time() - hit[0] > GROUPS_TTL:
        entries = query(ENTRY_SQL, (user_id,))
        titles, tags = catalogue(e["mal_id"] for e in entries)
        hit = (time.time(), attribute_groups(entries, titles, tags, series_of(), genre_names()))
        _groups_cache[user_id] = hit
    return hit[1]


def evidence_many(user_id: int, ids: Iterable[int], fresh: bool = False) -> dict[int, list[dict]]:
    """mal_id -> what the user's own ratings clearly say about that title's
    attributes, best evidenced first (at most EVIDENCE_SHOWN)."""
    groups = _user_groups(user_id, fresh)
    titles, tags = catalogue(ids)
    names = genre_names()
    return {m: [{"kind": x["kind"], "name": x["name"], "label": evidence_label(x["kind"], x["name"]),
                 "lift": round(x["lift"], 2), "series": x["series"]}
                for x in evidence(t, tags.get(m, []), groups, names)[:EVIDENCE_SHOWN]]
            for m, t in titles.items()}


def evidence_for(user_id: int, mal_id: int) -> list[dict]:
    """The "Why this?" panel's personal side."""
    return evidence_many(user_id, [mal_id]).get(mal_id, [])


def predictions(metrics: dict | None) -> dict | None:
    """The personal model's temporal holdout (model_run.metrics), as a
    section. A small error with the order wrong is no good prediction, so
    the wording needs both."""
    if not metrics or metrics.get("rmse") is None:
        return None
    rmse, rho = float(metrics["rmse"]), metrics.get("spearman")
    order = float(rho) if rho is not None else 0.0
    key = ("close" if rmse <= 1.0 and order >= 0.5 else
           "ok" if rmse <= 1.5 and order >= 0.3 else "hard")
    holdouts = [h.get("holdout") for h in metrics.get("per_holdout") or [] if h.get("holdout")]
    return {"headline": {"key": f"pred.{key}", "rmse": round(rmse, 2)},
            "rmse": round(rmse, 2), "spearman": round(float(rho), 2) if rho is not None else None,
            "holdout_min": min(holdouts) if holdouts else None,
            "holdout_max": max(holdouts) if holdouts else None}


# ------------------------------------------------------------------ loading --

TITLE_SQL = """SELECT mal_id, title, title_en, media_type, num_episodes, avg_episode_seconds,
                      season_year, mal_mean, mal_num_scoring_users, mal_genres, mal_studios,
                      picture_medium, source
                 FROM anime WHERE mal_id = ANY(%s)"""
TAG_SQL = """SELECT mal_id, tag, category FROM anime_tag
              WHERE rank >= %s AND mal_id = ANY(%s)"""
TAG_RANK_MIN = 60           # AniList's relevance; lower ranks are passing mentions


def theme_ok(category: str) -> bool:
    return category.startswith(THEME_CATEGORIES)


def catalogue(ids: Iterable[int]) -> tuple[dict[int, dict], dict[int, list[str]]]:
    from .db import query
    ids = list(set(ids))
    titles = {r["mal_id"]: r for r in query(TITLE_SQL, (ids,))}
    tags: dict[int, list[str]] = defaultdict(list)
    for r in query(TAG_SQL, (TAG_RANK_MIN, ids)):
        if theme_ok(r["category"]):
            tags[r["mal_id"]].append(r["tag"])
    return titles, tags


_genres: dict = {"at": 0.0, "names": []}


def genre_names() -> list[str]:
    """Every MAL genre name (tags of the same name are left to the genres);
    cached for an hour."""
    import time

    from .db import query
    if not _genres["names"] or time.time() - _genres["at"] > 3600:
        _genres["names"] = [r["g"] for r in query("SELECT DISTINCT unnest(mal_genres) g FROM anime")]
        _genres["at"] = time.time()
    return _genres["names"]


ENTRY_SQL = """SELECT mal_id, status, score, episodes_watched,
                      coalesce(finished_at::timestamptz, updated_at) AS at
                 FROM list_entry WHERE user_id = %s"""


def for_user(user_id: int) -> dict:
    from .db import query
    from .recsys.series import series_of
    entries = query(ENTRY_SQL, (user_id,))
    titles, tags = catalogue(e["mal_id"] for e in entries)
    out = breakdown(entries, titles, tags, series_of(), genre_names())
    # the stored run's metrics; loading the model itself refits it (seconds)
    run = query("SELECT metrics FROM model_run WHERE user_id = %s AND artifact IS NOT NULL"
                " ORDER BY trained_at DESC LIMIT 1", (user_id,))
    out["predictions"] = predictions(run[0]["metrics"]) if run else None
    return out
