"""The profile page's sections (malrec.taste): pure, so no database."""
from __future__ import annotations

import datetime as dt
import pathlib
import re

from malrec import taste

AT = dt.datetime(2026, 5, 1, tzinfo=dt.UTC)
TODAY = dt.date(2026, 10, 20)


def title(m, genres=(), year=2015, mean=7.5, scorers=50_000, studio="Studio A", eps=12):
    return {"mal_id": m, "title": f"t{m}", "title_en": None, "media_type": "tv",
            "num_episodes": eps, "avg_episode_seconds": 1440, "season_year": year,
            "mal_mean": mean, "mal_num_scoring_users": scorers, "mal_genres": list(genres),
            "mal_studios": [studio], "picture_medium": None}


def entry(m, score, status="completed"):
    return {"mal_id": m, "status": status, "score": score, "at": AT, "episodes_watched": None}


def run(entries, titles, tags=None, series=None):
    return taste.breakdown(entries, titles, tags or {}, series or {}, ["Drama", "Comedy"], TODAY)


def test_an_empty_or_unrated_list_says_so_without_failing():
    out = run([], {})
    assert out["stats"]["entries"] == 0 and out["scale"]["headline"]["key"] == "scale.none"
    titles = {1: title(1)}
    out = run([entry(1, 0, "plan_to_watch")], titles)
    assert out["scale"]["headline"]["key"] == "scale.none"
    assert out["genres"]["headline"] == {"key": "genres.few", "missing": taste.MIN_RATED_SERIES}
    assert "headline" not in out["themes"] and "headline" not in out["studios"]


def test_a_clear_genre_needs_five_series_and_a_clear_difference():
    titles, es = {}, []
    for m in range(1, 31):              # 30 series: Drama ones rated 9, the rest 6-8
        drama = m <= 8
        titles[m] = title(m, ["Drama"] if drama else ["Comedy"], studio=f"S{m}")
        es.append(entry(m, 9 if drama else 6 + m % 3))
    out = run(es, titles)
    clear = {g["name"]: g for g in out["genres"]["clear"]}
    assert "Drama" in clear and clear["Drama"]["lift"] > 0 and clear["Drama"]["series"] == 8
    assert out["genres"]["headline"]["key"] in ("genres.up_one", "genres.up_watched_lower",
                                                "genres.up_down")
    assert out["genres"]["headline"]["up"] == "Drama"
    # four Drama series are not enough, however high they are rated
    few = [e for e in es if e["mal_id"] > 4]
    assert "Drama" not in {g["name"] for g in run(few, titles)["genres"]["clear"]}


def test_seasons_of_one_show_count_once():
    titles = {m: title(m, ["Drama"] if m <= 10 else ["Comedy"]) for m in range(1, 31)}
    es = [entry(m, 10 if m <= 10 else 7) for m in range(1, 31)]
    series = {m: 1 for m in range(1, 11)}         # ten seasons of one Drama
    out = run(es, titles, series=series)
    assert out["taste"]["rated_series"] == 21
    assert "Drama" not in {g["name"] for g in out["genres"]["clear"]}


def test_labels_that_are_not_a_taste_stay_out():
    titles = {m: title(m, ["Award Winning" if m <= 10 else "Comedy"]) for m in range(1, 31)}
    tags = {m: ["Suicide"] if m <= 12 else ["Travel"] for m in range(1, 31)}
    es = [entry(m, 10 if m <= 10 else 6 + m % 2) for m in range(1, 31)]
    out = run(es, titles, tags)
    names = {g["name"] for g in out["genres"]["clear"]} | {g["name"] for g in out["themes"]["clear"]}
    names |= {g["name"] for g in out["genres"]["listed"]}
    assert not names & {"Award Winning", "Suicide"}


def test_one_score_for_everything_and_the_running_era():
    titles = {m: title(m, year=2025 if m <= 10 else 2010) for m in range(1, 21)}
    es = [entry(m, 10) for m in range(1, 21)]
    out = run(es, titles)
    assert out["scale"]["headline"] == {"key": "scale.one_score", "score": 10, "share": 1.0}
    assert all(r["to"] <= TODAY.year for r in out["eras"]["rows"])
    assert out["eras"]["headline"]["key"] in ("eras.none", "eras.favourite", "eras.lower")


def test_disagreements_only_on_widely_rated_titles():
    titles = {1: title(1, mean=8.5), 2: title(2, mean=6.0), 3: title(3, mean=8.5, scorers=500)}
    titles |= {m: title(m, mean=7.0) for m in range(4, 14)}
    es = [entry(1, 5), entry(2, 9), entry(3, 3)] + [entry(m, 7) for m in range(4, 14)]
    out = run(es, titles)
    d = out["disagree"]
    assert [c["mal_id"] for c in d["above"]] == [2] and [c["mal_id"] for c in d["below"]] == [1]
    assert d["headline"]["key"] == "disagree.both"


def test_predictions_wording_follows_the_error():
    assert taste.predictions(None) is None
    assert taste.predictions({"rmse": 0.85, "spearman": 0.8})["headline"]["key"] == "pred.close"
    assert taste.predictions({"rmse": 1.3, "spearman": 0.5})["headline"]["key"] == "pred.ok"
    assert taste.predictions({"rmse": 1.9, "spearman": 0.7})["headline"]["key"] == "pred.hard"
    # a small error with the order wrong is not a good prediction
    assert taste.predictions({"rmse": 0.9, "spearman": 0.05})["headline"]["key"] == "pred.hard"


def test_every_headline_has_an_english_and_a_german_sentence():
    src = (pathlib.Path(__file__).parents[1] / "app/src/i18n.tsx").read_text()
    de, en = src.split("const en: Record<string, string> = {")
    for key in taste.HEADLINES:
        for lang, text in (("de", de), ("en", en)):
            assert re.search(rf"'taste\.{re.escape(key)}':", text), (lang, key)


def test_only_known_headlines_come_out():
    titles = {m: title(m, ["Drama"] if m % 3 == 0 else ["Comedy"], year=1990 + m % 30,
                       mean=6 + m % 4, studio=f"S{m % 4}") for m in range(1, 80)}
    tags = {m: ["Travel"] if m % 2 else ["Magic"] for m in range(1, 80)}
    for scores in ((lambda m: 10), (lambda m: 5 + m % 6), (lambda m: 9 if m % 3 == 0 else 6)):
        out = run([entry(m, scores(m)) for m in range(1, 80)], titles, tags)
        for s in ("scale", "genres", "themes", "disagree", "eras", "studios"):
            h = out[s].get("headline")
            assert h is None or h["key"] in taste.HEADLINES, h


def test_evidence_names_only_attributes_the_ratings_clearly_back():
    titles, es = {}, []
    for m in range(1, 31):              # Drama rated 9, Comedy 6-8; one studio each
        titles[m] = title(m, ["Drama"] if m <= 8 else ["Comedy"], studio=f"S{m}")
        es.append(entry(m, 9 if m <= 8 else 6 + m % 3))
    groups = taste.attribute_groups(es, titles, {}, {}, ["Drama", "Comedy"])
    assert groups["genre"]["Drama"]["clear"] and groups["genre"]["Drama"]["lift"] > 0
    assert not groups["genre"]["Comedy"]["clear"] or groups["genre"]["Comedy"]["lift"] < 0
    new = title(99, ["Drama"], studio="S1")
    ev = taste.evidence(new, [], groups, ["Drama", "Comedy"])
    assert [(x["kind"], x["name"]) for x in ev][:1] == [("genre", "Drama")]
    assert all(x["kind"] != "studio" for x in ev)      # one series of S1: no evidence
    # with too few rated series nothing counts as clear
    few = taste.attribute_groups(es[:10], titles, {}, {}, ["Drama", "Comedy"])
    assert taste.evidence(new, [], few, ["Drama", "Comedy"]) == []


def test_evidence_keeps_what_survives_the_lists_many_tests(monkeypatch):
    """P1-12: a studio five series past the z bar (2.1, p .10 on t with 4
    degrees of freedom) among a list's other tests is no longer a reason on
    cards; the genre rated 9 across eight series still is."""
    titles, es = {}, []
    for m in range(1, 9):
        titles[m] = title(m, ["Drama"], studio=f"D{m}")
        es.append(entry(m, 9))
    for k in range(40):
        titles[100 + k] = title(100 + k, ["Comedy"], studio=f"S{k % 8}")
        es.append(entry(100 + k, 6 + k % 3))
    for k, s in enumerate([9, 8, 8, 7, 9]):
        titles[200 + k] = title(200 + k, ["Comedy"], studio="SX")
        es.append(entry(200 + k, s))

    def clear():
        g = taste.attribute_groups(es, titles, {}, {}, ["Drama", "Comedy"])
        return g["genre"]["Drama"]["clear"], g["studio"]["SX"]["clear"]
    monkeypatch.setattr(taste, "EVIDENCE_FDR", 1.0)            # no correction: as before
    assert clear() == (True, True)
    monkeypatch.setattr(taste, "EVIDENCE_FDR", 0.2)
    assert clear() == (True, False)


def test_profile_cards_survive_the_pages_other_tests(monkeypatch):
    """The profile's studio card leaves out the same borderline studio the
    evidence lines do; the clear genre stays, and no p reaches the page."""
    titles, es = {}, []
    for m in range(1, 9):
        titles[m] = title(m, ["Drama"], studio=f"D{m}")
        es.append(entry(m, 9))
    for k in range(40):
        titles[100 + k] = title(100 + k, ["Comedy"], studio=f"S{k % 8}")
        es.append(entry(100 + k, 6 + k % 3))
    for k, s in enumerate([9, 8, 8, 7, 9]):
        titles[200 + k] = title(200 + k, ["Comedy"], studio="SX")
        es.append(entry(200 + k, s))
    monkeypatch.setattr(taste, "EVIDENCE_FDR", 1.0)            # no correction: as before
    out = run(es, titles)
    assert [g["name"] for g in out["studios"]["clear"]] == ["SX"]
    monkeypatch.setattr(taste, "EVIDENCE_FDR", 0.2)
    out = run(es, titles)
    assert out["studios"]["clear"] == [] and "headline" not in out["studios"]
    assert "Drama" in [g["name"] for g in out["genres"]["clear"]]
    assert all("p" not in g for g in out["genres"]["clear"] + out["eras"]["rows"])


def test_benjamini_hochberg_steps_up():
    assert taste.benjamini_hochberg([0.001, 0.04, 0.03, 0.5], 0.2) == [True, True, True, False]
    assert taste.benjamini_hochberg([0.3, 0.6], 0.2) == [False, False]
    assert taste.benjamini_hochberg([], 0.2) == []


def test_evidence_labels_read_like_the_drivers():
    assert taste.evidence_label("era", "2015") == "2015–2019"
    assert taste.evidence_label("format", "tv") == "a TV series"
    assert taste.evidence_label("length", "14-26") == "14-26 episodes"
    assert taste.evidence_label("theme", "Yandere") == "Yandere"
