"""Age-fair popularity for the Hidden Gems pool (malrec.recsys.popularity)."""
from __future__ import annotations

import datetime as dt

from malrec.recsys import popularity as pop

TODAY = dt.date(2026, 10, 20)
CURVE = ((0.0, 0.0), (3.0, 0.35), (12.0, 0.55), (36.0, 0.75), (121.0, 1.0))


def test_young_titles_are_scaled_old_ones_keep_their_count(monkeypatch):
    monkeypatch.setattr(pop, "GROWTH", CURVE)
    a_year = TODAY - dt.timedelta(days=round(12 * 30.44))
    assert abs(pop.grown(10_000, a_year, TODAY) / (10_000 / 0.55) - 1) < 0.01
    assert pop.grown(10_000, dt.date(2005, 4, 1), TODAY) == 10_000
    assert pop.grown(10_000, None, TODAY) == 10_000                 # no start date: as counted
    assert pop.grown(10_000, TODAY, TODAY) == 10_000 / pop.SHARE_FLOOR   # capped scale-up
    assert pop.grown(None, TODAY, TODAY) == 0


def test_a_young_title_on_course_for_many_members_is_no_gem(monkeypatch):
    monkeypatch.setattr(pop, "GROWTH", CURVE)
    cuts = pop.bars([100_000 - m for m in range(1, 2500)])    # #1,200: 98,800; #2,000: 98,000
    young = {"mal_popularity": 5000, "members": 30_000, "start_date": TODAY - dt.timedelta(days=60)}
    assert pop.is_gem(young, "rank", cuts, TODAY)                  # raw: a "gem"
    assert not pop.is_gem(young, "grown", cuts, TODAY)             # on course for 150,000
    quiet = {"mal_popularity": 1300, "members": 98_000, "start_date": dt.date(2010, 1, 1)}
    known = {"mal_popularity": 1100, "members": 99_000, "start_date": dt.date(2010, 1, 1)}
    assert pop.is_gem(quiet, "grown", cuts, TODAY) and pop.is_gem(quiet, "rank", cuts, TODAY)
    assert not pop.is_gem(known, "grown", cuts, TODAY)


def test_scaled_new_titles_push_no_old_one_into_the_pool(monkeypatch):
    monkeypatch.setattr(pop, "GROWTH", CURVE)
    members = [300_000 - 100 * k for k in range(2500)]
    cuts = pop.bars(members + [90_000] * 500)                       # 500 new titles, scaled up
    old = {"mal_popularity": 1150, "members": members[1149], "start_date": dt.date(2012, 1, 1)}
    assert not pop.is_gem(old, "grown", cuts, TODAY)


def test_under_a_year_old_a_title_has_to_be_less_known_to_count(monkeypatch):
    monkeypatch.setattr(pop, "GROWTH", CURVE)
    cuts = (200_000.0, 100_000.0)
    old = {"members": 150_000, "start_date": dt.date(2010, 1, 1)}
    young = {"members": 70_000, "start_date": TODAY - dt.timedelta(days=200)}   # on course: ~127k
    quiet = {"members": 40_000, "start_date": TODAY - dt.timedelta(days=200)}
    assert pop.is_gem(old, "grown", cuts, TODAY)
    assert not pop.is_gem(young, "grown", cuts, TODAY)
    assert pop.is_gem(quiet, "grown", cuts, TODAY)
