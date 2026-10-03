"""Tests that run against the live local database.

They are read-only apart from a scratch user, so they can run against a
populated dev database without disturbing it. Anything requiring network
access is skipped; the point here is the SQL logic and the ranking rules,
which are where the subtle bugs live.
"""
from __future__ import annotations

import numpy as np
import pytest

from malrec.config import settings
from malrec.db import one, query, scalar
from malrec.ingest.store import upsert_anime
from malrec.rank import Candidate, novelty, rerank


@pytest.fixture(scope="session")
def db():
    try:
        scalar("SELECT 1")
    except Exception as e:                      # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    return True


@pytest.fixture(scope="session")
def user_id(db):
    row = one("SELECT id FROM app_user WHERE mal_username = %s", (settings().malrec_user,))
    if not row:
        pytest.skip("no synced user; run `malrec sync list` first")
    return row["id"]


# --------------------------------------------------------------- ingest --

def test_stub_upsert_does_not_clobber_full_row(db):
    """A recommendation stub carries only id+title; it must never erase the
    metadata of an anime already in the catalogue."""
    mid = 999_999_002
    upsert_anime([{"mal_id": mid, "title": "Full", "mal_genres": ["Action"],
                   "mal_studios": ["S"], "mal_mean": 8.5, "num_episodes": 12}])
    upsert_anime([{"mal_id": mid, "title": "Full"}])
    r = one("SELECT mal_genres, mal_mean, num_episodes FROM anime WHERE mal_id=%s", (mid,))
    try:
        assert r["mal_genres"] == ["Action"]
        assert r["mal_mean"] == pytest.approx(8.5)
        assert r["num_episodes"] == 12
    finally:
        query("DELETE FROM anime WHERE mal_id=%s", (mid,))


def test_rec_edges_are_symmetric(db):
    """Candidate-side affinity only works because edges are stored both ways."""
    row = one("SELECT src, dst, provider FROM rec_edge LIMIT 1")
    if not row:
        pytest.skip("no edges ingested yet")
    back = one("SELECT 1 AS ok FROM rec_edge WHERE src=%s AND dst=%s AND provider=%s",
               (row["dst"], row["src"], row["provider"]))
    assert back, "reverse edge missing - affinity lookups would silently miss"


# ------------------------------------------------------------ sql logic --

def test_recency_weight_decays_and_respects_floor(db):
    fresh = scalar("SELECT recency_weight(NULL, now(), 0.5::real, 0.05::real)")
    old = scalar("SELECT recency_weight(NULL, now() - interval '8 years', "
                 "0.5::real, 0.05::real)")
    off = scalar("SELECT recency_weight(NULL, now() - interval '8 years', "
                 "NULL::real, 0.05::real)")
    assert fresh == pytest.approx(1.0, abs=1e-3)
    assert 0.05 <= old < 0.1, "an 8-year-old rating should sit near the floor"
    assert off == pytest.approx(1.0), "a NULL half-life must disable decay"


def test_prerequisites_walk_the_whole_chain(db):
    """A third season must report both earlier seasons, not just its parent."""
    row = one(
        """
        SELECT r2.src AS s3, r2.dst AS s2, r1.dst AS s1
          FROM relation r2
          JOIN relation r1 ON r1.src = r2.dst AND r1.relation_type = 'prequel'
         WHERE r2.relation_type = 'prequel' AND r1.dst <> r2.src
         LIMIT 1
        """
    )
    if not row:
        pytest.skip("no two-deep prequel chain ingested yet")
    got = {r["mal_id"] for r in query("SELECT mal_id FROM prerequisites(%s)", (row["s3"],))}
    assert row["s2"] in got
    assert row["s1"] in got, "the recursive CTE stopped at depth 1"


def test_eligible_candidates_excludes_listed_and_suppressed(db, user_id):
    listed = {r["mal_id"] for r in query(
        "SELECT mal_id FROM list_entry WHERE user_id=%s", (user_id,))}
    elig = {r["mal_id"] for r in query(
        "SELECT mal_id FROM eligible_candidates(%s, 2000, false)", (user_id,))}
    assert elig, "no eligible candidates at all"
    assert not (elig & listed), "already-watched anime leaked into the pool"


def test_feedback_suppresses_a_candidate(db, user_id):
    target = scalar("SELECT mal_id FROM eligible_candidates(%s, 2000, false) LIMIT 1",
                    (user_id,))
    if target is None:
        pytest.skip("no candidates")
    query("INSERT INTO feedback (user_id, mal_id, action) VALUES (%s,%s,'not_interested')",
          (user_id, target))
    try:
        still = scalar(
            "SELECT count(*) FROM eligible_candidates(%s, 2000, false) WHERE mal_id=%s",
            (user_id, target))
        assert still == 0, "not_interested did not remove the anime"
    finally:
        query("DELETE FROM feedback WHERE user_id=%s AND mal_id=%s", (user_id, target))


def test_franchise_components_group_related_anime(db):
    row = one(
        """
        SELECT r.src, r.dst FROM relation r
         WHERE r.relation_type IN ('sequel','prequel')
         LIMIT 1
        """)
    if not row:
        pytest.skip("no relations ingested yet")
    a = scalar("SELECT franchise_id FROM franchise WHERE mal_id=%s", (row["src"],))
    b = scalar("SELECT franchise_id FROM franchise WHERE mal_id=%s", (row["dst"],))
    if a is None or b is None:
        pytest.skip("franchise table not refreshed")
    assert a == b, "sequel and prequel landed in different franchises"


# --------------------------------------------------------------- ranking --

def test_novelty_is_monotonic_in_popularity():
    assert novelty(1) < novelty(500) < novelty(5000)
    assert 0.0 <= novelty(1) <= 1.0
    assert novelty(10**9) == pytest.approx(1.0)


def _cand(mal_id, franchise, pred, pop, genres):
    return Candidate(mal_id=mal_id, title=f"t{mal_id}", franchise_id=franchise,
                     predicted=pred, novelty=novelty(pop), final=pred,
                     row={"mal_genres": genres})


def test_rerank_keeps_one_entry_per_franchise():
    cands = [_cand(i, franchise=1, pred=9 - i * 0.1, pop=100, genres=["Action"])
             for i in range(5)]
    cands.append(_cand(99, franchise=2, pred=7.0, pop=100, genres=["Comedy"]))
    out = rerank(cands, novelty_weight=0.0, diversity_weight=0.0, limit=10)
    assert len({c.franchise_id for c in out}) == len(out)
    assert len(out) == 2


def test_novelty_weight_promotes_obscure_titles():
    famous = _cand(1, 1, pred=8.5, pop=10, genres=["Action"])
    obscure = _cand(2, 2, pred=8.2, pop=9000, genres=["Drama"])
    off = rerank([famous, obscure], novelty_weight=0.0, diversity_weight=0.0, limit=2)
    on = rerank([famous, obscure], novelty_weight=2.0, diversity_weight=0.0, limit=2)
    assert off[0].mal_id == 1, "with novelty off the higher prediction should win"
    assert on[0].mal_id == 2, "with novelty on the obscure title should win"


def test_format_class_splits_main_from_side(db):
    rows = {r["mt"]: r["fc"] for r in query(
        "SELECT m AS mt, format_class(m) AS fc FROM unnest(ARRAY"
        "['tv','movie','ona','ova','special','tv_special','music','pv']) m")}
    assert rows["tv"] == rows["movie"] == rows["ona"] == "main"
    assert rows["ova"] == rows["special"] == rows["tv_special"] == "side"
    assert rows["music"] == rows["pv"] == "noise"


def test_next_up_excludes_side_formats(db, user_id):
    """The bug this guards: MAL calls a 4-minute special a 'sequel', so
    continuations used to fill with OVAs instead of real next seasons."""
    from malrec.surfaces import MAIN, SIDE
    assert MAIN == ["main"] and SIDE == ["side"]
    bad = query(
        """
        SELECT a.title FROM recommendation r JOIN anime a ON a.mal_id = r.mal_id
         WHERE r.user_id=%s AND r.surface='next_up'
           AND format_class(a.media_type) <> 'main'
        """,
        (user_id,))
    assert not bad, f"side-format entries leaked into next_up: {bad[:3]}"


def test_discovery_tabs_leave_out_side_stories_and_recaps(db, user_id):
    """A side story belongs next to its parent, in Side Stories; a recap
    either spoils or repeats. Both used to reach Hidden Gems and Discover."""
    bad = query(
        """
        SELECT r.surface, a.title FROM recommendation r JOIN anime a ON a.mal_id = r.mal_id
         WHERE r.user_id=%s
           AND r.surface IN ('safe_bets', 'discover', 'hidden_gems', 'this_season')
           AND EXISTS (SELECT 1 FROM relation x WHERE x.src = r.mal_id
                         AND x.relation_type IN ('parent_story', 'full_story'))
        """,
        (user_id,))
    assert not bad, f"side stories / recaps in discovery tabs: {bad[:3]}"


def test_reasons_do_not_repeat_the_same_anime(db, user_id):
    """An anime linked by both providers must not be listed twice."""
    rows = query(
        "SELECT reasons FROM recommendation WHERE user_id=%s AND surface='discover'"
        " LIMIT 40", (user_id,))
    if not rows:
        pytest.skip("no discover surface built")
    for r in rows:
        titles = [x["title"] for x in r["reasons"] if x.get("kind") == "because_you_liked"]
        assert len(titles) == len(set(titles)), f"duplicate reason: {titles}"


# ----------------------------------------------------------------- model --

def test_features_are_aligned_between_train_and_score(db, user_id):
    """The same vocabulary must produce the same width on both sides, or the
    model silently scores candidates against the wrong columns."""
    from malrec.features import build_vocabulary, load_rows, vectorise
    rated = [r["mal_id"] for r in query(
        "SELECT mal_id FROM list_entry WHERE user_id=%s AND score>0 LIMIT 40", (user_id,))]
    cand = [r["mal_id"] for r in query(
        "SELECT mal_id FROM eligible_candidates(%s, 2000, false) LIMIT 40", (user_id,))]
    if len(rated) < 10 or len(cand) < 10:
        pytest.skip("not enough data")
    tr = load_rows(user_id, rated)
    te = load_rows(user_id, cand)
    vocab = build_vocabulary(list(tr.values()))
    Xtr = vectorise(list(tr.values()), vocab)
    Xte = vectorise(list(te.values()), vocab)
    assert Xtr.shape[1] == Xte.shape[1] == vocab.width
    assert np.isfinite(Xtr).all() and np.isfinite(Xte).all()


def test_temporal_holdout_beats_the_community_mean(db, user_id):
    """The whole point of the project: personalisation must add something over
    just sorting by MAL's public score."""
    from malrec.eval import spearman, temporal_holdout
    m = temporal_holdout(user_id)
    if "spearman" not in m:
        pytest.skip(m.get("note", "not enough history"))

    rated = query(
        """
        SELECT le.score, a.mal_mean
          FROM list_entry le JOIN anime a ON a.mal_id = le.mal_id
         WHERE le.user_id=%s AND le.score>0 AND a.mal_mean IS NOT NULL
         ORDER BY coalesce(le.finished_at::timestamptz, le.updated_at) DESC
         LIMIT 40
        """,
        (user_id,))
    baseline = spearman([r["mal_mean"] for r in rated], [r["score"] for r in rated])
    assert m["spearman"] > baseline, (
        f"model {m['spearman']:.3f} did not beat the MAL-mean baseline {baseline:.3f}")


def test_implicit_rows_respect_the_cutoff(db, user_id):
    """The temporal holdout must not see an entry the user only touched after
    the cutoff, or the evaluation silently leaks the future."""
    import datetime as dt

    from malrec.features import implicit_rows

    all_rows = implicit_rows(user_id, 7.5)
    if not all_rows:
        pytest.skip("no unscored entries")
    early = dt.datetime(2019, 1, 1, tzinfo=dt.UTC)
    assert len(implicit_rows(user_id, 7.5, cutoff=early)) < len(all_rows)


def test_implicit_rows_exclude_plan_to_watch(db, user_id):
    """plan_to_watch is an interest signal, not a quality one; including it
    measurably cost nDCG, so it must stay out of the training set."""
    from malrec.config import settings
    from malrec.features import implicit_rows

    assert "plan_to_watch" not in settings().implicit_offsets
    ptw = {r["mal_id"] for r in query(
        "SELECT mal_id FROM list_entry WHERE user_id=%s AND status='plan_to_watch'",
        (user_id,))}
    used = {r["mal_id"] for r in implicit_rows(user_id, 7.5)}
    assert not (used & ptw)


def test_calibration_is_monotonic_and_bounded(db, user_id):
    """Calibration may fix the displayed number but must never reorder."""
    import numpy as np

    from malrec.model import load_latest

    model, _ = load_latest(user_id)
    if model is None:
        pytest.skip("no model trained")
    raw = np.linspace(5.0, 10.0, 25)
    out = model.calibrate(raw)
    assert np.all(np.diff(out) >= -1e-9), "calibration reordered predictions"
    assert out.min() >= 1.0 and out.max() <= 10.0
