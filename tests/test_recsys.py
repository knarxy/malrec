"""The population model, the hybrid user model and the ranking rules.

The synthetic tests need no database: two groups of viewers who each love
one half of a catalogue and dislike the other. A new user who rates a few
items of group A highly must then see the *rest* of group A ranked above
group B - that is the whole point of fold-in, and it must hold with only a
handful of ratings.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from malrec.recsys.cf import CFParams, PopulationModel, knn_contributors
from malrec.recsys.hybrid import size_weight
from malrec.recsys.scorer import novelty_scale

N_ITEMS = 40
A_ITEMS = list(range(1000, 1000 + N_ITEMS // 2))       # loved by group A
B_ITEMS = list(range(1000 + N_ITEMS // 2, 1000 + N_ITEMS))


def _two_tribes(n_users: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    users, items, scores = [], [], []
    for u in range(n_users):
        likes, dislikes = (A_ITEMS, B_ITEMS) if u % 2 == 0 else (B_ITEMS, A_ITEMS)
        for m in rng.choice(likes, 14, replace=False):
            users.append(u); items.append(m); scores.append(rng.integers(8, 11))
        for m in rng.choice(dislikes, 8, replace=False):
            users.append(u); items.append(m); scores.append(rng.integers(3, 6))
    return np.array(users), np.array(items), np.array(scores, dtype=float)


@pytest.fixture(scope="module")
def pop():
    u, i, s = _two_tribes()
    p = CFParams(min_item_raters=5, sim_shrink=5, knn_k=15, mf_rank=4, mf_reg=1.0, mf_iters=8)
    m = PopulationModel(p).fit(u, i, s)
    m.fit_occurrence(u, i)
    return m


def test_fold_in_ranks_the_users_tribe_first(pop):
    seen = A_ITEMS[:4]
    fold = pop.fold_in({m: 10.0 for m in seen}, {m: 1.0 for m in seen})
    rest_a, all_b = A_ITEMS[4:], B_ITEMS
    sig = fold.signals(rest_a + all_b)
    for name in ("knn", "mf"):
        a, b = sig[name][:len(rest_a)], sig[name][len(rest_a):]
        assert a.mean() > b.mean(), f"{name} did not prefer the user's own tribe"


def test_fold_in_works_from_dislikes_alone(pop):
    """Low scores are information too: hating group B should lift group A."""
    seen = B_ITEMS[:4]
    fold = pop.fold_in({m: 3.0 for m in seen} | {A_ITEMS[0]: 8.0},
                       {m: 1.0 for m in [*seen, A_ITEMS[0]]})
    sig = fold.signals(A_ITEMS[1:] + B_ITEMS[4:])
    n = len(A_ITEMS) - 1
    assert sig["knn"][:n].mean() > sig["knn"][n:].mean()


def test_relevance_follows_the_list_not_the_scores(pop):
    """Co-occurrence relevance must work from list membership alone."""
    listed = {m: 1.0 for m in A_ITEMS[:5]}
    fold = pop.fold_in({}, {}, listed=listed)
    rel = fold.signals(A_ITEMS[5:] + B_ITEMS)["rel"]
    assert rel[:len(A_ITEMS) - 5].mean() > rel[len(A_ITEMS) - 5:].mean()


def test_unknown_items_get_neutral_signals(pop):
    fold = pop.fold_in({A_ITEMS[0]: 9.0}, {A_ITEMS[0]: 1.0})
    sig = fold.signals([999_999])
    assert sig["known"][0] == 0 and sig["bias"][0] == 0 and sig["knn"][0] == 0


def test_knn_contributors_only_name_rated_items_that_helped(pop):
    seen = {A_ITEMS[0]: 10.0, A_ITEMS[1]: 10.0, B_ITEMS[0]: 3.0}
    fold = pop.fold_in(seen, {m: 1.0 for m in seen})
    for mid, contrib in knn_contributors(fold, A_ITEMS[5], top=3):
        assert mid in seen and contrib > 0


# ------------------------------------------------------------------ rules --

def test_size_weight_hands_over_smoothly():
    ws = [size_weight(n, 80, 15) for n in (5, 25, 60, 80, 100, 143, 400)]
    assert all(a < b for a, b in pairwise(ws)), "must increase with list size"
    assert ws[0] < 0.01 and ws[-1] > 0.99
    assert abs(size_weight(80, 80, 15) - 0.5) < 1e-9


def test_novelty_scale_never_lets_novelty_decide_a_flat_ranking():
    assert novelty_scale(np.full(50, 8.3), 0.5) == 0.0      # a thin-list failure
    assert novelty_scale(np.linspace(6, 9, 50), 0.5) == 1.0  # a confident model is untouched
    assert 0 < novelty_scale(np.linspace(8.0, 8.4, 50), 0.5) < 1


def test_relevance_bonus_follows_the_personal_share(monkeypatch):
    from malrec import rank
    from malrec.config import settings
    cfg = settings()
    monkeypatch.setattr(cfg, "relevance_weight", 0.5)
    monkeypatch.setattr(cfg, "relevance_weight_personal", 0.0)
    z = np.array([-3.0, 0.0, 4.0])

    class M:
        personal_share = 1.0
    assert np.allclose(rank.relevance_bonus(M(), z), 0), "long histories must be left alone"
    M.personal_share = 0.0
    assert np.allclose(rank.relevance_bonus(M(), z), 0.5 * z), "thin lists get the full pull"
    monkeypatch.setattr(cfg, "relevance_weight_personal", 1.0)
    M.personal_share = 1.0
    assert rank.relevance_bonus(M(), z).max() == 2.0, "the personal dial is bounded"


# --------------------------------------------------------- against the DB --

def test_item_store_reproduces_the_sql_features():
    """The in-memory path must produce the production feature vectors, or
    every comparison against the old model is meaningless."""
    from malrec.config import settings
    from malrec.db import one, query, scalar
    from malrec.features import build_vocabulary, load_rows, vectorise
    from malrec.recsys.items import ItemStore
    try:
        scalar("SELECT 1")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"database not reachable: {e}")
    row = one("SELECT id FROM app_user WHERE mal_username = %s", (settings().malrec_user,))
    if not row:
        pytest.skip("no synced user")
    cfg = settings()
    scored = query("""SELECT mal_id, score, recency_weight(finished_at, updated_at, %s::real,
                             %s::real) w FROM list_entry WHERE user_id=%s AND score>0""",
                   (cfg.recency_half_life_years, cfg.recency_floor, row["id"]))
    ids = [r["mal_id"] for r in scored]
    prod = load_rows(row["id"], ids)
    store = ItemStore.load(ids)
    taste = store.taste_vector({r["mal_id"]: float(r["score"]) for r in scored},
                               {r["mal_id"]: float(r["w"]) for r in scored})
    mem = {r["mal_id"]: r for r in store.user_rows(ids, taste)}
    common = [i for i in ids if i in prod and i in mem]
    vocab = build_vocabulary([prod[i] for i in common])
    diff = np.abs(vectorise([prod[i] for i in common], vocab)
                  - vectorise([mem[i] for i in common], vocab))
    assert diff.max() < 1e-4


# ------------------------------------------------ app features (offline) --

def test_filters_keep():
    from malrec.surfaces import Filters
    tv = {"media_type": "tv", "num_episodes": 12, "season_year": 2019, "mal_genres": ["Drama"]}
    film = {"media_type": "movie", "num_episodes": 1, "season_year": 2005, "mal_genres": ["Horror"]}
    assert Filters().keep(tv) and not Filters().active()
    assert not Filters(media_types=["movie"]).keep(tv)
    assert Filters(eps_max=13).keep(tv) and not Filters(eps_min=13).keep(tv)
    assert Filters(eps_min=13).keep(film)                  # films ignore episode filters
    assert not Filters(year_min=2010).keep(film)
    assert not Filters(exclude_genres=["Horror"]).keep(film)


def test_audience_stats_scaling():
    from malrec.features import audience_stats
    mid = audience_stats(0.0437, None, None)
    assert abs(mid["drop_c"]) < 0.1 and mid["polar_c"] == 0.0
    assert audience_stats(None, 0.2, None)["drop_c"] > 1       # AniList fallback
    split = {"10": 300, "100": 300}                            # 1s and 10s only
    agree = {"70": 290, "80": 310}
    assert audience_stats(None, None, split)["polar_c"] > 5
    assert audience_stats(None, None, agree)["polar_c"] < -3
    assert audience_stats(None, None, {"70": 10})["polar_c"] == 0.0   # too few


def test_staff_signal_is_leave_one_out():
    import scipy.sparse as sp

    from malrec.recsys.items import ItemStore
    st = ItemStore(rows={}, vec_ids=np.array([]), vecs=np.zeros((0, 256)))
    # items 1,2,3 share director A; item 4 has director B
    st.staff = sp.csr_matrix(np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=float))
    st.staff_row = {1: 0, 2: 1, 3: 2, 4: 3}
    rated, dev, w = [1, 2], np.array([2.0, 1.0]), np.array([1.0, 1.0])
    out = st.staff_signal(rated, dev, w, [1, 3, 4, 99], shrink=0.0)
    assert out[0] == pytest.approx(1.0)        # item 1 sees only item 2, never itself
    assert out[1] == pytest.approx(1.5)        # unrated item 3 sees both
    assert out[2] == 0.0 and out[3] == 0.0     # other director / unknown item


def test_size_calibration_interpolates_in_log_n():
    from malrec.recsys.service import size_calibration
    t = [{"n": 10, "slope": 0.5, "intercept": 3.0}, {"n": 100, "slope": 0.9, "intercept": 1.0}]
    assert size_calibration(None, 50) is None
    assert size_calibration(t, 3) == (0.5, 3.0)
    assert size_calibration(t, 1000) == (0.9, 1.0)
    a, b = size_calibration(t, 31.6227766)     # halfway in log space
    assert a == pytest.approx(0.7, abs=1e-6) and b == pytest.approx(2.0, abs=1e-6)


def test_learn_weights_needs_enough_feedback_and_recovers_the_rate(monkeypatch):
    from malrec import learn
    rng = np.random.default_rng(0)

    def rows(n):
        out = []
        for _ in range(n):
            pred, z, nov = rng.normal(7.5, 0.7), rng.normal(), rng.uniform()
            # people act as if one sd of relevance were worth 0.5 points
            p = 1 / (1 + np.exp(-2.0 * ((pred - 7.5) + 0.5 * z)))
            pos = bool(rng.uniform() < p)
            out.append({"user_id": 1, "predicted": pred, "relevance_z": z, "novelty": nov,
                        "positive": pos, "negative": not pos})
        return out

    monkeypatch.setattr(learn, "query", lambda sql, params=None: rows(50))
    assert learn.learn_weights()["status"].startswith("not enough")
    monkeypatch.setattr(learn, "query", lambda sql, params=None: rows(4000))
    r = learn.learn_weights(boot=50)
    assert r["status"] == "ok"
    assert 0.35 < r["relevance_points_per_sd"] < 0.65
    lo, hi = r["relevance_ci90"]
    assert lo < 0.5 < hi


def test_top_of_list_calibration_is_optional_per_size():
    from malrec.recsys.service import size_calibration
    t = [{"n": 10, "slope": 0.8, "intercept": 1.5, "top_slope": 0.5, "top_intercept": 4.0},
         {"n": 100, "slope": 0.9, "intercept": 1.0}]            # no top line fitted
    assert size_calibration(t, 10, prefix="top_") == (0.5, 4.0)
    assert size_calibration(t, 1000, prefix="top_") == (0.5, 4.0)   # held flat
    assert size_calibration([{"n": 5, "slope": 1, "intercept": 0}], 5, prefix="top_") is None


def test_weekly_upcoming_rebuild_survives_a_broken_account(monkeypatch):
    import json

    from typer.testing import CliRunner

    from malrec import cli, surfaces
    from malrec.ingest import jobs
    built = []

    def fake_build(uid, surface, limit):
        if uid == 2:
            raise ValueError("user 2 has no scored entries to learn from")
        built.append(uid)
        return []

    monkeypatch.setattr(cli, "_setup", lambda *a, **k: None)
    monkeypatch.setattr(cli, "query", lambda sql, *a: [
        {"id": i, "mal_username": f"u{i}"} for i in (1, 2, 3)])
    monkeypatch.setattr(jobs, "sync_upcoming", lambda: {"catalog": 0, "related": 0})
    monkeypatch.setattr(surfaces, "build_surface", fake_build)
    r = CliRunner().invoke(cli.app, ["sync", "upcoming"])
    assert r.exit_code == 0, r.output
    out = json.loads(r.output[r.output.index("{"):])
    assert built == [1, 3] and list(out["failed"]) == ["u2"]


def test_uncertainty_band_and_coverage():
    from malrec.uncertainty import band, coverage, residual_quantiles
    rng = np.random.default_rng(0)
    pred = rng.uniform(6, 8.5, 4000)
    truth = np.clip(np.round(pred + rng.normal(0, 1.0, 4000)), 1, 10)
    q = residual_quantiles(pred[:2000], truth[:2000])
    assert 0.82 < coverage(q, pred[2000:], truth[2000:]) < 0.97   # 5-95 % band
    b = band(8.0, q)
    assert b["low"] <= 8 <= b["high"] and 0.2 < b["p9"] < 0.5
    assert band(8.0, None) is None and band(float("nan"), q) is None
    assert band(6.0, q)["p9"] < band(8.0, q)["p9"]


def test_calibrated_list_is_ordered_by_score_not_by_genre_fit():
    from malrec.rank import Candidate, calibrated
    cands = [Candidate(mal_id=i, title=str(i), franchise_id=i, predicted=p, novelty=0.0,
                       final=p, row={"mal_genres": g})
             for i, (p, g) in enumerate([(8.3, ["Comedy"]), (8.0, ["Action"]),
                                         (6.8, ["Action", "Drama", "Fantasy"]), (7.5, ["Comedy"])])]
    out = calibrated(cands, {"Action": 0.4, "Drama": 0.3, "Fantasy": 0.3}, 0.7, 3)
    finals = [c.final for c in out]
    assert finals == sorted(finals, reverse=True)
    assert out[0].predicted == max(c.predicted for c in out)


def test_tokens_are_sealed_at_rest(monkeypatch):
    from cryptography.fernet import Fernet

    from malrec import tokenbox
    from malrec.config import settings
    monkeypatch.setattr(settings(), "token_key", Fernet.generate_key().decode())
    tokenbox._fernet.cache_clear()
    try:
        sealed = tokenbox.seal("secret-token")
        assert sealed.startswith("enc:v1:") and "secret-token" not in sealed
        assert tokenbox.open_(sealed) == "secret-token"
        assert tokenbox.seal(sealed) == sealed            # never double-sealed
        assert tokenbox.open_("legacy-plain") == "legacy-plain"
    finally:
        tokenbox._fernet.cache_clear()


def test_gate_lets_the_exam_decide_and_keeps_a_user_veto():
    from malrec.recsys.service import gate_decision
    user = [0.75, 0.70, 0.74, 0.72, 0.77, 0.80, 0.82, 0.76]
    # one noisy split down 0.03, mean down 0.005: the old rule would refuse;
    # an exam that says the candidate is better lets it through
    noisy = [u - (0.03 if k == 2 else 0.002) for k, u in enumerate(user)]
    ok, _ = gate_decision(user, noisy, None)
    assert not ok
    ok, why = gate_decision(user, noisy, {"delta": 0.004})
    assert ok and "exam" in why
    # the exam alone can refuse ...
    ok, why = gate_decision(user, user, {"delta": -0.005})
    assert not ok and why.startswith("exam failed")
    # ... and the gate user can veto a clear loss on their own splits
    ok, why = gate_decision(user, [u - 0.02 for u in user], {"delta": 0.01})
    assert not ok and "veto" in why


def test_blend_options_follow_config(monkeypatch):
    from malrec.config import settings
    from malrec.recsys.service import blend_opts
    monkeypatch.setattr(settings(), "blend_size_n0", 150.0)
    monkeypatch.setattr(settings(), "blend_lam_max", 0.5)
    assert blend_opts() == {"size_n0": 150.0, "lam_max": 0.5}


def test_ranking_share_can_follow_list_size_instead_of_the_blend(monkeypatch):
    from malrec import rank
    from malrec.config import settings
    from malrec.recsys.scorer import HybridScorer
    cfg = settings()
    monkeypatch.setattr(cfg, "relevance_weight", 0.5)
    monkeypatch.setattr(cfg, "relevance_weight_personal", 0.5)

    class UM:                       # a blend that gives the personal model 43%
        mode, lam, inner = "sized", 0.43, None
    s = HybridScorer.__new__(HybridScorer)
    s.um, s._scores = UM(), {m: 8.0 for m in range(146)}
    z = np.array([0.0, 3.0, 7.0])
    monkeypatch.setattr(cfg, "ranking_share_n0", 0.0)
    assert s.ranking_share == s.personal_share == 0.43
    coupled = rank.relevance_bonus(s, z)
    monkeypatch.setattr(cfg, "ranking_share_n0", 80.0)
    assert s.ranking_share > 0.98            # 146 ratings on the n0=80 curve
    decoupled = rank.relevance_bonus(s, z)
    # coupled, the unbounded thin-list pull would take over this long list
    assert coupled[2] - coupled[1] > 0.8 and decoupled[2] - decoupled[1] < 0.05


def test_feature_importance_is_empty_without_a_personal_model():
    """A short list under the late handover never fits a personal ridge; the
    CLI and the "Your taste" panel must not crash on it."""
    from malrec.model import feature_importance
    from malrec.recsys.scorer import HybridScorer

    class UM:
        mode, lam, inner = "sized", 0.0, None
    s = HybridScorer.__new__(HybridScorer)
    s.um = UM()
    assert s.personal is None and feature_importance(s) == []


def test_exam_tolerance_allows_for_noise():
    from malrec.recsys.service import gate_decision
    user = [0.75] * 8
    # the 2026-10-03 dry run: -.0029 with a standard error of .0031 is noise
    assert gate_decision(user, user, {"delta": -0.0029, "se": 0.0031})[0]
    # a loss beyond both the tolerance and the noise still fails
    assert not gate_decision(user, user, {"delta": -0.008, "se": 0.003})[0]


def test_roles_and_draws_do_not_depend_on_who_else_is_sampled():
    from malrec.recsys.hybrid import unit_hash, user_rng
    a = [unit_hash("role", 0, u) for u in range(500)]
    assert a == [unit_hash("role", 0, u) for u in range(500)]
    share = np.mean([x < 0.15 for x in a])
    assert 0.10 < share < 0.20, "about the configured share goes to the stacker"
    assert user_rng("cal", 0, 7, 25).random() == user_rng("cal", 0, 7, 25).random()
    assert user_rng("cal", 0, 7, 25).random() != user_rng("cal", 0, 8, 25).random()


def test_als_start_values_belong_to_the_item(pop):
    """Adding a user must not change where the factorisation starts."""
    u, i, s = _two_tribes()
    p = CFParams(min_item_raters=5, sim_shrink=5, knn_k=15, mf_rank=4, mf_reg=1.0, mf_iters=8)
    m1 = PopulationModel(p).fit(u, i, s)
    extra = np.array([u.max() + 1] * 3)
    m2 = PopulationModel(p).fit(np.concatenate([u, extra]), np.concatenate([i, A_ITEMS[:3]]),
                                np.concatenate([s, [9.0, 9.0, 9.0]]))
    assert np.allclose(m1.V, m2.V, atol=0.05), "one extra user moved the factors far"


def test_evidence_is_zero_without_rated_neighbours(pop):
    fold = pop.fold_in({A_ITEMS[0]: 9.0, A_ITEMS[1]: 8.0}, {A_ITEMS[0]: 1.0, A_ITEMS[1]: 1.0})
    ev = fold.evidence([A_ITEMS[2], B_ITEMS[0], 999_999])
    assert ev[0] > 0 and ev[2] == 0
    assert pop.fold_in({}, {}).evidence([A_ITEMS[2]])[0] == 0


def test_risk_penalty_hits_only_population_optimism_and_follows_trust(monkeypatch):
    from malrec.config import settings
    from malrec.recsys.scorer import HybridScorer
    cfg = settings()
    monkeypatch.setattr(cfg, "disagreement_penalty", 1.0)
    monkeypatch.setattr(cfg, "unfamiliar_genre_penalty", 0.0)
    monkeypatch.setattr(cfg, "long_series_penalty", 0.0)
    monkeypatch.setattr(cfg, "acclaim_penalty", 0.0)
    monkeypatch.setattr(cfg, "blend_lam_max", 0.5)

    class UM:
        mode, inner = "sized", object()
        lam = 0.5

        def predict_parts(self, ids):          # population 9.0 / 7.0, personal 8.0
            return np.array([9.0, 7.0]), np.array([8.0, 8.0])
    s = HybridScorer.__new__(HybridScorer)
    s.um, s.calibration = UM(), (1.2, -1.0)
    r = s.risk_penalty([1, 2])
    assert np.allclose(r, [1.2, 0.0]), "only where the population is the optimist, in display points"
    UM.lam = 0.25
    assert np.allclose(s.risk_penalty([1, 2]), [0.6, 0.0]), "half the trust, half the penalty"
    monkeypatch.setattr(cfg, "disagreement_penalty", 0.0)
    assert not s.risk_penalty([1, 2]).any()


def test_risk_penalty_for_unfamiliar_genres_and_long_series(monkeypatch):
    from malrec.config import settings
    from malrec.recsys.scorer import HybridScorer
    cfg = settings()
    monkeypatch.setattr(cfg, "disagreement_penalty", 0.0)
    monkeypatch.setattr(cfg, "unfamiliar_genre_penalty", 0.3)
    monkeypatch.setattr(cfg, "long_series_penalty", 0.2)
    monkeypatch.setattr(cfg, "acclaim_penalty", 0.0)
    monkeypatch.setattr(cfg, "blend_lam_max", 0.5)

    from types import SimpleNamespace
    store = SimpleNamespace(rows={
        1: {"mal_genres": ["Sports", "Shounen"], "num_episodes": 75},   # Ippo
        2: {"mal_genres": ["Comedy", "Seinen"], "num_episodes": 12},
        3: {"mal_genres": ["Comedy", "Kids"], "num_episodes": 60}})

    class UM:
        mode, inner, lam = "sized", object(), 0.5
    UM.store = store
    s = HybridScorer.__new__(HybridScorer)
    s.um, s.calibration = UM(), (1.0, 0.0)
    r = s.risk_penalty([1, 2, 3], familiar={"Comedy"})
    # demographics (Shounen, Seinen, Kids) never count as unfamiliar
    assert np.allclose(r, [0.5, 0.0, 0.2])
    UM.inner = None
    # no personal model: no genre judgement, but length counts at any size
    assert np.allclose(s.risk_penalty([1, 2, 3], familiar={"Comedy"}), [0.2, 0.0, 0.2])


def test_long_series_includes_long_running_ongoing_shows():
    from malrec.recsys.scorer import is_long_series
    assert is_long_series({"num_episodes": 75})
    assert not is_long_series({"num_episodes": 24})
    conan = {"num_episodes": 0, "status": "currently_airing", "season_year": 1996}
    fresh = {"num_episodes": 0, "status": "currently_airing", "season_year": 2026}
    assert is_long_series(conan, year=2026) and not is_long_series(fresh, year=2026)
    assert not is_long_series({"num_episodes": 0, "status": "finished_airing",
                               "season_year": 1996}, year=2026)


def test_memory_adjustment_can_be_capped(monkeypatch):
    from malrec import rank
    from malrec.config import settings
    cfg = settings()

    class Long:                      # long memory loves item 2, which the key ranks last
        personal_share = ranking_share = 1.0

        def predict_ids(self, ids):
            return np.array([7.0, 7.2, 9.5])

        def calibrate(self, raw):
            return np.asarray(raw)

        def relevance(self, ids):
            return np.zeros(len(ids))
    key = np.array([8.0, 7.8, 6.9])
    monkeypatch.setattr(cfg, "safe_bets_memory_cap", 0.0)
    free = rank._mix_bonus(None, Long(), [1, 2, 3], key, 0.5)
    assert free[2] > 0.3
    monkeypatch.setattr(cfg, "safe_bets_memory_cap", 0.3)
    capped = rank._mix_bonus(None, Long(), [1, 2, 3], key, 0.5)
    assert np.abs(capped).max() <= 0.3 + 1e-9 and capped[2] == 0.3


def test_side_contributions_recombine_to_the_blend():
    """The "two voices" reasons are each side's own contributions; weighted by
    the blend share they must add back up to the combined drivers."""
    from types import SimpleNamespace

    from malrec.recsys.hybrid import STACK_FEATURES
    from malrec.recsys.scorer import HybridScorer

    rng = np.random.default_rng(0)
    nf = len(STACK_FEATURES)
    names = ["g:Drama", "g:Comedy", "mal_mean"]
    personal = SimpleNamespace(
        ridge=SimpleNamespace(coef_=np.array([0.4, -0.3, 0.5])),
        vocab=SimpleNamespace(names=names), stacker=SimpleNamespace(),
        with_stack_features=False, _features=lambda rows: np.array([[1.0, 1.0, 1.2]] * len(rows)))
    um = SimpleNamespace(
        mode="sized", lam=0.4, inner=personal,
        stacker=SimpleNamespace(coef_=rng.normal(0, 0.3, nf)),
        stack_inputs=lambda ids: rng.normal(0, 1, (len(ids), nf)))
    sc = HybridScorer.__new__(HybridScorer)
    sc.um = um
    stack, pers = sc._side_contributions([7], [{"mal_id": 7}])
    assert pers[0] and stack[0]
    assert all(v != 0 for v in pers[0].values())
    fixed = rng.normal(0, 1, (1, nf))      # same inputs for both calls below
    um.stack_inputs = lambda ids: fixed
    stack, pers = sc._side_contributions([7], [{"mal_id": 7}])
    combined = sc._contributions([7], [{"mal_id": 7}])[0]
    for lbl in set(stack[0]) | set(pers[0]):
        want = 0.6 * stack[0].get(lbl, 0.0) + 0.4 * pers[0].get(lbl, 0.0)
        assert combined[lbl] == pytest.approx(want)


def test_negative_reasons_read_as_their_opposite():
    from malrec.recsys.scorer import signed_label
    assert signed_label("fits your taste profile", -0.3) == "outside your usual taste"
    assert signed_label("fits your taste profile", 0.3) == "fits your taste profile"
    assert signed_label("Revenge", -0.1) == "Revenge"


def test_acclaim_penalty_hits_lift_from_acclaim_and_fades_with_evidence(monkeypatch):
    """Cowboy Bebop for the reference profile: the personal model's lift was
    almost all MAL/AniList score, with no rated title linked to it."""
    from types import SimpleNamespace

    from malrec.config import settings
    from malrec.recsys.scorer import HybridScorer
    cfg = settings()
    for k in ("disagreement_penalty", "unfamiliar_genre_penalty", "long_series_penalty"):
        monkeypatch.setattr(cfg, k, 0.0)
    monkeypatch.setattr(cfg, "acclaim_penalty", 0.6)
    monkeypatch.setattr(cfg, "acclaim_evidence_e0", 0.1)
    monkeypatch.setattr(cfg, "blend_lam_max", 0.5)
    feats = {10: [1.0, 0.0], 11: [0.0, 0.0],      # rated: average [0.5, 0]
             1: [0.0, 2.0],                       # acclaimed, unlike the rated ones
             2: [1.0, -1.0]}                      # below-average acclaim
    inner = SimpleNamespace(
        ridge=SimpleNamespace(coef_=np.array([0.5, 0.4])),
        vocab=SimpleNamespace(names=["g:Drama", "mal_mean"]),
        rows=lambda ids: [{"mal_id": m} for m in ids if m in feats],
        _features=lambda rows: np.array([feats[r["mal_id"]] for r in rows]))
    evidence = {"v": np.array([0.0, 0.0])}
    um = SimpleNamespace(mode="sized", lam=0.5, inner=inner, store=SimpleNamespace(rows={}),
                         fold=SimpleNamespace(evidence=lambda ids: evidence["v"]))
    s = HybridScorer.__new__(HybridScorer)
    s.um, s.calibration, s._scores = um, (1.0, 0.0), {10: 8.0, 11: 7.0}
    assert np.allclose(s.acclaim_lift([1, 2]), [0.8, -0.4])
    parts = s.risk_parts([1, 2])
    assert np.allclose(parts["acclaim"], [0.48, 0.0]), "only acclaim above the rated average"
    evidence["v"] = np.array([0.1, 0.0])
    assert np.allclose(s.risk_penalty([1, 2]), [0.24, 0.0]), "co-rating evidence halves it"
    um.lam = 0.125
    assert np.allclose(s.risk_penalty([1, 2]), [0.06, 0.0]), "follows the trust"


def test_predict_parts_puts_each_prediction_at_its_item():
    from types import SimpleNamespace

    from malrec.recsys.hybrid import UserModel
    inner = SimpleNamespace(
        rows=lambda ids: [{"mal_id": m} for m in reversed(ids) if m != 3],   # unordered, 3 unknown
        _predict_known=lambda rows: np.array([r["mal_id"] * 10.0 for r in rows]))
    um = SimpleNamespace(inner=inner, lam=0.5,
                         stack=lambda ids: np.array([m + 0.5 for m in ids]))
    st, pe = UserModel.predict_parts(um, [1, 2, 3, 4])
    assert np.allclose(st[[0, 1, 3]], [1.5, 2.5, 4.5]) and np.isnan(st[2])
    assert np.allclose(pe[[0, 1, 3]], [10, 20, 40]) and np.isnan(pe[2])


def test_other_tabs_tag_titles_safe_bets_also_lists(monkeypatch):
    from malrec import surfaces
    monkeypatch.setattr(surfaces, "query", lambda sql, p=None: [{"mal_id": 2}])
    out = surfaces.mark_shared(7, "hidden_gems", [{"mal_id": 1}, {"mal_id": 2}])
    assert [it.get("also_in") for it in out] == [None, "safe_bets"]
    # Safe Bets itself carries no tag
    assert surfaces.mark_shared(7, "safe_bets", [{"mal_id": 2}]) == [{"mal_id": 2}]
