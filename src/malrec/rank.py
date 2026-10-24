"""Two-stage ranking.

Retrieval and ranking are deliberately separate. Measured on the reference profile, using
the community recommendation graph directly as a *ranker* scores rho 0.08 -
barely above chance. "People who liked X also watched Y" predicts what someone
will watch, not what they will rate highly. So the graph generates candidates
and the taste model orders them.

    1. retrieve  - eligible candidates from Postgres (prerequisites, feedback
                   and franchise rules already applied by the SQL layer)
    2. score     - the trained taste model
    3. rerank    - novelty + diversity, collapsed to one entry per franchise
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .config import settings
from .db import query
from .features import load_rows, vectorise
from .model import Model, TasteModel

if TYPE_CHECKING:
    from .recsys.ltr import RankContext

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    mal_id: int
    title: str
    franchise_id: int
    predicted: float          # calibrated 1-10, what the UI shows
    novelty: float
    final: float              # raw score plus novelty/diversity, for ordering
    raw: float = 0.0          # uncalibrated model output
    relevance: float = 0.0    # ranking bonus for sitting in the user's viewing neighbourhood
    relevance_z: float = 0.0  # the relevance signal itself, before weighting
    personal_share: float = 1.0   # the share the ranking dials used (model.ranking_share)
    shown: float | None = None   # display value if it differs from `predicted`
    bonus: float = 0.0           # extra ordering term (Safe Bets' long-memory mix + pop_bonus)
    pop_bonus: float = 0.0       # the popularity prior's part of `bonus`
    risk: float = 0.0            # the disagreement penalty's part of `bonus` (>= 0, subtracted)
    row: dict = field(repr=False, default_factory=dict)
    reasons: list[dict] = field(default_factory=list)


# ------------------------------------------------------------- release age --

# How old a title is when someone watches it. Their own mix, read from their
# list at the time they watched each title, is what the era balance in
# rerank() aims a list at: without it the learned orders filled even a
# back-catalogue watcher's lists with this year's shows (2026-10-13).
AGE_EDGES = (2, 5, 10, 20)     # buckets: 0-1, 2-4, 5-9, 10-19 and 20+ years
AGE_MIN_ENTRIES = 10           # fewer dated titles: no era balance


def age_bucket(age: float) -> int:
    return int(np.searchsorted(AGE_EDGES, max(age, 0.0), side="right"))


@dataclass
class AgeMix:
    """A user's release-age mix (shares per AGE_EDGES bucket) and how to age
    a candidate: `ref_year` minus its release year."""
    share: np.ndarray
    ref_year: int
    year_of: Callable[[int], int | None]

    def bucket(self, mal_id: int) -> int | None:
        y = self.year_of(mal_id)
        return None if y is None else age_bucket(self.ref_year - y)


def age_mix(entries: list[dict], year_of: Callable[[int], int | None],
            ref_year: int) -> AgeMix | None:
    """From a list's entries (mal_id, status, at): every started title's age
    when it was watched (`at`'s year minus its release year), counted equally -
    the whole history, since 'I also watch older shows' is a long-run habit.
    None below AGE_MIN_ENTRIES dated titles."""
    counts = np.zeros(len(AGE_EDGES) + 1)
    for e in entries:
        y = year_of(e["mal_id"])
        if e["status"] == "plan_to_watch" or y is None or e.get("at") is None:
            continue
        counts[age_bucket(e["at"].year - y)] += 1
    if counts.sum() < AGE_MIN_ENTRIES:
        return None
    return AgeMix(counts / counts.sum(), ref_year, year_of)


def age_aim(mix: AgeMix, fresh_weight: float) -> AgeMix:
    """The mix a list is held to: the user's, with the share of titles at
    most a year old scaled by `fresh_weight` (config.age_fresh_weight)."""
    if fresh_weight == 1.0:
        return mix
    share = mix.share.copy()
    share[0] *= fresh_weight
    return replace(mix, share=share / share.sum()) if share.sum() > 0 else mix


# --------------------------------------------------------------- retrieval --

RETRIEVE_SQL = """
    SELECT c.mal_id, c.franchise_id, c.title, c.mal_mean, c.mal_popularity,
           c.known_franchise, c.format_class
      FROM eligible_candidates(%(uid)s, %(min_scorers)s, false) c
     WHERE (%(formats)s::text[] IS NULL OR c.format_class = ANY(%(formats)s))
"""


def retrieve(user_id: int, min_scorers: int | None = None,
             media_types: list[str] | None = None) -> list[dict]:
    """`media_types` takes format *classes* ('main' / 'side'), not raw MAL
    media types - see format_class() in the schema."""
    cfg = settings()
    rows = query(RETRIEVE_SQL, {
        "uid": user_id,
        "min_scorers": cfg.min_scoring_users if min_scorers is None else min_scorers,
        "formats": media_types,
    })
    log.info("retrieved %d eligible candidates", len(rows))
    return rows


# ------------------------------------------------------------------ scoring --

def novelty(popularity: int | None) -> float:
    """0 = everyone has heard of it, 1 = genuinely obscure.

    A user with 250 entries has already considered the top of the popularity
    chart, so a recommender that only surfaces famous titles is not useful
    however accurate its score prediction is.
    """
    return min(1.0, math.log10(max(popularity or 9999, 1)) / 3.7)


def score_candidates(user_id: int | None, model: Model, rows: list[dict],
                     shortlist: int | None = None, enrich: bool = True,
                     long_model=None, mix: float = 0.0,
                     pop_prior: float = 0.0,
                     learned: RankContext | None = None,
                     risk_scale: float = 1.0) -> list[Candidate]:
    """Predict for every eligible candidate, then keep the best `shortlist`.

    Scoring happens in two passes so that nothing is fetched speculatively:
    a cheap pass ranks everything on data already held, then only the
    resulting shortlist is enriched from AniList and rescored. A pool of
    several thousand therefore costs ~10 batched requests, not thousands.
    """
    if not rows:
        return []
    cfg = settings()
    n = shortlist or cfg.shortlist_size
    ids = [r["mal_id"] for r in rows]

    from .recsys.scorer import HybridScorer
    if isinstance(model, HybridScorer):
        return _score_hybrid(model, rows, n, enrich, long_model, mix, pop_prior, user_id,
                             learned, risk_scale)
    if user_id is None:
        raise ValueError("the legacy per-user model scores a stored user only")

    if enrich and len(ids) > n:
        cheap = load_rows(user_id, ids)
        usable0 = [r for r in rows if r["mal_id"] in cheap]
        if usable0:
            p0 = model.predict([cheap[r["mal_id"]] for r in usable0])
            order = np.argsort(-p0)[:n]
            rows = [usable0[i] for i in order]
            ids = [r["mal_id"] for r in rows]

    if enrich:
        from .ingest.ondemand import ensure_for_candidates
        ensure_for_candidates(ids)

    feat = load_rows(user_id, ids)
    usable = [r for r in rows if r["mal_id"] in feat]
    raw = model.predict([feat[r["mal_id"]] for r in usable])
    # Rank on the raw output, display the calibrated one. The map is monotonic
    # so the two orderings are identical; only the number shown changes.
    shown = model.calibrate(raw)

    cands = []
    for r, p, disp in zip(usable, raw, shown):
        f = feat[r["mal_id"]]
        nov = novelty(r.get("mal_popularity"))
        cands.append(Candidate(
            mal_id=r["mal_id"], title=r["title"],
            franchise_id=r["franchise_id"] or r["mal_id"],
            predicted=float(disp), raw=float(p), novelty=nov, final=float(p), row=f,
        ))
    cands.sort(key=lambda c: -c.predicted)
    return cands[:n]


def familiar_genres(user_id: int | None) -> set[str] | None:
    """Every genre on the user's list, any status."""
    if user_id is None:
        return None
    return {r["g"] for r in query(
        "SELECT DISTINCT unnest(a.mal_genres) AS g FROM list_entry le"
        " JOIN anime a ON a.mal_id = le.mal_id WHERE le.user_id = %s", (user_id,))}


def relevance_bonus(model, z: np.ndarray, personal_weight: float | None = None) -> np.ndarray:
    """Relevance, weighted by how much the ranking leans on the population.

    Thin lists need a strong, unbounded pull toward their viewing neighbourhood
    - without it the population layer's devotee classics take over (measured:
    recall below today, top-20 lists shared across strangers). A long history
    is already well served by the personal model, and the pull reshapes it
    toward mainstream titles, so there it gets its own, bounded dial.
    """
    cfg = settings()
    lam = getattr(model, "ranking_share", getattr(model, "personal_share", 1.0))
    rwp = cfg.relevance_weight_personal if personal_weight is None else personal_weight
    return (cfg.relevance_weight * (1 - lam) * z
            + rwp * lam * np.clip(z, -2.0, 2.0))


def _mix_bonus(model, long_model, ids: list[int], key: np.ndarray, weight: float) -> np.ndarray:
    """Ordering adjustment that blends in a no-decay model's key.

    With every rating weighted equally the model finds more of what the user
    goes on to love but orders their ratings worse (experiments/exp_tabs.py);
    so it only shapes the order, never the displayed score. Both keys are
    z-scored, mixed, and mapped back onto the first key's scale, so the
    diversity penalty in rerank() keeps its meaning."""
    raw2 = long_model.predict_ids(ids)
    key2 = long_model.calibrate(raw2) + relevance_bonus(long_model, long_model.relevance(ids))
    ok = np.isfinite(key) & np.isfinite(key2)
    if ok.sum() < 3:
        return np.zeros(len(ids))
    m1, s1 = key[ok].mean(), key[ok].std() + 1e-9
    m2, s2 = key2[ok].mean(), key2[ok].std() + 1e-9
    mixed = m1 + s1 * ((1 - weight) * (key - m1) / s1 + weight * (key2 - m2) / s2)
    out = np.where(ok, mixed - key, 0.0)
    cap = settings().safe_bets_memory_cap
    return np.clip(out, -cap, cap) if cap > 0 else out


@dataclass
class OrderTerms:
    """Every term of the order key, per row of the pool (rank.order_terms)."""
    raw: np.ndarray            # model output
    ok: np.ndarray             # rows the model can score
    shown: np.ndarray          # calibrated prediction (what ordering uses)
    z: np.ndarray              # relevance, z-scored over the pool
    rel: np.ndarray            # relevance bonus
    pop: np.ndarray            # popularity prior (thin lists)
    mix: np.ndarray            # long-memory bonus (Safe Bets, when configured)
    risk: np.ndarray           # risk penalty, subtracted
    key: np.ndarray            # shown + rel + mix + pop - risk + ltr; -inf where not ok
    lam: float                 # the personal share the dials used
    ltr: np.ndarray | None = None   # learned-order shift (recsys/ltr.py), when on


def order_terms(model, rows: list[dict], long_model=None, mix: float = 0.0,
                pop_prior: float = 0.0, user_id: int | None = None,
                learned: RankContext | None = None,
                familiar: set[str] | None = None, risk_scale: float = 1.0) -> OrderTerms:
    """The order key of a tab and every term in it, for the whole pool in one
    vectorised pass. _score_hybrid ranks by it; diagnostics (experiments/
    bench_misses.py) read the terms to tell why a title placed where it did.
    `learned` mixes in the learned order (Safe Bets, recsys/ltr.py);
    `familiar` stands in for the user's genres when there is no stored user;
    `risk_scale` is the risk term's share of the key (Safe Bets'
    safe_bets_key_risk - the learned order reads the parts either way)."""
    ids = [r["mal_id"] for r in rows]
    raw = model.predict_ids(ids)
    ok = ~np.isnan(raw)
    # Rank on predicted score *plus* relevance. The score says how much they
    # would like a title if they watched it; relevance says whether it is in
    # their world at all. Score alone hands every newcomer the same devotee
    # classics (measured: 38% of top-20s shared between unrelated users).
    z = model.relevance(ids)
    rel = relevance_bonus(model, z)
    rank_all = model.calibrate(raw)
    key = np.where(ok, rank_all + rel, -np.inf)
    bonus = (_mix_bonus(model, long_model, ids, np.where(ok, rank_all + rel, np.nan), mix)
             if long_model is not None and mix > 0 else np.zeros(len(ids)))
    # the share the dials use; stored with each candidate so a filtered read
    # recomputes the same relevance bonus
    lam = float(getattr(model, "ranking_share", getattr(model, "personal_share", 1.0)))
    w = pop_prior * (1.0 - lam)
    pop_part = np.zeros(len(ids))
    if w > 1e-3 and ok.sum() > 2:
        # thin lists lean on "well known and well liked": popularity in
        # standard deviations of the key, fading as the personal model takes over
        lp = np.array([-math.log10(max(r.get("mal_popularity") or 99999, 1)) for r in rows])
        popz = (lp - lp[ok].mean()) / (lp[ok].std() + 1e-9)
        pop_part = w * float(key[ok].std()) * popz
        bonus = bonus + pop_part
    # a pick the user's own model doubts is not a safe one (exp_toplist.py)
    fam = familiar if familiar is not None else familiar_genres(user_id)
    risk = (risk_scale * np.asarray(model.risk_penalty(ids, fam), dtype=float)
            if hasattr(model, "risk_penalty") and risk_scale > 0 else np.zeros(len(ids)))
    bonus = bonus - risk
    key = key + bonus
    mix_part = bonus - pop_part + risk
    t = OrderTerms(raw=raw, ok=ok, shown=rank_all, z=z, rel=rel, pop=pop_part,
                   mix=mix_part, risk=risk, key=key, lam=lam, ltr=np.zeros(len(ids)))
    from .recsys.scorer import HybridScorer
    if learned is not None and isinstance(model, HybridScorer):
        from .recsys.ltr import learned_shift
        from .recsys.service import item_store
        shift = learned_shift(model, t, ids, learned, item_store())
        t.ltr, t.key = shift, key + shift
    return t


def _score_hybrid(model, rows: list[dict], n: int, enrich: bool,
                  long_model=None, mix: float = 0.0, pop_prior: float = 0.0,
                  user_id: int | None = None,
                  learned: RankContext | None = None,
                  risk_scale: float = 1.0) -> list[Candidate]:
    """The whole eligible pool is scored in one vectorised pass; the item
    store already holds every candidate's features, so nothing is fetched
    except AniList data for the rare shortlisted title that still lacks it."""
    t = order_terms(model, rows, long_model, mix, pop_prior, user_id, learned,
                    risk_scale=risk_scale)
    raw, ok, z, rel, key = t.raw, t.ok, t.z, t.rel, t.key
    pop_part, risk, lam = t.pop, t.risk, t.lam
    # the learned shift rides in the bonus, so rerank and a filtered read
    # (predicted + relevance + bonus) keep the same order
    bonus = t.mix + t.pop - t.risk + (t.ltr if t.ltr is not None else 0.0)
    order = [k for k in np.argsort(-key) if ok[k]][:n]
    top = [rows[k] for k in order]
    if enrich:
        from .ingest.ondemand import ensure_for_candidates
        ensure_for_candidates([r["mal_id"] for r in top])
    feats = {r["mal_id"]: r for r in model.rows([r["mal_id"] for r in top])}
    # ordering uses calibrate(); the card may show a list-specific map
    rank_value = model.calibrate(raw[order])
    display_value = (model.calibrate_list(raw[order]) if hasattr(model, "calibrate_list")
                     else rank_value)
    cands = []
    for r, k, rv, dv in zip(top, order, rank_value, display_value):
        cands.append(Candidate(
            shown=float(dv),
            mal_id=r["mal_id"], title=r["title"],
            franchise_id=r["franchise_id"] or r["mal_id"],
            predicted=float(rv), raw=float(raw[k]),
            novelty=novelty(r.get("mal_popularity")), final=float(rv + rel[k]),
            row=feats.get(r["mal_id"], {}), relevance=float(rel[k]),
            relevance_z=float(z[k]), personal_share=lam, bonus=float(bonus[k]),
            pop_bonus=float(pop_part[k]), risk=float(risk[k]),
        ))
    return cands


# ----------------------------------------------------------------- reranking --

def rerank(cands: list[Candidate], novelty_weight: float | None = None,
           diversity_weight: float | None = None, limit: int = 50,
           scale_novelty: bool = True,
           considered: list[Candidate] | None = None,
           profile: dict[str, float] | None = None,
           ages: AgeMix | None = None) -> list[Candidate]:
    """Collapse franchises, then trade predicted score against novelty and
    genre balance.

    With the user's genre `profile` the list is built by calibrated
    re-ranking (Steck, RecSys 2018): greedily, each step taking the title
    that best trades relevance against the KL divergence between the user's
    genre mix and the list's. Measured (experiments/exp_calib.py): recall@50
    up in both tabs at every list size, genre mix closer to the user's. The
    older overlap penalty is the fallback without a profile. With `ages`
    (and age_calibration > 0) the list's release-age mix is held to the
    user's the same way.

    Franchise collapse matters more than it looks: without it the list fills
    with four seasons of the same show, all scoring nearly identically.
    """
    cfg = settings()
    nw = cfg.novelty_weight if novelty_weight is None else novelty_weight
    dw = cfg.diversity_weight if diversity_weight is None else diversity_weight
    if scale_novelty and cands:
        from .recsys.scorer import novelty_scale
        nw *= novelty_scale(np.array([c.predicted for c in cands]), cfg.novelty_ref_sd)

    ordered = sorted(cands, key=lambda c: -(c.predicted + c.relevance + c.bonus
                                            + nw * c.novelty))
    seen_franchises: set[int] = set()
    genre_counts: Counter[str] = Counter()
    picked: list[Candidate] = []

    for c in ordered:
        if c.franchise_id in seen_franchises:
            continue
        genres = set(c.row.get("mal_genres") or [])
        # marginal penalty for a genre mix already well represented above
        overlap = (sum(genre_counts[g] for g in genres) / len(genres)) if genres else 0.0
        c.final = c.predicted + c.relevance + c.bonus + nw * c.novelty - dw * overlap
        picked.append(c)
        seen_franchises.add(c.franchise_id)
        for g in genres:
            genre_counts[g] += 1
        if len(picked) >= (CALIB_POOL if profile else limit * 3):
            break

    if considered is not None:
        considered.extend(picked)     # everything whose genres shaped the list
    lam = settings().genre_calibration
    if profile and lam > 0 and picked:
        for c in picked:                # the plain key, without the overlap penalty
            c.final = c.predicted + c.relevance + c.bonus + nw * c.novelty
        aim = age_aim(ages, settings().age_fresh_weight) if ages is not None else None
        return calibrated(picked, profile, lam, limit, aim, settings().age_calibration)
    picked.sort(key=lambda c: -c.final)
    return picked[:limit]


CALIB_POOL = 300
CALIB_ALPHA = 0.01          # smoothing of the list's genre mix toward the user's


def calibrated(cands: list[Candidate], profile: dict[str, float], lam: float,
               limit: int, ages: AgeMix | None = None, mu: float = 0.0) -> list[Candidate]:
    """Steck's greedy calibrated selection over the franchise-collapsed
    candidates; returns the selected titles ordered by their ranking key.
    With `ages` each step also pays mu x the KL divergence between the
    user's release-age mix and the list's."""
    genres = sorted({g for c in cands for g in (c.row.get("mal_genres") or [])} | set(profile))
    gi = {g: k for k, g in enumerate(genres)}
    G = np.zeros((len(cands), len(genres)))
    for r, c in enumerate(cands):
        gs = c.row.get("mal_genres") or []
        for g in gs:
            G[r, gi[g]] = 1.0 / len(gs)
    p = np.array([profile.get(g, 0.0) for g in genres])
    mask = p > 0
    key = np.array([c.final for c in cands])
    rel = (key - key.min()) / (key.max() - key.min() + 1e-9)
    counts = np.zeros(len(genres))
    aged = ages is not None and mu > 0
    pa = ages.share if ages is not None else np.ones(1)
    A = np.zeros((len(cands), len(pa)))
    acounts = np.zeros(len(pa))
    if aged and ages is not None:
        for r, c in enumerate(cands):
            b = ages.bucket(c.mal_id)
            if b is not None:
                A[r, b] = 1.0
    left = np.ones(len(cands), bool)
    out, total = [], 0.0
    for _ in range(min(limit, len(cands))):
        idx = np.nonzero(left)[0]
        C = counts + G[idx]
        tot = C.sum(1, keepdims=True)
        q = np.divide(C, tot, out=np.zeros_like(C), where=tot > 0)
        qt = (1 - CALIB_ALPHA) * q + CALIB_ALPHA * p
        kl = (p[mask] * np.log(p[mask] / qt[:, mask])).sum(1) if mask.any() else 0.0
        gain = (1 - lam) * (total + rel[idx]) - lam * kl
        if aged:
            gain = gain - mu * _kl(pa, acounts + A[idx])
        b = idx[int(np.argmax(gain))]
        out.append(cands[b]); left[b] = False
        total += rel[b]; counts += G[b]
        acounts += A[b]
    # The calibration decides WHICH titles make the list; their ORDER is the
    # ranking key's. In selection order the first picks are made almost
    # purely on genre fit (a 6.8 title led a list of 8s); exp_calib.py
    # measured the set (recall@50), which this keeps. Sorting the whole list
    # by the key, though, puts whatever the key favours at the top whatever
    # the mix: a back-catalogue watcher's 60 had their age mix, the first 20
    # were all this year's. With calibration_block the key orders each block
    # of that many picks - the greedy picks are calibrated as a prefix, so
    # every block keeps the mix and the best of each block leads it. Only
    # for lists held to an age mix: the rest (This Season) was not measured.
    block = (settings().calibration_block if ages is not None and mu > 0 else 0) or len(out)
    return [c for k in range(0, len(out), block)
            for c in sorted(out[k:k + block], key=lambda c: -c.final)]


def _kl(p: np.ndarray, C: np.ndarray) -> np.ndarray:
    """KL(p || q) for each row of counts C, q smoothed toward p by CALIB_ALPHA."""
    tot = C.sum(1, keepdims=True)
    q = np.divide(C, tot, out=np.zeros_like(C), where=tot > 0)
    qt = (1 - CALIB_ALPHA) * q + CALIB_ALPHA * p
    m = p > 0
    return (p[m] * np.log(p[m] / qt[:, m])).sum(1) if m.any() else np.zeros(len(C))


# --------------------------------------------------------------- explanation --

# An anime can be linked by both providers, which would otherwise list the
# same "because you rated X" twice; collapse to the strongest edge per source.
WHY_SQL = """
    SELECT a.title, le.score, max(e.weight) AS weight
      FROM rec_edge e
      JOIN list_entry le ON le.mal_id = e.dst AND le.user_id = %(uid)s AND le.score > 0
      JOIN anime a ON a.mal_id = e.dst
     WHERE e.src = %(mid)s
     GROUP BY e.dst, a.title, le.score
     ORDER BY max(e.weight) * (le.score - %(mu)s) DESC
     LIMIT 3
"""

TAGS_SQL = """
    SELECT tag FROM anime_tag WHERE mal_id = %s AND rank >= 60 ORDER BY rank DESC LIMIT 4
"""


def explain(user_id: int, cand: Candidate, user_mean: float) -> list[dict]:
    """Human-readable drivers. The graph is a poor ranker but an excellent
    explainer - "because you rated Steins;Gate 10" is exactly the right reason
    to show, even though it is not what produced the ordering."""
    reasons: list[dict] = []
    for r in query(WHY_SQL, {"uid": user_id, "mid": cand.mal_id, "mu": user_mean}):
        if (r["score"] or 0) > user_mean:
            reasons.append({"kind": "because_you_liked", "title": r["title"],
                            "your_score": r["score"]})
    tags = [r["tag"] for r in query(TAGS_SQL, (cand.mal_id,))]
    if tags:
        reasons.append({"kind": "tags", "tags": tags})
    if cand.row.get("franchise_known"):
        reasons.append({"kind": "same_franchise"})
    if cand.novelty > 0.85:
        reasons.append({"kind": "deep_cut"})
    return reasons


def attach_reasons(user_id: int, cands: list[Candidate], user_mean: float,
                   model: Model | None = None) -> None:
    """Graph-based reasons plus, when the model can decompose, what actually
    drove the score. Without the latter a recommendation with no neighbour in
    the user's list shows no explanation at all, which is the case for most
    content-driven picks."""
    from .recsys.scorer import HybridScorer
    if isinstance(model, HybridScorer):
        explained = model.explain([c.mal_id for c in cands], [c.novelty for c in cands],
                                  [c.relevance for c in cands])
        for c, reasons in zip(cands, explained):
            c.reasons = reasons
        attach_evidence(user_id, cands)
        return
    baseline = None
    if model is not None:
        rows = [c.row for c in cands if c.row]
        if rows:
            m = contribution_matrix(model, rows)
            if m is not None:
                baseline = m.mean(axis=0)
    for c in cands:
        c.reasons = explain(user_id, c, user_mean)
        if model is not None and c.row:
            drivers = model_drivers(model, c.row, baseline=baseline)
            if drivers:
                c.reasons.append(drivers)


def attach_evidence(user_id: int, cands: list[Candidate]) -> None:
    """What the user's own ratings clearly say about each title's attributes
    (taste.evidence_many): the card's personal reasons, the same lines as in
    "Why this?" - not the personal model's terms, which often said the
    opposite (experiments/exp_evidence.py)."""
    from . import taste
    ev = taste.evidence_many(user_id, [c.mal_id for c in cands], fresh=True)
    for c in cands:
        if ev.get(c.mal_id):
            c.reasons.append({"kind": "evidence", "items": ev[c.mal_id]})


# ------------------------------------------------------- model attribution --
#
# A ridge prediction decomposes exactly: it is the intercept plus one
# coefficient-times-value term per feature. That makes it possible to say what
# actually drove a recommendation rather than guessing, and to distinguish a
# genuinely personal pick from "this is simply a well-regarded show".

# Features that say nothing about this particular user.
CONSENSUS_FEATURES = {"mal_mean", "al_score", "mal_log_scorers", "al_log_favs", "mal_log_pop",
                      "drop_c", "polar_c"}
# Features that come from what the user has already watched and rated.
AFFINITY_FEATURES = {"aff_mal", "aff_anilist", "franchise_best", "franchise_known", "tag_cosine"}

_FIXED_LABELS = {
    "mal_mean": "highly rated on MyAnimeList",
    "al_score": "highly rated on AniList",
    "mal_log_scorers": "widely watched",
    "al_log_favs": "a favourite for many people",
    # mal_log_pop is sign-dependent, so it is resolved in _label() instead.
    "aff_mal": "recommended alongside shows you rate highly",
    "aff_anilist": "recommended alongside shows you rate highly",
    "franchise_best": "from a franchise you already like",
    "franchise_known": "from a franchise you already know",
    "tag_cosine": "matches your tag profile",
    "staff_aff": "by people whose work you rate highly",
}

_SOURCE_LABELS = {"manga": "manga adaptation", "light_novel": "light-novel adaptation",
                  "original": "an original story", "visual_novel": "visual-novel adaptation",
                  "web_manga": "web-manga adaptation", "game": "game adaptation",
                  "novel": "novel adaptation", "4_koma_manga": "4-koma adaptation"}
_TYPE_LABELS = {"tv": "a TV series", "movie": "a film", "ona": "an ONA",
                "ova": "an OVA", "special": "a special"}


def _label(name: str, value: float = 0.0) -> str | None:
    """Turn an internal feature name into something worth showing a person.

    `value` matters for popularity: MAL's figure is a *rank*, so a small number
    means a very popular show. Labelling it without checking the sign claimed
    "under the radar" for titles inside the global top 100.
    """
    if name == "mal_log_pop":
        return "a popular title" if value < 0 else "under the radar"
    if name == "drop_c":
        return "rarely dropped" if value < 0 else "often dropped"
    if name == "polar_c":
        return "divides opinion" if value > 0 else "broadly liked"
    for prefix, verb in (("dir:", "directed by"), ("wri:", "written by"),
                         ("orig:", "based on the work of"), ("mus:", "music by")):
        if name.startswith(prefix):
            return f"{verb} {name[len(prefix):]}"
    if name in _FIXED_LABELS:
        return _FIXED_LABELS[name]
    if name.startswith("g:"):
        return name[2:]
    if name.startswith("st:"):
        return f"animated by {name[3:]}"
    if name.startswith("src:"):
        return _SOURCE_LABELS.get(name[4:])
    if name.startswith("mt:"):
        return _TYPE_LABELS.get(name[3:])
    if name.startswith("len:"):
        return f"{name[4:]} episodes"
    if name.startswith("dec:"):                 # five-year buckets (features._cat_tokens)
        y = int(name[4:])
        return f"{y}–{y + 4}"
    if name.startswith("rt:"):
        return None                     # age rating is noise in an explanation
    return name                         # AniList tags are already readable


def contribution_matrix(model: TasteModel, rows: list[dict]) -> np.ndarray | None:
    """coefficient x value for every candidate, one row each."""
    coef = getattr(model.estimator, "coef_", None)
    if coef is None:
        return None                     # tree models do not decompose this way
    X = vectorise(rows, model.vocab)
    return X * np.asarray(coef, dtype=float)


def model_drivers(model: TasteModel, row: dict, top: int = 4,
                  baseline: np.ndarray | None = None) -> dict | None:
    """What drove this prediction, expressed as what makes it *distinctive*.

    Raw contributions are useless as an explanation: "highly rated on
    MyAnimeList" is the largest term for nearly every candidate, so it appears
    on every card and differentiates nothing. Ranking instead by how far each
    term sits above the average candidate surfaces the reasons that actually
    single this title out.
    """
    contrib = contribution_matrix(model, [row])
    if contrib is None:
        return None
    contrib = contrib[0]
    raw_values = vectorise([row], model.vocab)[0]
    names = model.vocab.names

    total = float(np.abs(contrib).sum())
    if total < 1e-9:
        return None
    acclaim = float(sum(abs(contrib[i]) for i, n in enumerate(names)
                        if n in CONSENSUS_FEATURES))
    personal = 1.0 - acclaim / total

    distinctive = contrib if baseline is None else contrib - baseline
    items = []
    for i in np.argsort(-distinctive):
        if distinctive[i] <= 0.01 or len(items) >= top:
            break
        label = _label(names[i], float(raw_values[i]))
        if label:
            items.append({"label": label, "weight": round(float(contrib[i]), 3),
                          "personal": names[i] not in CONSENSUS_FEATURES})
    if not items:
        return None
    # Only the graph and franchise signals count as "we know this about you";
    # tag_cosine is almost always faintly non-zero and would make the flag
    # meaningless.
    strong = {"aff_mal", "aff_anilist", "franchise_best"}
    return {"kind": "drivers", "items": items,
            "personal_share": round(personal, 3),
            "has_affinity": any(abs(contrib[i]) > 0.02 for i, n in enumerate(names)
                                if n in strong)}
