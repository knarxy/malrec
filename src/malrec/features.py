"""Feature assembly.

Everything the ranker knows about an anime, built for the user's rated set
(to train on) and for the candidate set (to score) through the same code path,
so a feature can never mean two different things on the two sides.

Feature groups, in the order they are concatenated:
    cat   MAL genres, studios, source, media type, rating, length, era
    tag   AniList weighted tags (value = relevance/100, not binary)
    num   consensus signals from both sites
    aff   recency-weighted affinity to what the user already rated
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .config import settings
from .db import query

log = logging.getLogger(__name__)

MIN_CAT_COUNT = 3       # a feature seen once is memorisation, not signal
MIN_TAG_RANK = 40

# The creative roles that plausibly carry taste: who directed it, who shaped
# the story, whose work it adapts. (Music is kept for tokens only.)
KEY_STAFF_ROLES = ("Director", "Series Composition", "Original Creator", "Script", "Music")
_STAFF_PREFIX = {"Director": "dir", "Series Composition": "wri", "Script": "wri",
                 "Original Creator": "orig", "Music": "mus"}
# roles whose people feed the staff-affinity signal
AFFINITY_ROLES = {"Director", "Series Composition", "Original Creator"}


def audience_stats(mal_drop: float | None, al_drop: float | None,
                   al_score_dist: dict | None) -> dict:
    """Two audience facts, centred and scaled to roughly unit spread over the
    candidate pool (medians: drop rate 4.4%, score sd 1.78 points).

    drop_c   log drop rate; MAL's figure (larger sample) with AniList's as the
             fallback - the two correlate at 0.98
    polar_c  standard deviation of AniList's score distribution: how divided
             opinion is, which the mean alone hides
    """
    drop = mal_drop if mal_drop is not None else al_drop
    drop_c = (math.log10(drop + 0.01) + 1.27) / 0.4 if drop is not None else 0.0
    polar_c = 0.0
    if al_score_dist:
        pts = [(int(k) / 10.0, int(v)) for k, v in al_score_dist.items()]
        n = sum(v for _, v in pts)
        if n >= 50:
            m = sum(k * v for k, v in pts) / n
            polar_c = (math.sqrt(sum(v * (k - m) ** 2 for k, v in pts) / n) - 1.78) / 0.3
    return {"drop_c": drop_c, "polar_c": polar_c}


@dataclass
class Vocabulary:
    cats: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # optional numeric columns after the fixed block (EXTRA_NUMERIC keys);
    # empty for artefacts stored before they existed, which keeps them valid
    extra: tuple[str, ...] = ()

    @property
    def names(self) -> list[str]:
        return (self.cats + self.tags +
                ["mal_mean", "mal_log_pop", "mal_log_scorers", "al_score", "al_log_favs",
                 "aff_mal", "aff_anilist", "franchise_best", "franchise_known",
                 "tag_cosine"] + list(self.extra))

    @property
    def width(self) -> int:
        return len(self.cats) + len(self.tags) + 10 + len(self.extra)


# numeric row keys a vocabulary may carry as extra columns
EXTRA_NUMERIC = ("drop_c", "polar_c", "staff_aff", "age_y")


def extra_columns() -> tuple[str, ...]:
    cfg = settings()
    out = []
    if cfg.content_audience:
        out += ["drop_c", "polar_c"]
    if cfg.content_staff_aff:
        out.append("staff_aff")
    if cfg.level_trend:
        out.append("age_y")          # 0 for anything predicted, see hybrid.fit_user
    return tuple(out)


def _staff_tokens(r: dict) -> list[str]:
    return sorted({f"{_STAFF_PREFIX[role]}:{name}"
                   for role, name in zip(r.get("staff_roles") or [], r.get("staff_names") or [])
                   if role in _STAFF_PREFIX and name})


def _cat_tokens(r: dict) -> list[str]:
    toks = [f"g:{g}" for g in (r.get("mal_genres") or [])]
    toks += [f"st:{s}" for s in (r.get("mal_studios") or [])]
    toks += [f"src:{r.get('source')}", f"mt:{r.get('media_type')}", f"rt:{r.get('rating')}"]
    e = r.get("num_episodes") or 0
    toks.append("len:" + ("1-13" if e <= 13 else "14-26" if e <= 26
                          else "27-60" if e <= 60 else "60+"))
    if r.get("season_year"):
        toks.append(f"dec:{int(r['season_year']) // 5 * 5}")
    if settings().content_staff:
        toks += _staff_tokens(r)
    return toks


CATALOG_SQL = """
    SELECT a.mal_id, a.title, a.mal_genres, a.mal_studios, a.source, a.media_type,
           a.rating, a.num_episodes, a.season_year, a.mal_mean, a.mal_popularity,
           a.mal_num_scoring_users, a.al_average_score, a.al_favourites,
           coalesce(f.franchise_id, a.mal_id) AS franchise_id,
           CASE WHEN a.tag_vec IS NULL OR tv.v IS NULL THEN 0.0
                ELSE 1.0 - (a.tag_vec <=> tv.v) END AS tag_cosine,
           coalesce(am.aff, 0)  AS aff_mal,
           coalesce(aa.aff, 0)  AS aff_anilist,
           coalesce(fa.best_delta, 0) AS franchise_best,
           coalesce(fa.has_history, false) AS franchise_known,
           coalesce(tg.tags, '{}'::text[])  AS tag_names,
           coalesce(tg.ranks, '{}'::int[])  AS tag_ranks,
           a.mal_drop_rate, a.al_drop_rate, a.al_score_dist,
           coalesce(sf.roles, '{}'::text[]) AS staff_roles,
           coalesce(sf.names, '{}'::text[]) AS staff_names
      FROM anime a
      LEFT JOIN franchise f ON f.mal_id = a.mal_id
      CROSS JOIN LATERAL (SELECT %(taste)s::vector AS v) tv
      LEFT JOIN (SELECT mal_id, affinity AS aff FROM rec_affinity(%(uid)s, 'mal', 1.5::real, %(hl)s::real)) am
             ON am.mal_id = a.mal_id
      LEFT JOIN (SELECT mal_id, affinity AS aff FROM rec_affinity(%(uid)s, 'anilist', 1.5::real, %(hl)s::real)) aa
             ON aa.mal_id = a.mal_id
      LEFT JOIN (SELECT mal_id, best_delta, has_history FROM franchise_affinity(%(uid)s, %(hl)s::real)) fa
             ON fa.mal_id = a.mal_id
      LEFT JOIN (SELECT mal_id, array_agg(tag) AS tags, array_agg(rank) AS ranks
                   FROM anime_tag WHERE rank >= %(minrank)s GROUP BY mal_id) tg
             ON tg.mal_id = a.mal_id
      LEFT JOIN (SELECT mal_id, array_agg(role ORDER BY role, staff_id) AS roles,
                        array_agg(name ORDER BY role, staff_id) AS names
                   FROM anime_staff WHERE role = ANY(%(roles)s) GROUP BY mal_id) sf
             ON sf.mal_id = a.mal_id
     WHERE a.mal_id = ANY(%(ids)s)
"""


def load_rows(user_id: int, mal_ids: list[int], half_life: float | None = None) -> dict[int, dict]:
    """One query pulls catalog columns plus every user-dependent aggregate.

    The affinity functions run once for the whole set rather than per anime,
    which is the difference between a second and several minutes.
    """
    if not mal_ids:
        return {}
    hl = settings().recency_half_life_years if half_life is None else half_life
    taste = query("SELECT user_taste_vector(%s, %s::real) AS v", (user_id, hl))[0]["v"]
    rows = query(CATALOG_SQL, {"uid": user_id, "ids": mal_ids, "hl": hl,
                               "taste": taste, "minrank": MIN_TAG_RANK,
                               "roles": list(KEY_STAFF_ROLES)})
    for r in rows:
        r.update(audience_stats(r.pop("mal_drop_rate"), r.pop("al_drop_rate"),
                                r.pop("al_score_dist")))
        # staff affinity needs population deviations; the hybrid path
        # (recsys.items.ItemStore) computes it, this legacy path does not
        r["staff_aff"] = 0.0
    return {r["mal_id"]: r for r in rows}


def build_vocabulary(rows: list[dict]) -> Vocabulary:
    """Fit only on the training rows, so a candidate cannot introduce a column
    the model was never trained on."""
    cc = Counter(t for r in rows for t in _cat_tokens(r))
    tc = Counter(t for r in rows for t in (r.get("tag_names") or []))
    return Vocabulary(
        cats=sorted(t for t, n in cc.items() if n >= MIN_CAT_COUNT),
        tags=sorted(t for t, n in tc.items() if n >= MIN_CAT_COUNT),
        extra=extra_columns(),
    )


def vectorise(rows: list[dict], vocab: Vocabulary) -> np.ndarray:
    """The affinity columns stay in the layout even when disabled, so a model
    trained either way keeps a stable feature index and stored artefacts do
    not silently misalign."""
    use_aff = settings().use_affinity_features
    ci = {t: i for i, t in enumerate(vocab.cats)}
    ti = {t: i for i, t in enumerate(vocab.tags)}
    off_tag = len(vocab.cats)
    off_num = off_tag + len(vocab.tags)
    X = np.zeros((len(rows), vocab.width), dtype=np.float64)

    for i, r in enumerate(rows):
        for t in _cat_tokens(r):
            j = ci.get(t)
            if j is not None:
                X[i, j] = 1.0
        names = r.get("tag_names") or []
        ranks = r.get("tag_ranks") or []
        for name, rank in zip(names, ranks):
            j = ti.get(name)
            if j is not None:
                X[i, off_tag + j] = rank / 100.0
        X[i, off_num + 0] = (r.get("mal_mean") or 7.5) - 7.5
        X[i, off_num + 1] = math.log10(max(r.get("mal_popularity") or 5000, 1)) - 3
        X[i, off_num + 2] = math.log10(max(r.get("mal_num_scoring_users") or 1000, 1)) - 4
        X[i, off_num + 3] = ((r.get("al_average_score") or 75) - 75) / 10.0
        X[i, off_num + 4] = math.log10(max(r.get("al_favourites") or 10, 1)) - 2
        if use_aff:
            X[i, off_num + 5] = r.get("aff_mal") or 0.0
            X[i, off_num + 6] = r.get("aff_anilist") or 0.0
            X[i, off_num + 7] = r.get("franchise_best") or 0.0
            X[i, off_num + 8] = 1.0 if r.get("franchise_known") else 0.0
        X[i, off_num + 9] = r.get("tag_cosine") or 0.0
        for k, key in enumerate(vocab.extra):
            X[i, off_num + 10 + k] = r.get(key) or 0.0
    return X


IMPLICIT_SQL = """
    SELECT le.mal_id, le.status,
           recency_weight(le.finished_at, le.updated_at, %(hl)s::real, %(floor)s::real) AS w,
           coalesce(le.finished_at::timestamptz, le.updated_at, now()) AS at
      FROM list_entry le
     WHERE le.user_id = %(uid)s AND le.score = 0 AND le.status = ANY(%(statuses)s)
     ORDER BY le.mal_id
"""


def implicit_rows(user_id: int, mean: float, cutoff=None,
                  half_life: float | None = None) -> list[dict]:
    """Unscored entries turned into weak pseudo-ratings.

    Dropping a show is an opinion, and so is stalling on one or finishing it
    without rating it. Each becomes a training row at an offset from the
    user's mean, carrying a fraction of a real rating's weight.

    `cutoff` exists for the temporal holdout: an entry the user only touched
    after the cutoff was not knowable at prediction time and must be excluded,
    or the evaluation leaks the future.
    """
    cfg = settings()
    hl = cfg.recency_half_life_years if half_life is None else half_life
    offsets = cfg.implicit_offsets
    if not offsets or cfg.implicit_weight <= 0:
        return []
    rows = query(IMPLICIT_SQL, {"uid": user_id, "hl": hl, "floor": cfg.recency_floor,
                                "statuses": list(offsets)})
    out = []
    for r in rows:
        if cutoff is not None and r["at"] >= cutoff:
            continue
        out.append({
            "mal_id": r["mal_id"],
            "score": float(min(max(mean + offsets[r["status"]], 1.0), 10.0)),
            "w": float(r["w"]) * cfg.implicit_weight,
        })
    return out


def training_set(user_id: int, half_life: float | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Vocabulary, list[int]]:
    """Returns (X, y, sample_weight, vocab, mal_ids) for the user's rated anime.

    sample_weight is the recency decay: a rating from last month tells us more
    about what to recommend tonight than one from five years ago. Measured on a
    temporal holdout, this lifts Spearman from ~0.65 to ~0.77.
    """
    hl = settings().recency_half_life_years if half_life is None else half_life
    scored = query(
        """
        SELECT le.mal_id, le.score,
               recency_weight(le.finished_at, le.updated_at, %s::real, %s::real) AS w
          FROM list_entry le
         WHERE le.user_id = %s AND le.score > 0
         ORDER BY le.mal_id
        """,
        (hl, settings().recency_floor, user_id),
    )
    if not scored:
        raise ValueError(f"user {user_id} has no scored entries to learn from")

    mean = float(np.average([r["score"] for r in scored],
                            weights=[max(float(r["w"]), 1e-6) for r in scored]))
    examples = list(scored) + implicit_rows(user_id, mean, half_life=hl)

    ids = [r["mal_id"] for r in examples]
    rows_by_id = load_rows(user_id, ids, half_life=hl)
    keep = [r for r in examples if r["mal_id"] in rows_by_id]
    rows = [rows_by_id[r["mal_id"]] for r in keep]
    vocab = build_vocabulary(rows)
    X = vectorise(rows, vocab)
    y = np.array([float(r["score"]) for r in keep])
    w = np.array([float(r["w"]) for r in keep])
    log.info("training set: %d rated + %d implicit, %d features, effective n=%.0f",
             len(scored), len(keep) - len(scored), vocab.width,
             w.sum() ** 2 / (w ** 2).sum())
    return X, y, w, vocab, [r["mal_id"] for r in keep]
