"""Static per-anime data held in memory.

The SQL feature path (features.load_rows) is tied to users stored in
list_entry, so it cannot score the thousands of sampled MAL users needed to
evaluate the model properly, and it re-derives item facts on every call.
This loads every item's static facts once and builds row dicts with exactly
the keys load_rows returns, so features.vectorise() - the one feature
definition - is shared and cannot drift.

The only user-dependent column in those rows is tag_cosine; `taste_vector`
reproduces the SQL user_taste_vector() for it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..db import query
from ..features import KEY_STAFF_ROLES, MIN_TAG_RANK, audience_stats

log = logging.getLogger(__name__)

STATIC_SQL = """
    SELECT a.mal_id, a.title, a.mal_genres, a.mal_studios, a.source, a.media_type,
           a.rating, a.num_episodes, a.season_year, a.mal_mean, a.mal_popularity,
           a.mal_num_scoring_users, a.al_average_score, a.al_favourites, a.nsfw,
           a.status, format_class(a.media_type) AS format_class,
           coalesce(f.franchise_id, a.mal_id) AS franchise_id,
           a.tag_vec::text AS tag_vec,
           coalesce(tg.tags, '{}'::text[]) AS tag_names,
           coalesce(tg.ranks, '{}'::int[]) AS tag_ranks,
           a.mal_drop_rate, a.al_drop_rate, a.al_score_dist,
           coalesce(sf.ids, '{}'::int[]) AS staff_ids,
           coalesce(sf.roles, '{}'::text[]) AS staff_roles,
           coalesce(sf.names, '{}'::text[]) AS staff_names
      FROM anime a
      LEFT JOIN franchise f ON f.mal_id = a.mal_id
      LEFT JOIN (SELECT mal_id, array_agg(tag ORDER BY tag) AS tags,
                        array_agg(rank ORDER BY tag) AS ranks
                   FROM anime_tag WHERE rank >= %(minrank)s GROUP BY mal_id) tg
             ON tg.mal_id = a.mal_id
      LEFT JOIN (SELECT mal_id, array_agg(staff_id ORDER BY role, staff_id) AS ids,
                        array_agg(role ORDER BY role, staff_id) AS roles,
                        array_agg(name ORDER BY role, staff_id) AS names
                   FROM anime_staff WHERE role = ANY(%(roles)s) GROUP BY mal_id) sf
             ON sf.mal_id = a.mal_id
"""

# Keys of a load_rows() row that depend on the user and are filled per user.
_USER_KEYS = ("tag_cosine", "aff_mal", "aff_anilist", "franchise_best", "franchise_known",
              "staff_aff")


@dataclass
class ItemStore:
    rows: dict[int, dict]                    # mal_id -> static row
    vec_ids: np.ndarray                      # items that have a tag vector
    vecs: np.ndarray                         # (n, 256) L2-normalised tag vectors
    vec_index: dict[int, int] = field(default_factory=dict)
    staff: sp.csr_matrix | None = None       # items x key staff, binary
    staff_row: dict[int, int] = field(default_factory=dict)

    @classmethod
    def load(cls, ids: list[int] | None = None) -> ItemStore:
        sql = STATIC_SQL + (" WHERE a.mal_id = ANY(%(ids)s)" if ids is not None else "")
        rows = {}
        vec_ids, vecs = [], []
        for r in query(sql, {"minrank": MIN_TAG_RANK, "ids": ids,
                             "roles": list(KEY_STAFF_ROLES)}):
            tv = r.pop("tag_vec")
            if tv:
                vec_ids.append(r["mal_id"])
                vecs.append(np.fromstring(tv.strip("[]"), sep=",", dtype=np.float32))
            r.update(audience_stats(r.pop("mal_drop_rate"), r.pop("al_drop_rate"),
                                    r.pop("al_score_dist")))
            rows[r["mal_id"]] = r
        v = np.vstack(vecs) if vecs else np.zeros((0, 256), dtype=np.float32)
        store = cls(rows=rows, vec_ids=np.array(vec_ids), vecs=v,
                    vec_index={m: i for i, m in enumerate(vec_ids)})
        store._index_staff()
        log.info("item store: %d anime, %d with tag vectors", len(rows), len(vec_ids))
        return store

    def _index_staff(self) -> None:
        """Items x people matrix over the key creative roles, for staff_signal."""
        mids = [m for m, r in self.rows.items() if r.get("staff_ids")]
        people = sorted({p for m in mids for p in self.rows[m]["staff_ids"]})
        col = {p: k for k, p in enumerate(people)}
        ri, ci = [], []
        for k, m in enumerate(mids):
            for p in set(self.rows[m]["staff_ids"]):
                ri.append(k); ci.append(col[p])
        self.staff = sp.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(mids), len(people)))
        self.staff_row = {m: k for k, m in enumerate(mids)}

    def staff_signal(self, rated: list[int], dev: np.ndarray, w: np.ndarray,
                     items: list[int], shrink: float = 2.0) -> np.ndarray:
        """How the user rated other work by the same key people.

        For each item: the weighted mean of the user's deviations over the rated
        titles that share a director, series writer or original creator with
        it, shrunk toward 0 by `shrink` pseudo-titles. `dev` is whatever the
        caller considers a deviation (bias-free residual, or score minus mean).
        An item's own rating never counts toward itself, so the value is safe
        to use on training rows.
        """
        out = np.zeros(len(items))
        if self.staff is None or not len(rated):
            return out
        rk = [(k, self.staff_row[m]) for k, m in enumerate(rated) if m in self.staff_row]
        ik = [(k, self.staff_row[m]) for k, m in enumerate(items) if m in self.staff_row]
        if not rk or not ik:
            return out
        r_pos, r_rows = zip(*rk)
        i_pos, i_rows = zip(*ik)
        C = (self.staff[list(i_rows)] @ self.staff[list(r_rows)].T).tocoo()
        rated_ids = np.array([rated[k] for k in r_pos])
        item_ids = np.array([items[k] for k in i_pos])
        keep = item_ids[C.row] != rated_ids[C.col]          # leave one out
        rows, cols = C.row[keep], C.col[keep]
        wv = np.asarray(w, dtype=float)[list(r_pos)]
        dv = np.asarray(dev, dtype=float)[list(r_pos)]
        num = np.bincount(rows, weights=wv[cols] * dv[cols], minlength=len(ik))
        den = np.bincount(rows, weights=wv[cols], minlength=len(ik))
        tot = den + shrink
        out[list(i_pos)] = np.divide(num, tot, out=np.zeros_like(num), where=tot > 0)
        return out

    def taste_vector(self, ratings: dict[int, float], weights: dict[int, float]) -> np.ndarray | None:
        """numpy twin of SQL user_taste_vector(): recency-weighted,
        mean-centred sum of the rated items' tag vectors, L2-normalised."""
        ids = [m for m in ratings if m in self.vec_index]
        if not ids:
            return None
        w_all = np.array([weights[m] for m in ratings])
        s_all = np.array([ratings[m] for m in ratings])
        mu = float((w_all * s_all).sum() / max(w_all.sum(), 1e-9))
        coef = np.array([(ratings[m] - mu) * weights[m] for m in ids], dtype=np.float64)
        if np.abs(coef).sum() == 0:
            return None
        v = coef @ self.vecs[[self.vec_index[m] for m in ids]].astype(np.float64)
        n = np.linalg.norm(v)
        return (v / n) if n > 0 else None

    def user_rows(self, ids: list[int], taste: np.ndarray | None,
                  staff_aff: np.ndarray | None = None) -> list[dict]:
        """load_rows()-compatible rows for these items, for one user.
        `staff_aff` is staff_signal() for exactly `ids`, when the caller uses it."""
        out = []
        for k, m in enumerate(ids):
            base = self.rows.get(m)
            if base is None:
                continue
            row = dict(base)
            for key in _USER_KEYS:
                row[key] = 0.0
            row["franchise_known"] = False
            if taste is not None and m in self.vec_index:
                row["tag_cosine"] = float(self.vecs[self.vec_index[m]] @ taste)
            if staff_aff is not None:
                row["staff_aff"] = float(staff_aff[k])
            out.append(row)
        return out
