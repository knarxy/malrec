"""Population model learned from the sampled MAL lists.

Three signals, each computed on ratings centred by the rater's own mean (a
7 from someone who gives out 9s is a complaint, from someone who averages 5
it is praise):

  bias    b_i   how far above their own mean people rate item i
                (shrunk toward 0 when few people have rated it)
  knn           item-item similarity from real co-ratings: "people who rated
                j above their usual rated i above theirs too"
  mf            low-rank matrix factorisation of what is left after the bias

None of them needs the user to be in the training data. For any user - an
app user, or a sampled one held out for evaluation - `fold_in` derives their
position from their own ratings alone, which is what makes this useful for
someone with eleven ratings.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

log = logging.getLogger(__name__)


@dataclass
class CFParams:
    min_item_raters: int = 15     # items rated by fewer are left to the content model
    bias_shrink: float = 10.0     # pseudo-raters pulling b_i toward 0
    sim_shrink: float = 25.0      # co-raters needed before a similarity is trusted
    knn_k: int = 60               # neighbours kept per item
    knn_shrink: float = 1.0       # damping in the kNN weighted average
    mf_rank: int = 32
    mf_reg: float = 8.0
    mf_iters: int = 12


class PopulationModel:
    def __init__(self, params: CFParams | None = None):
        self.p = params or CFParams()
        self.items: np.ndarray = np.zeros(0, dtype=np.int64)   # index -> mal_id
        self.index: dict[int, int] = {}
        self.bias: np.ndarray = np.zeros(0)
        self.n_raters: np.ndarray = np.zeros(0, dtype=np.int64)
        self.sim: sp.csr_matrix | None = None                  # items x items, top-K
        self.occ: sp.csr_matrix | None = None                  # list co-occurrence, top-K
        self.V: np.ndarray = np.zeros((0, 0))                  # item factors
        self.ease: sp.csc_matrix | None = None                 # EASE weights, top-K per item
        self.ease_index: dict[int, int] = {}

    # ------------------------------------------------------------------ fit --

    def fit(self, user_ids: np.ndarray, item_ids: np.ndarray, scores: np.ndarray
            ) -> PopulationModel:
        p = self.p
        # keep items with enough raters to say anything
        uniq, counts = np.unique(item_ids, return_counts=True)
        keep = uniq[counts >= p.min_item_raters]
        mask = np.isin(item_ids, keep)
        u_raw, i_raw, r = user_ids[mask], item_ids[mask], scores[mask].astype(np.float64)
        self.items = keep
        self.index = {int(m): k for k, m in enumerate(keep)}
        u_uniq, u = np.unique(u_raw, return_inverse=True)
        i = np.searchsorted(keep, i_raw)
        n_u, n_i = len(u_uniq), len(keep)

        mu = np.bincount(u, weights=r, minlength=n_u) / np.maximum(np.bincount(u, minlength=n_u), 1)
        c = r - mu[u]
        self.n_raters = np.bincount(i, minlength=n_i)
        self.bias = np.bincount(i, weights=c, minlength=n_i) / (self.n_raters + p.bias_shrink)
        e = c - self.bias[i]

        E = sp.csr_matrix((e, (u, i)), shape=(n_u, n_i))
        self.sim = self._similarity(E, sp.csr_matrix((np.ones_like(e), (u, i)), shape=(n_u, n_i)))
        self.V = self._als(E, n_u, n_i)
        log.info("population model: %d users, %d items, %d ratings", n_u, n_i, len(r))
        return self

    def fit_occurrence(self, user_ids: np.ndarray, item_ids: np.ndarray) -> PopulationModel:
        """Which anime share *lists*, regardless of score.

        The rating signals answer "how much would they like it, given they
        watch it". They cannot answer "would they ever pick it up": a 1970s
        classic watched only by devotees has a glowing conditional rating, so
        ranking by it alone hands every newcomer the same list of devotee
        favourites. Co-occurrence of list entries - including plan-to-watch,
        dropped and unscored ones - measures whether a title sits in this
        user's viewing neighbourhood at all.
        """
        mask = np.isin(item_ids, self.items)
        u_raw, i_raw = user_ids[mask], item_ids[mask]
        pairs = np.unique(np.column_stack([u_raw, i_raw]), axis=0)
        u_uniq, u = np.unique(pairs[:, 0], return_inverse=True)
        i = np.searchsorted(self.items, pairs[:, 1])
        B = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(len(u_uniq), len(self.items)))
        self.occ = self._similarity(B, B)
        log.info("co-occurrence model: %d users, %d list entries", len(u_uniq), len(u))
        return self

    def fit_ease(self, user_ids: np.ndarray, item_ids: np.ndarray, min_users: int = 50,
                 lam: float = 400.0, top_k: int = 100) -> PopulationModel:
        """EASE (Steck, WWW 2019) on list membership: B = I - P / diag(P) with
        P = (X'X + lam I)^-1, so a user's score for j is sum_i x_i B_ij. A
        closed-form item-item model and one of the strongest "what will they
        watch" baselines; used as a second relevance signal. Each item keeps
        only its `top_k` strongest incoming weights (dense B would be ~360 MB).
        """
        pairs = np.unique(np.column_stack([user_ids, item_ids]), axis=0)
        items, counts = np.unique(pairs[:, 1], return_counts=True)
        keep = items[counts >= min_users]
        m = np.isin(pairs[:, 1], keep)
        u = np.unique(pairs[m, 0], return_inverse=True)[1]
        i = np.searchsorted(keep, pairs[m, 1])
        X = sp.csr_matrix((np.ones(len(u)), (u, i)), shape=(u.max() + 1, len(keep)))
        # Memory: ~10k items make each dense matrix ~0.8 GB, so the Gram matrix
        # is inverted and turned into B in place, and the top-K selection runs
        # in column blocks - peak ~2 full matrices instead of ~4.
        G = (X.T @ X).toarray()
        G[np.diag_indices_from(G)] += lam
        P = np.linalg.inv(G)
        del G
        d = np.diag(P).copy()
        P /= -d                          # B = -P / diag(P), column-wise
        P[np.diag_indices_from(P)] = 0.0
        n = P.shape[1]
        k = min(top_k, len(keep) - 1)
        rows_l, cols_l, vals_l = [], [], []
        for c0 in range(0, n, 1024):
            blk = P[:, c0:c0 + 1024]
            r = np.argpartition(-np.abs(blk), k, axis=0)[:k]
            cc = np.broadcast_to(np.arange(blk.shape[1]), r.shape)
            rows_l.append(r.ravel()); cols_l.append((cc + c0).ravel())
            vals_l.append(blk[r, cc].astype(np.float32).ravel())
        self.ease = sp.csc_matrix((np.concatenate(vals_l),
                                   (np.concatenate(rows_l), np.concatenate(cols_l))),
                                  shape=(n, n))
        del P
        self.ease_index = {int(x): n for n, x in enumerate(keep)}
        log.info("EASE: %d items, top-%d weights kept", len(keep), k)
        return self

    def _similarity(self, E: sp.csr_matrix, B: sp.csr_matrix) -> sp.csr_matrix:
        """Shrunk cosine over co-raters, top-K per item, computed in blocks so
        the full items x items matrix is never materialised."""
        p = self.p
        Ec, Bc = E.tocsc(), B.tocsc()
        norms = np.sqrt(np.asarray(Ec.multiply(Ec).sum(axis=0)).ravel()) + 1e-9
        n = E.shape[1]
        rows, cols, vals = [], [], []
        for start in range(0, n, 512):
            stop = min(start + 512, n)
            dot = (Ec[:, start:stop].T @ Ec).toarray()
            co = (Bc[:, start:stop].T @ Bc).toarray()
            s = dot / np.outer(norms[start:stop], norms) * (co / (co + p.sim_shrink))
            s[np.arange(stop - start), np.arange(start, stop)] = 0.0   # no self-similarity
            k = min(p.knn_k, n - 1)
            top = np.argpartition(-s, k, axis=1)[:, :k]
            for r_off, idx in enumerate(top):
                v = s[r_off, idx]
                good = v > 0
                rows.extend([start + r_off] * int(good.sum()))
                cols.extend(idx[good].tolist())
                vals.extend(v[good].tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    def _als(self, E: sp.csr_matrix, n_u: int, n_i: int) -> np.ndarray:
        """Alternating least squares on the bias-free residuals."""
        p = self.p
        # Each item's starting factors come from its own id, so a refit with
        # a few more users starts from the same place. (U needs no start: the
        # first sweep solves it from V.)
        U = np.zeros((n_u, p.mf_rank))
        V = np.stack([np.random.default_rng([0, int(m)]).normal(0, 0.1, p.mf_rank)
                      for m in self.items]) if n_i else np.zeros((0, p.mf_rank))
        Ecsr, Ecsc = E.tocsr(), E.tocsc()
        eye = p.mf_reg * np.eye(p.mf_rank)
        for _ in range(p.mf_iters):
            for uu in range(n_u):
                a, b = Ecsr.indptr[uu], Ecsr.indptr[uu + 1]
                if a == b:
                    continue
                Vj = V[Ecsr.indices[a:b]]
                U[uu] = np.linalg.solve(Vj.T @ Vj + eye, Vj.T @ Ecsr.data[a:b])
            for ii in range(n_i):
                a, b = Ecsc.indptr[ii], Ecsc.indptr[ii + 1]
                if a == b:
                    continue
                Uj = U[Ecsc.indices[a:b]]
                V[ii] = np.linalg.solve(Uj.T @ Uj + eye, Uj.T @ Ecsc.data[a:b])
        return V

    # -------------------------------------------------------------- fold-in --

    def fold_in(self, ratings: dict[int, float], weights: dict[int, float] | None = None,
                listed: dict[int, float] | None = None) -> FoldIn:
        """Place a user who was never in the training data, from their ratings.
        `listed` is every entry on their list (any status) with its weight,
        for the co-occurrence relevance signal; defaults to the rated ones."""
        weights = weights or {}
        listed = listed if listed is not None else {m: weights.get(m, 1.0) for m in ratings}
        lid = [m for m in listed if m in self.index]
        L = np.array([self.index[m] for m in lid], dtype=np.int64)
        lw = np.array([listed[m] for m in lid], dtype=np.float64)
        ids = [m for m in ratings if m in self.index]
        all_w = np.array([weights.get(m, 1.0) for m in ratings])
        all_r = np.array([ratings[m] for m in ratings], dtype=np.float64)
        mu = float((all_w * all_r).sum() / max(all_w.sum(), 1e-9)) if len(all_r) else 7.0
        if not ids:
            return FoldIn(self, mu, np.zeros(0, dtype=np.int64), np.zeros(0), np.zeros(0), None,
                          L, lw)
        J = np.array([self.index[m] for m in ids])
        w = np.array([weights.get(m, 1.0) for m in ids])
        d = np.array([ratings[m] for m in ids]) - mu - self.bias[J]
        Vj = self.V[J]
        pvec = np.linalg.solve((Vj.T * w) @ Vj + self.p.mf_reg * np.eye(self.p.mf_rank),
                               (Vj.T * w) @ d)
        return FoldIn(self, mu, J, d, w, pvec, L, lw)


@dataclass
class FoldIn:
    model: PopulationModel
    mu: float
    J: np.ndarray            # rated item indices known to the model
    d: np.ndarray            # their bias-free deviations
    w: np.ndarray            # their weights (recency)
    pvec: np.ndarray | None  # latent position
    L: np.ndarray = None     # every listed item index known to the model
    lw: np.ndarray = None    # their weights

    def signals(self, mal_ids: list[int]) -> dict[str, np.ndarray]:
        """Per-item bias, kNN deviation, MF deviation and kNN support for the
        requested items; zeros for items the population model does not know."""
        m = self.model
        n = len(mal_ids)
        idx = np.array([m.index.get(x, -1) for x in mal_ids])
        known = idx >= 0
        out = {k: np.zeros(n) for k in ("bias", "knn", "mf", "support", "known", "rel")}
        out["known"] = known.astype(float)
        if not known.any():
            return out
        ki = idx[known]
        out["bias"][known] = m.bias[ki]
        out["support"][known] = np.log1p(m.n_raters[ki])
        if len(self.J):
            S = m.sim[ki][:, self.J]                       # candidates x rated
            num = S @ (self.w * self.d)
            den = np.abs(S) @ self.w
            out["knn"][known] = num / (den + m.p.knn_shrink)
            if self.pvec is not None:
                out["mf"][known] = m.V[ki] @ self.pvec
        if m.occ is not None and self.L is not None and len(self.L):
            # weighted mean co-occurrence with the user's list
            out["rel"][known] = (m.occ[ki][:, self.L] @ self.lw) / max(self.lw.sum(), 1e-9)
        return out

    def evidence(self, mal_ids: list[int]) -> np.ndarray:
        """How much of the user's own rating history speaks to each item:
        the recency-weighted similarity mass of their rated neighbours (the
        kNN denominator). 0 means the population prediction for it rests on
        other people's opinions alone - item bias and consensus, which for
        titles watched mostly by devotees is optimistic."""
        m = self.model
        out = np.zeros(len(mal_ids))
        if not len(self.J):
            return out
        idx = np.array([m.index.get(x, -1) for x in mal_ids])
        known = idx >= 0
        if known.any():
            out[known] = np.abs(m.sim[idx[known]][:, self.J]) @ self.w
        return out

    def ease_scores(self, mal_ids: list[int]) -> np.ndarray | None:
        """EASE score of each item for this user's list (NaN if not modelled);
        None when the population model has no EASE part."""
        m = self.model
        E, idx = getattr(m, "ease", None), getattr(m, "ease_index", {})
        if E is None or self.L is None:
            return None
        x = np.zeros(E.shape[0], dtype=np.float32)
        for j, w in zip(self.L.tolist(), self.lw.tolist()):
            k = idx.get(int(m.items[j]))
            if k is not None:
                x[k] = max(w, 0.0)
        s = E.T @ x
        return np.array([s[idx[x_]] if x_ in idx else np.nan for x_ in mal_ids], dtype=float)


def knn_contributors(fold: FoldIn, mal_id: int, top: int = 3) -> list[tuple[int, float]]:
    """Which of the user's rated anime pushed this one up, via co-ratings.

    The kNN deviation is a weighted sum over the user's rated neighbours of
    the item; each term is that neighbour's contribution. Returned as
    (rated mal_id, contribution), positive contributions only - these are
    the honest "because you rated X" reasons.
    """
    m = fold.model
    i = m.index.get(mal_id)
    if i is None or not len(fold.J):
        return []
    row = m.sim[i]
    sims = dict(zip(row.indices.tolist(), row.data.tolist()))
    terms = [(int(m.items[j]), sims[j] * w * d)
             for j, w, d in zip(fold.J.tolist(), fold.w.tolist(), fold.d.tolist()) if j in sims]
    terms = [t for t in terms if t[1] > 0]
    terms.sort(key=lambda t: -t[1])
    return terms[:top]
