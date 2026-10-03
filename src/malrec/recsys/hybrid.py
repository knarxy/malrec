"""Hybrid taste model: population layer plus a personal residual.

    prediction(u, i) = stack(u, i) + residual_u(i)

stack      A small global model that blends the population signals for this
           user - item bias, item-kNN, matrix factorisation - with community
           consensus. Fitted once, across thousands of sampled MAL users, so
           its weights are stable. Every user gets a meaningful ranking from it,
           even one with ten ratings.

residual   The per-user content ridge the project started with, now fitted to
           what the stack gets *wrong* for this user rather than to raw scores.
           With few ratings, cross-validation picks heavy regularisation and it
           shrinks to nothing, so the stack carries the ranking. With many, it
           adds back the user's own quirks.

The old model failed small users in a specific way: with ten ratings its CV
chose alpha=1000, every prediction became the same constant, and the novelty
bonus alone decided the order - a Frieren fan got Precure. The stack cannot
collapse like that, because its signals come from other people's ratings.
"""
from __future__ import annotations

import datetime as dt
import io
import logging
import math
import pickle
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge

from ..features import Vocabulary, build_vocabulary, vectorise
from .cf import FoldIn, PopulationModel
from .items import ItemStore

log = logging.getLogger(__name__)

STACK_FEATURES = ("bias", "knn", "mf", "support", "known", "mal_c", "al_c", "pop",
                  "knn_x_support")
# Optional stacker inputs (config.stack_extras). A fitted stacker records the
# ones it was trained with in `extras_`, so a stored model always gets the
# columns it expects whatever the current config says.
STACK_EXTRAS = ("staff", "drop", "polar", "tag_cos")


def stacker_extras(stacker) -> tuple[str, ...]:
    return tuple(getattr(stacker, "extras_", ()))


def stack_matrix(sig: dict[str, np.ndarray], items: list[int], store: ItemStore,
                 extras: tuple[str, ...] = ()) -> np.ndarray:
    def val(m, key, default):
        return (store.rows.get(m) or {}).get(key) or default
    mal = np.array([val(m, "mal_mean", 7.5) - 7.5 for m in items])
    al = np.array([(val(m, "al_average_score", 75) - 75) / 10 for m in items])
    pop = np.array([math.log10(max(val(m, "mal_popularity", 5000), 1)) - 3 for m in items])
    cols = [sig["bias"], sig["knn"], sig["mf"], sig["support"], sig["known"],
            mal, al, pop, sig["knn"] * sig["support"]]
    for e in extras:
        if e == "drop":
            cols.append(np.array([val(m, "drop_c", 0.0) for m in items], dtype=float))
        elif e == "polar":
            cols.append(np.array([val(m, "polar_c", 0.0) for m in items], dtype=float))
        else:                                   # staff, tag_cos: user-dependent
            cols.append(sig[e])
    return np.column_stack(cols)


def extra_signals(sig: dict[str, np.ndarray], fold: FoldIn, store: ItemStore,
                  taste: np.ndarray | None, items: list[int],
                  extras: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Adds the user-dependent extras to a FoldIn.signals() dict."""
    if "staff" in extras:
        sig["staff"] = staff_affinity(fold, store, items)
    if "tag_cos" in extras:
        sig["tag_cos"] = np.array([float(store.vecs[store.vec_index[m]] @ taste)
                                   if taste is not None and m in store.vec_index else 0.0
                                   for m in items])
    return sig


def staff_affinity(fold: FoldIn, store: ItemStore, items: list[int]) -> np.ndarray:
    """store.staff_signal on the fold's bias-free deviations: how much more
    than the population expected the user liked other work by these people."""
    if not len(fold.J):
        return np.zeros(len(items))
    rated = fold.model.items[fold.J].tolist()
    return store.staff_signal(rated, fold.d, fold.w, items)


@dataclass
class Example:
    """One training row for a user: a real rating or an implicit pseudo-rating."""
    mal_id: int
    score: float
    weight: float
    at: object = None          # datetime; used to hold out the newest for blending


MODES = ("personal", "stack", "hierarchical", "augmented", "blend", "sized")


def size_weight(n: int, n0: float, scale: float) -> float:
    """How much of the prediction the personal model gets, from list size.

    Measured on sampled users, the population layer's advantage shrinks
    steadily as history grows (+0.21 rho at 10 ratings, +0.02 at 150), while
    on a long, idiosyncratic history it can cost accuracy. A logistic in n
    hands over smoothly - no jump when a user crosses a threshold.
    """
    return 1.0 / (1.0 + math.exp(-(n - n0) / max(scale, 1e-6)))


@dataclass
class UserModel:
    """
    personal      content ridge on raw scores - the original production model
    stack         population layer only
    hierarchical  stack + ridge(content + stack inputs) fitted to its residual
    augmented     ridge(content + stack inputs) on raw scores: population
                  signals are just more features, used only as far as they help
    blend         lam * personal-side model + (1 - lam) * stack, with lam
                  chosen on the user's own newest ratings when there are enough
                  of them, and from list size otherwise
    sized         lam * personal + (1 - lam) * stack with lam = size_weight(n):
                  the population carries thin lists, the personal model long ones
    """
    mode: str
    fold: FoldIn
    stacker: Ridge
    store: ItemStore
    taste: np.ndarray | None
    vocab: Vocabulary | None = None
    ridge: Ridge | None = None
    with_stack_features: bool = False
    on_residual: bool = False
    alpha: float | None = None
    lam: float = 1.0
    inner: UserModel | None = None   # the personal-side model inside a blend

    def stack_inputs(self, items: list[int]) -> np.ndarray:
        ex = stacker_extras(self.stacker)
        sig = extra_signals(self.fold.signals(items), self.fold, self.store, self.taste,
                            items, ex)
        return stack_matrix(sig, items, self.store, ex)

    def stack(self, items: list[int]) -> np.ndarray:
        return self.fold.mu + self.stacker.predict(self.stack_inputs(items))

    def rows(self, items: list[int]) -> list[dict]:
        """Feature rows for the personal model, with staff affinity filled in
        when a content column uses it."""
        aff = None
        if self.vocab is not None and "staff_aff" in self.vocab.extra:
            aff = staff_affinity(self.fold, self.store, items)
        return self.store.user_rows(items, self.taste, staff_aff=aff)

    def _features(self, rows: list[dict]) -> np.ndarray:
        X = vectorise(rows, self.vocab)
        if not self.with_stack_features:
            return X
        ids = [r["mal_id"] for r in rows]
        return np.hstack([X, self.stack_inputs(ids)])

    def _predict_known(self, rows: list[dict]) -> np.ndarray:
        ids = [r["mal_id"] for r in rows]
        if self.mode == "stack":
            return self.stack(ids)
        if self.mode in ("blend", "sized"):
            st = self.stack(ids)
            if self.inner is None or self.lam <= 0:
                return st
            return self.lam * self.inner._predict_known(rows) + (1 - self.lam) * st
        if self.ridge is None:           # too few ratings to fit anything
            return self.stack(ids)
        pred = self.ridge.predict(self._features(rows))
        return pred + self.stack(ids) if self.on_residual else pred

    def relevance(self, items: list[int]) -> np.ndarray:
        """Co-occurrence relevance, z-scored across the given items: how far
        each sits inside the user's viewing neighbourhood. A ranking signal,
        not part of the predicted score."""
        def z(v):
            sd = float(np.nanstd(v))
            return (v - np.nanmean(v)) / sd if sd > 1e-9 else np.zeros_like(v)
        rel = z(self.fold.signals(items)["rel"])
        ease = self.fold.ease_scores(items)
        if ease is not None and np.isfinite(ease).any():
            # co-occurrence and EASE, equally weighted (experiments/exp_signals.py
            # --part ease/final: recall up at every list size, popularity unchanged)
            ease = z(np.where(np.isnan(ease), np.nanmin(ease), ease))
            rel = 0.5 * (rel + ease)
        return rel

    def predict(self, items: list[int]) -> np.ndarray:
        """Raw score for each item (NaN for items the store does not know)."""
        out = np.full(len(items), np.nan)
        src = self.inner if self.inner is not None else self
        rows = src.rows(items)
        if rows:
            pos = {m: k for k, m in enumerate(items)}
            for r, v in zip(rows, self._predict_known(rows)):
                out[pos[r["mal_id"]]] = v
        return out

    def predict_parts(self, items: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """(population stack, personal model) raw predictions per item; the
        personal one is NaN where no personal model is fitted or used."""
        st = np.full(len(items), np.nan)
        pe = np.full(len(items), np.nan)
        src = self.inner if self.inner is not None else self
        rows = src.rows(items)
        if not rows:
            return st, pe
        at = {m: k for k, m in enumerate(items)}
        pos = [at[r["mal_id"]] for r in rows]
        st[pos] = self.stack([r["mal_id"] for r in rows])
        if self.inner is not None and self.lam > 1e-3:
            pe[pos] = self.inner._predict_known(rows)
        return st, pe


def _fit_ridge_part(model: UserModel, rows, ex, alpha):
    from ..model import choose_alpha
    ids = [e.mal_id for e in ex]
    y = np.array([e.score for e in ex])
    if model.on_residual:
        y = y - model.stack(ids)
    w = np.array([e.weight for e in ex])
    model.vocab = build_vocabulary(rows)
    X = model._features(rows)
    model.alpha = alpha if alpha is not None else choose_alpha(X, y, w, folds=min(10, len(y)))
    model.ridge = Ridge(alpha=model.alpha).fit(X, y, sample_weight=w)


def fit_user(pop: PopulationModel, stacker: Ridge, store: ItemStore,
             scored: list[Example], implicit: list[Example] | None = None,
             mode: str = "hierarchical", alpha: float | None = None,
             blend_of: str = "augmented", blend_k: float = 40.0, blend_min_n: int = 40,
             size_n0: float = 80.0, size_scale: float = 15.0,
             implicit_in_foldin: bool = False,
             listed: dict[int, float] | None = None,
             size_n: int | None = None, lam_max: float = 1.0) -> UserModel:
    """Fit one user's model. `alpha=None` picks ridge penalties by weighted CV.
    `lam_max` caps the personal model's share in the "sized" and "blend" modes."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    implicit = implicit or []
    examples = scored + implicit
    placed = examples if implicit_in_foldin else scored
    fold = pop.fold_in({e.mal_id: e.score for e in placed},
                       {e.mal_id: e.weight for e in placed}, listed=listed)
    # taste vector from real ratings only, as the SQL version does
    taste = store.taste_vector({e.mal_id: e.score for e in scored},
                               {e.mal_id: e.weight for e in scored})
    model = UserModel(mode, fold, stacker, store, taste)
    if mode == "stack":
        return model

    if mode == "sized":
        # size_n lets a holdout evaluate the weight production would really
        # use for this user, rather than the one implied by the truncated
        # training window
        model.lam = min(size_weight(size_n if size_n is not None else len(scored),
                                    size_n0, size_scale), lam_max)
        if model.lam > 1e-3:
            model.inner = fit_user(pop, stacker, store, scored, implicit, mode="personal",
                                   alpha=alpha, implicit_in_foldin=implicit_in_foldin,
                                   listed=listed)
        return model

    if mode == "blend":
        model.inner = fit_user(pop, stacker, store, scored, implicit, mode=blend_of,
                               alpha=alpha, implicit_in_foldin=implicit_in_foldin,
                               listed=listed)
        model.lam = min(_choose_lambda(pop, stacker, store, scored, implicit, blend_of, alpha,
                                       blend_k, blend_min_n, implicit_in_foldin), lam_max)
        return model

    from ..features import extra_columns
    if "staff_aff" in extra_columns():
        model.vocab = Vocabulary(extra=("staff_aff",))     # makes rows() fill it
    rows = model.rows([e.mal_id for e in examples])
    by_id = {e.mal_id: e for e in examples}
    ex = [by_id[r["mal_id"]] for r in rows]
    trend = settings_level_trend()
    if trend:
        # how long before the newest rating each example was given; rows
        # being predicted never carry it, so they are scored at today's level
        dated = [e.at for e in ex if e.at is not None]
        ref = max(dated) if dated else None
        if trend == "koren" and dated:
            # Koren (KDD 2009): dev_u(t) = sign(t - t_u) |t - t_u|^0.4 in days,
            # t_u the user's mean rating date; shifted so today's level is 0
            t_u = min(dated) + (sum(((d - min(dated)) for d in dated), dt.timedelta())
                                / len(dated))

            def dev(t):
                days = (t - t_u).total_seconds() / 86_400.0
                return math.copysign(abs(days) ** 0.4, days) / 10.0
            dev_ref = dev(ref)
        for r, e in zip(rows, ex):
            if ref is None or e.at is None:
                r["age_y"] = 0.0
            elif trend == "koren":
                r["age_y"] = dev(e.at) - dev_ref
            else:
                yrs = (ref - e.at).total_seconds() / 31_557_600.0
                r["age_y"] = min(yrs, 3.0) if trend == "lin3" else math.log1p(max(yrs, 0.0))
    if len(ex) < 5:
        model.vocab = None
        return model
    model.with_stack_features = mode in ("hierarchical", "augmented")
    model.on_residual = mode == "hierarchical"
    _fit_ridge_part(model, rows, ex, alpha)
    return model


def settings_level_trend() -> str:
    from ..config import settings
    return settings().level_trend


def _choose_lambda(pop, stacker, store, scored, implicit, blend_of, alpha,
                   k, min_n, implicit_in_foldin) -> float:
    """How much to trust the personal-side model over the population stack.

    With enough history this is decided on the user's own newest ratings: fit
    on the older ones, see which mix predicts the newer ones best. That is
    what guarantees a long-standing account is not dragged toward the
    population average when its own model is better. Thin lists use n/(n+k).
    """
    n = len(scored)
    if n < min_n:
        return n / (n + k)
    ordered = sorted(scored, key=lambda e: (e.at is None, e.at))
    cut = int(n * 0.8)
    tr, va = ordered[:cut], ordered[cut:]
    first_val = va[0].at
    imp = [e for e in implicit if e.at is None or first_val is None or e.at < first_val]
    inner = fit_user(pop, stacker, store, tr, imp, mode=blend_of, alpha=alpha,
                     implicit_in_foldin=implicit_in_foldin)
    stk = fit_user(pop, stacker, store, tr, imp, mode="stack",
                   implicit_in_foldin=implicit_in_foldin)
    items = [e.mal_id for e in va]
    truth = np.array([e.score for e in va])
    a, b = inner.predict(items), stk.predict(items)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 5:
        return n / (n + k)
    best, best_err = 1.0, float("inf")
    for lam in np.linspace(0, 1, 11):
        err = float(np.mean((lam * a[ok] + (1 - lam) * b[ok] - truth[ok]) ** 2))
        if err < best_err - 1e-9:
            best, best_err = float(lam), err
    return best


# ------------------------------------------------------------ stacker fit --

def recency_weight(at, cutoff, half_life: float | None = None) -> float:
    """Python twin of SQL recency_weight(): exponential decay from `cutoff`.
    `half_life` overrides the configured one (chosen per user, see
    model.train_hybrid)."""
    from ..config import settings
    cfg = settings()
    hl = cfg.recency_half_life_years if half_life is None else half_life
    age = max((cutoff - at).total_seconds() / 31_557_600.0, 0.0)
    return cfg.recency_floor + (1 - cfg.recency_floor) * 0.5 ** (age / hl)


def unit_hash(*parts) -> float:
    """A stable number in [0, 1) from the parts - the same on every run and
    every host, and unaffected by which other users are in the sample."""
    import hashlib
    d = hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2.0 ** 64


def user_rng(*parts) -> np.random.Generator:
    """A random stream that belongs to one user (and purpose), so a refit
    draws the same for them whoever else joined or left the sample."""
    return np.random.default_rng(int(unit_hash(*parts) * 2 ** 63))


def split_user(ratings: list[tuple], rng: np.random.Generator, test_frac: float = 0.2):
    """(mal_id, score, at) -> train, test, cutoff. Temporal, unless the dates
    are one bulk import, in which case the order carries no information."""
    days: dict = {}
    for _, _, at in ratings:
        days[at.date()] = days.get(at.date(), 0) + 1
    bulk = max(days.values()) / len(ratings) > 0.5
    rs = list(ratings)
    if bulk:
        rng.shuffle(rs)
    else:
        rs.sort(key=lambda x: x[2])
    k = max(3, round(len(rs) * test_frac))
    train, test = rs[:-k], rs[-k:]
    cutoff = max(x[2] for x in rs) if bulk else test[0][2]
    return train, test, cutoff


def fit_stacker(pop: PopulationModel, users: dict[int, list[tuple]], store: ItemStore,
                seed: int = 0, extras: tuple[str, ...] | None = None) -> Ridge:
    """Learn how much to trust each population signal, from users the
    population model never saw.

    Each user's newest ratings are predicted from a random subset of their
    older ones, with the subset size drawn from 10 up to everything - so the
    weights reflect how reliable the signals are across list sizes, not just
    for heavy users.
    """
    from ..config import settings
    extras = tuple(settings().stack_extras) if extras is None else tuple(extras)
    X, y = [], []
    for u, ratings in sorted(users.items()):
        if len(ratings) < 10:
            continue
        rng = user_rng("stacker", seed, u)
        train, test, cutoff = split_user(ratings, rng)
        n = int(rng.choice([10, 25, 60, 150, 10_000]))
        train = [train[i] for i in rng.permutation(len(train))[:n]]
        r = {m: s for m, s, _ in train}
        w = {m: recency_weight(a, cutoff) for m, _, a in train}
        fi = pop.fold_in(r, w)
        taste = store.taste_vector(r, w) if "tag_cos" in extras else None
        items = [m for m, _, _ in test]
        sig = extra_signals(fi.signals(items), fi, store, taste, items, extras)
        X.append(stack_matrix(sig, items, store, extras))
        y.append(np.array([s for _, s, _ in test]) - fi.mu)
    model = Ridge(alpha=5.0).fit(np.vstack(X), np.concatenate(y))
    model.extras_ = extras
    return model


# ------------------------------------------------------------- persistence --

@dataclass
class GlobalModel:
    """The population model and stacker, fitted once from the CF sample."""
    pop: PopulationModel
    stacker: Ridge
    meta: dict

    def dumps(self) -> bytes:
        buf = io.BytesIO()
        pickle.dump({"pop": self.pop, "stacker": self.stacker, "meta": self.meta}, buf)
        return buf.getvalue()

    @staticmethod
    def loads(blob: bytes) -> GlobalModel:
        d = pickle.loads(blob)
        return GlobalModel(d["pop"], d["stacker"], d["meta"])
