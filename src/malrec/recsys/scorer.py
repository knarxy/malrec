"""Adapter that lets the hybrid model drop into the existing ranking code.

rank.py and surfaces.py were written against TasteModel: `predict(rows)`,
`calibrate(raw)`, `metrics`, `params`. HybridScorer offers the same surface,
plus `explain()` - which replaces coefficient-only drivers with reasons that
come from the population layer too ("because you rated X", backed by real
co-ratings rather than a hand-curated recommendation list).
"""
from __future__ import annotations

import math

import numpy as np

from ..rank import CONSENSUS_FEATURES, _label
from .cf import knn_contributors
from .hybrid import STACK_FEATURES, UserModel, stacker_extras

# How each stack input reads to a person. Sign-dependent ones are resolved in
# _stack_label.
_STACK_LABELS = {
    "bias": "loved by those who've seen it",
    "knn": "similar to shows you rated highly",
    "knn_x_support": "similar to shows you rated highly",
    "mf": "fits your taste profile",
    "mal_c": "highly rated on MyAnimeList",
    "al_c": "highly rated on AniList",
    "staff": "by people whose work you rate highly",
    "tag_cos": "matches your tag profile",
}
# MAL lists demographics among the genres; they say who it is marketed to,
# not what it is, so they never make a title "unfamiliar".
DEMOGRAPHICS = frozenset({"Shounen", "Seinen", "Shoujo", "Josei", "Kids"})
LONG_EPISODES = 50


def is_long_series(row: dict, year: int | None = None) -> bool:
    """50+ episodes, or still airing three or more years after it began
    (MAL lists such a series with 0 episodes until it ends)."""
    import datetime as dt
    if (row.get("num_episodes") or 0) >= LONG_EPISODES:
        return True
    year = year or dt.datetime.now(dt.UTC).year
    return (not row.get("num_episodes") and row.get("status") == "currently_airing"
            and (row.get("season_year") or year) <= year - 3)


# Stack inputs that say nothing about this particular user.
_STACK_CONSENSUS = {"bias", "mal_c", "al_c", "pop", "drop", "polar"}


def _stack_label(name: str, value: float) -> str | None:
    if name == "pop":
        return "a popular title" if value < 0 else "under the radar"
    if name == "drop":
        return "rarely dropped" if value < 0 else "often dropped"
    if name == "polar":
        return "divides opinion" if value > 0 else "broadly liked"
    return _STACK_LABELS.get(name)


# Labels that name a reason in its favourable direction; a negative
# contribution under one of them reads as its opposite ("− fits your taste
# profile" would say the reverse of what it means).
_NEGATED = {
    "fits your taste profile": "outside your usual taste",
    "similar to shows you rated highly": "unlike shows you rated highly",
    "matches your tag profile": "tags unlike your favourites",
    "loved by those who've seen it": "mixed reception from those who've seen it",
    "highly rated on MyAnimeList": "modest MyAnimeList score",
    "highly rated on AniList": "modest AniList score",
    "by people whose work you rate highly": "by people whose work you rate lower",
    "recommended alongside shows you rate highly": "rarely recommended alongside your favourites",
}


def signed_label(label: str, value: float) -> str:
    return _NEGATED.get(label, label) if value < 0 else label


class HybridScorer:
    kind = "hybrid"

    def __init__(self, um: UserModel, mode: str, params: dict, metrics: dict,
                 calibration: tuple[float, float], scores: dict[int, float],
                 titles: dict[int, str], list_calibration: tuple[float, float] | None = None):
        self.um = um
        self.algo = mode
        self.params = params
        self.metrics = metrics
        self.calibration = calibration
        # the number shown on list cards, for accounts calibrated from the
        # population (see service.fit_size_calibration); ranking never uses it
        self.list_calibration = list_calibration
        self.baseline = um.fold.mu
        self._scores = scores          # the user's own ratings, for "you rated X 9"
        self._titles = titles

    # ------------------------------------------------ TasteModel interface --

    def predict(self, rows: list[dict]) -> np.ndarray:
        return self.um.predict([r["mal_id"] for r in rows])

    def risk_penalty(self, ids: list[int], familiar: set[str] | None = None) -> np.ndarray:
        """Order-key penalty in display points for a pick the user's own
        history does not back up: the sum of risk_parts."""
        return sum(self.risk_parts(ids, familiar).values(), np.zeros(len(ids)))

    def risk_parts(self, ids: list[int], familiar: set[str] | None = None
                   ) -> dict[str, np.ndarray]:
        """The risk penalty by reason (experiments/exp_toplist.py).

        Scaled by the trust in the personal model (personal share /
        blend_lam_max), so thin lists, where it is weak, are left alone:
          disagree    disagreement_penalty per point the population layer is
                      more optimistic than the personal model
          unfamiliar  unfamiliar_genre_penalty for a genre absent from
                      `familiar` (the genres on the user's whole list)
          acclaim     acclaim_penalty per point of the personal model's lift
                      over the user's average rated title that comes from the
                      acclaim columns (MAL/AniList score, popularity,
                      favourites), fading as rated titles of the user's
                      vouch for it through co-ratings: e0 / (e0 + evidence)
        At every list size:
          long        long_series_penalty for 50+ episodes, or a series that
                      has been airing for three years or more (MAL gives those
                      0 episodes: Detective Conan, Crayon Shin-chan were thin
                      lists' Hidden Gems)
        """
        from ..config import settings
        cfg = settings()
        zero = np.zeros(len(ids))
        out = {"long": zero, "disagree": zero, "unfamiliar": zero, "acclaim": zero}
        rows = self.um.store.rows if (cfg.unfamiliar_genre_penalty > 0
                                      or cfg.long_series_penalty > 0) else {}
        if cfg.long_series_penalty > 0:
            out["long"] = cfg.long_series_penalty * np.array(
                [float(is_long_series(rows.get(m) or {})) for m in ids])
        if self.um.mode not in ("sized", "blend") or self.um.inner is None:
            return out
        trust = min(1.0, float(self.um.lam) / max(cfg.blend_lam_max, 1e-6))
        if trust <= 0:
            return out
        slope = float(self.calibration[0])
        if cfg.disagreement_penalty > 0:
            st, pe = self.um.predict_parts(ids)
            gap = np.nan_to_num(np.maximum(st - pe, 0.0), nan=0.0)
            out["disagree"] = trust * cfg.disagreement_penalty * slope * gap
        if cfg.unfamiliar_genre_penalty > 0 and familiar is not None:
            out["unfamiliar"] = trust * cfg.unfamiliar_genre_penalty * np.array([
                float(any(g not in DEMOGRAPHICS and g not in familiar
                          for g in ((rows.get(m) or {}).get("mal_genres") or [])))
                for m in ids])
        if cfg.acclaim_penalty > 0:
            lift = np.maximum(self.acclaim_lift(ids), 0.0)
            e0 = cfg.acclaim_evidence_e0
            fade = e0 / (e0 + self.um.fold.evidence(ids)) if e0 > 0 else 1.0
            out["acclaim"] = trust * cfg.acclaim_penalty * slope * lift * fade
        return out

    def acclaim_lift(self, ids: list[int]) -> np.ndarray:
        """Per item, the part of the personal model's prediction above the
        user's average rated title that comes from the acclaim columns
        (CONSENSUS_FEATURES). Zero without a personal ridge."""
        out = np.zeros(len(ids))
        p = self.personal if getattr(self.um.inner, "ridge", None) is not None else None
        if p is None or p.vocab is None:
            return out
        names = list(p.vocab.names)
        coef = np.asarray(p.ridge.coef_, dtype=float)
        if getattr(self, "_acclaim_ref", None) is None:
            rated = p.rows([m for m, v in self._scores.items() if v > 0])
            self._acclaim_ref = p._features(rated).mean(0) if rated else None
        rows = p.rows(ids)
        if self._acclaim_ref is None or not rows:
            return out
        cons = np.array([n in CONSENSUS_FEATURES for n in names]
                        + [False] * (len(coef) - len(names)))
        D = (p._features(rows) - self._acclaim_ref) * coef
        pos = {m: k for k, m in enumerate(ids)}
        for r, d in zip(rows, D):
            out[pos[r["mal_id"]]] = d[cons].sum()
        return out

    def predict_ids(self, ids: list[int]) -> np.ndarray:
        return self.um.predict(ids)

    def calibrate(self, raw) -> np.ndarray:
        a, b = self.calibration
        return np.clip(a * np.asarray(raw, dtype=float) + b, 1.0, 10.0)

    def calibrate_list(self, raw) -> np.ndarray:
        """What a list card shows. Same order as calibrate()."""
        if self.list_calibration is None:
            return self.calibrate(raw)
        a, b = self.list_calibration
        return np.clip(a * np.asarray(raw, dtype=float) + b, 1.0, 10.0)

    def relevance(self, ids: list[int]) -> np.ndarray:
        return self.um.relevance(ids)

    def rows(self, ids: list[int]) -> list[dict]:
        src = self.um.inner if self.um.inner is not None else self.um
        return src.rows(ids)

    @property
    def personal_share(self) -> float:
        """Share of the prediction that comes from the personal model."""
        m = self.um.mode
        if m in ("blend", "sized"):
            return float(self.um.lam)
        return 0.0 if m == "stack" else 1.0

    @property
    def ranking_share(self) -> float:
        """The share the ranking dials work from (config.ranking_share_n0):
        the blend's personal share, or list size on a fixed curve."""
        from ..config import settings
        from .hybrid import size_weight
        n0 = settings().ranking_share_n0
        if n0 <= 0:
            return self.personal_share
        return size_weight(sum(1 for v in self._scores.values() if v > 0), n0, 15.0)

    @property
    def personal(self) -> UserModel | None:
        """The ridge-bearing part: itself, or the inner model of a blend."""
        um = self.um.inner if self.um.mode in ("blend", "sized") else self.um
        return um if um is not None and um.ridge is not None else None

    @property
    def estimator(self):
        p = self.personal
        return p.ridge if p is not None else None

    @property
    def vocab(self):
        p = self.personal
        return p.vocab if p is not None else None

    # ---------------------------------------------------------- explanation --

    def _side_contributions(self, ids: list[int], rows: list[dict]
                            ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """Per item, label -> contribution within each side's own prediction,
        unweighted: (population stack, personal ridge). A side not in use
        gives empty dicts."""
        um = self.um
        lam = um.lam if um.mode in ("blend", "sized") else (0.0 if um.mode == "stack" else 1.0)
        stack_out: list[dict[str, float]] = [{} for _ in ids]
        pers_out: list[dict[str, float]] = [{} for _ in ids]
        if um.mode in ("blend", "sized", "stack", "hierarchical"):
            S = um.stack_inputs(ids)
            C = S * um.stacker.coef_
            stack_names = STACK_FEATURES + stacker_extras(um.stacker)
            for k in range(len(ids)):
                for j, name in enumerate(stack_names):
                    label = _stack_label(name, S[k, j])
                    if label:
                        stack_out[k][label] = stack_out[k].get(label, 0.0) + C[k, j]
        p = self.personal
        if p is not None and lam > 0:
            X = p._features(rows)
            coef = np.asarray(p.ridge.coef_, dtype=float)
            p_stack = STACK_FEATURES + stacker_extras(p.stacker)
            names = list(p.vocab.names) + (list(p_stack) if p.with_stack_features else [])
            C = X * coef
            for k in range(len(ids)):
                for j in np.nonzero(C[k])[0]:
                    name = names[j]
                    label = (_stack_label(name, X[k, j]) if name in p_stack
                             else _label(name, float(X[k, j])))
                    if label:
                        pers_out[k][label] = pers_out[k].get(label, 0.0) + C[k, j]
        return stack_out, pers_out

    def _contributions(self, ids: list[int], rows: list[dict]) -> list[dict[str, float]]:
        """Per item: label -> contribution to the final score, from both the
        population stack and the personal ridge, weighted by their share."""
        um = self.um
        lam = um.lam if um.mode in ("blend", "sized") else (0.0 if um.mode == "stack" else 1.0)
        share = 1.0 - lam if um.mode in ("blend", "sized") else 1.0
        stack, pers = self._side_contributions(ids, rows)
        out: list[dict[str, float]] = [{} for _ in ids]
        for k in range(len(ids)):
            for lbl, v in stack[k].items():
                out[k][lbl] = out[k].get(lbl, 0.0) + v * share
            for lbl, v in pers[k].items():
                out[k][lbl] = out[k].get(lbl, 0.0) + v * lam
        return out

    def explain(self, ids: list[int], novelty: list[float] | None = None,
                relevance: list[float] | None = None) -> list[list[dict]]:
        rows = self.rows(ids)
        by_id = {r["mal_id"]: r for r in rows}
        known = [m for m in ids if m in by_id]
        contribs = dict(zip(known, self._contributions(known, [by_id[m] for m in known])))

        # distinctive = above what the average candidate gets for that reason
        labels = {lbl for c in contribs.values() for lbl in c}
        mean = {lbl: float(np.mean([c.get(lbl, 0.0) for c in contribs.values()]))
                for lbl in labels} if contribs else {}
        consensus_labels = {_STACK_LABELS.get(n) for n in _STACK_CONSENSUS} | {
            _label(n) for n in CONSENSUS_FEATURES} | {"a popular title", "under the radar",
            "rarely dropped", "often dropped", "divides opinion", "broadly liked"}

        out = []
        for k, m in enumerate(ids):
            reasons: list[dict] = []
            for rid, _ in knn_contributors(self.um.fold, m, top=3):
                sc = self._scores.get(rid)
                if sc is not None and sc > self.um.fold.mu:
                    reasons.append({"kind": "because_you_liked",
                                    "title": self._titles.get(rid, str(rid)), "your_score": sc})
            row = by_id.get(m)
            if row is not None:
                tags = [t for t, r in sorted(zip(row.get("tag_names") or [],
                                                 row.get("tag_ranks") or []),
                                             key=lambda x: -x[1]) if r >= 60][:4]
                if tags:
                    reasons.append({"kind": "tags", "tags": tags})
            c = contribs.get(m)
            if c:
                total = sum(abs(v) for v in c.values()) or 1.0
                acclaim = sum(abs(v) for lbl, v in c.items() if lbl in consensus_labels)
                items = sorted(((lbl, v - mean.get(lbl, 0.0), v) for lbl, v in c.items()),
                               key=lambda x: -x[1])
                picked = [{"label": lbl, "weight": round(float(v), 3),
                           "personal": lbl not in consensus_labels}
                          for lbl, dv, v in items if dv > 0.01][:4]
                # The relevance bonus shapes the order but is not part of the
                # predicted score, so it is reported separately - and only when
                # it was a real reason this title made the list.
                if relevance is not None and k < len(relevance) and relevance[k] > 0.25:
                    picked.insert(0, {"label": "common on lists like yours",
                                      "weight": round(float(relevance[k]), 3),
                                      "personal": True})
                    picked = picked[:4]
                if picked:
                    linked = any(r["kind"] == "because_you_liked" for r in reasons) or any(
                        i["label"] == "common on lists like yours" for i in picked)
                    reasons.append({"kind": "drivers", "items": picked,
                                    "personal_share": round(1 - acclaim / total, 3),
                                    "has_affinity": linked})
            if novelty is not None and k < len(novelty) and novelty[k] > 0.85:
                reasons.append({"kind": "deep_cut"})
            out.append(reasons)
        return out


    def breakdown(self, mal_id: int, on_list: bool = True) -> dict | None:
        """Everything behind one prediction, for the "Why this?" panel:
        your average, what the population layer and the personal model each
        predict and how they are weighted, the calibration to the shown
        score, every driver pushing it up or down, and the rated titles that
        vouch for it through real co-ratings."""
        rows = self.rows([mal_id])
        if not rows:
            return None
        um = self.um
        raw = float(self.predict_ids([mal_id])[0])
        if math.isnan(raw):
            return None
        lam = self.personal_share
        stack = float(um.stack([mal_id])[0])
        personal = None
        p = self.personal
        if p is not None and lam > 0:
            personal = float(p._predict_known(rows)[0])
        contrib = self._contributions([mal_id], rows)[0]
        items = sorted(((lbl, float(v)) for lbl, v in contrib.items() if abs(v) >= 0.005),
                       key=lambda x: -abs(x[1]))
        up = [{"label": lbl, "value": round(v, 3)} for lbl, v in items if v > 0][:8]
        down = [{"label": signed_label(lbl, v), "value": round(v, 3)}
                for lbl, v in items if v < 0][:6]
        liked = [{"title": self._titles.get(rid, str(rid)), "your_score": self._scores.get(rid),
                  "weight": round(float(w), 3)}
                 for rid, w in knn_contributors(um.fold, mal_id, top=5)
                 if self._scores.get(rid) is not None]
        a, b = (self.list_calibration if on_list and self.list_calibration
                else self.calibration)

        def side(c: dict[str, float]) -> dict:
            its = sorted(((lbl, float(v)) for lbl, v in c.items() if abs(v) >= 0.01),
                         key=lambda x: -abs(x[1]))
            return {"up": [{"label": lbl, "value": round(v, 3)} for lbl, v in its if v > 0][:4],
                    "down": [{"label": signed_label(lbl, v), "value": round(v, 3)}
                             for lbl, v in its if v < 0][:3]}
        st_c, pe_c = self._side_contributions([mal_id], rows)
        return {
            "your_mean": round(float(um.fold.mu), 2),
            "population": round(stack, 2),
            "personal": round(personal, 2) if personal is not None else None,
            "personal_share": round(lam, 3),
            "raw": round(raw, 2),
            "calibration": {"slope": round(float(a), 3), "intercept": round(float(b), 3)},
            "shown": round(float(np.clip(a * raw + b, 1.0, 10.0)), 2),
            "up": up, "down": down, "because_you_liked": liked,
            # each side's own reasons, in that side's points (the "two voices")
            "sides": {"population": side(st_c[0]),
                      "personal": side(pe_c[0]) if personal is not None else None},
        }


def novelty_scale(shown: np.ndarray, ref_sd: float) -> float:
    """Shrink the novelty bonus when predictions are nearly flat.

    Novelty is meant to break near-ties between good candidates. When a model
    has little to go on its predictions barely differ, and a full-strength
    bonus then decides the ranking by itself - which is exactly how a Tokyo
    Revengers fan was once shown Precure. Scaling by the prediction spread,
    capped at 1, leaves a confident model untouched.
    """
    sd = float(np.nanstd(shown)) if len(shown) else 0.0
    if not math.isfinite(sd) or ref_sd <= 0:
        return 1.0
    return max(0.0, min(1.0, sd / ref_sd))
