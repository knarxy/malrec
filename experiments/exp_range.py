"""Which range should a card show next to a prediction?

Today: the 5-95 % band of past errors (own holdout) or the 10-90 % band
(population users with a list this size), "likely 5-8". A user read that as
"you might rate it a 5". Candidates are narrower: the 25-75 % band, or the
smallest run of whole ratings that holds half (or 60 %) of the error
distribution, "most likely 7-8". Whichever is shown has to keep its promise
strictly out of time - on ratings given after everything the range was
derived from - for both sources, overall and for the top of a list.

As in production (model.train_hybrid, service.fit_size_calibration):
  population   budgets of 25 / 60 / 150 entries; calibration line and error
               quantiles from even users at that budget, checked on odd ones
  own          users with 150 / 400 entries: their newest 40 ratings before
               the cutoff are the holdout a model on the older ones predicts;
               its line and quantiles are applied to the ratings after the
               cutoff, predicted by a model on everything before it
"top" are each user's predictions in the top quarter of their test titles.

  python -m experiments.exp_range [--users 300]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments import bench
from malrec.recsys.hybrid import split_user
from malrec.recsys.service import calibration_from
from malrec.uncertainty import QS, _ends, residual_quantiles

OWN_HOLDOUT = 40
VARIANTS = (("5-95 %", (1, 19)), ("10-90 %", (2, 18)), ("25-75 %", (5, 15)),
            ("half", 0.5), ("60 %", 0.6))


def mass_set(pred: float, q: np.ndarray, mass: float) -> tuple[int, int]:
    """The shortest run of whole ratings around the likeliest one that holds
    `mass` of the error distribution."""
    def cdf(x):
        return float(np.interp(x, q, QS, left=0.0, right=1.0))
    edges = [cdf(k + 0.5 - pred) for k in range(11)]          # F at 0.5 .. 10.5
    probs = np.diff(edges)
    probs[0] += edges[0]                                         # below 0.5 counts as 1
    probs[-1] += 1.0 - edges[-1]                                 # above 10.5 as 10
    lo = hi = int(np.argmax(probs))
    cum = probs[lo]
    while cum < mass and (lo > 0 or hi < 9):
        left = probs[lo - 1] if lo > 0 else -1.0
        right = probs[hi + 1] if hi < 9 else -1.0
        if right >= left:
            hi += 1
            cum += probs[hi]
        else:
            lo -= 1
            cum += probs[lo]
    return lo + 1, hi + 1


def ends(v, pred: float, q: np.ndarray) -> tuple[int, int]:
    return _ends(pred, q, v) if isinstance(v, tuple) else mass_set(pred, q, v)


def score(records, q_of):
    """records: (user, shown, truth, top, q_key); q_of(q_key) -> quantiles."""
    out = {}
    for name, v in VARIANTS:
        for part in ("all", "top"):
            hit, width, single = [], [], []
            for _, shown, truth, top, key in records:
                q = q_of(key)
                if q is None:
                    continue
                for p, t, is_top in zip(shown, truth, top, strict=True):
                    if part == "top" and not is_top:
                        continue
                    lo, hi = ends(v, float(p), q)
                    hit.append(lo <= t <= hi)
                    width.append(hi - lo + 1)
                    single.append(lo == hi)
            if hit:
                out[(name, part)] = (np.mean(hit), np.mean(width), np.mean(single), len(hit))
    return out


def show(title, res):
    print(f"\n== {title}")
    print(f"   {'range':<9}{'covers':>8}{'width':>7}{'one #':>7}   {'top: covers':>11}{'width':>7}"
          f"{'one #':>7}{'ratings':>9}")
    for name, _ in VARIANTS:
        a, t = res.get((name, "all")), res.get((name, "top"))
        if a and t:
            print(f"   {name:<9}{a[0]:8.3f}{a[1]:7.2f}{a[2]:7.2f}   {t[0]:11.3f}{t[1]:7.2f}{t[2]:7.2f}"
                  f"{a[3]:9d}")


def main(max_eval: int):
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    (pop, stacker), how = bench.population(entries, store, A, B)
    print(f"population model {how} [{time.time() - t0:.0f}s]", flush=True)

    def fit_predict(seen, cutoff, items):
        sc, im, li = bench.examples(seen, cutoff)
        if len(sc) < 5:
            return None
        s = bench.make_scorer(pop, stacker, store, sc, im, li, (1.0, 0.0))
        return np.asarray(s.predict_ids(items), dtype=float)

    def top_mask(p):
        return p >= np.quantile(p, 0.75)

    # ---- population-based ranges
    for budget in (25, 60, 150):
        raw = []                                    # (user, raw preds, truth)
        urng = np.random.default_rng(bench.SEED + budget)
        for u in C:
            if len(raw) >= 2 * max_eval:
                break
            es = entries[u]
            scored = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test, cutoff = split_user(scored, urng)
            test_ids = {m for m, _, _ in test}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in urng.permutation(len(history))[:budget]]
            p = fit_predict(seen, cutoff, [m for m, _, _ in test])
            if p is None:
                continue
            t = np.array([s for _, s, _ in test])
            ok = ~np.isnan(p)
            if ok.sum() >= 4:
                raw.append((u, p[ok], t[ok]))
        even = [r for k, r in enumerate(raw) if k % 2 == 0]
        odd = [r for k, r in enumerate(raw) if k % 2 == 1]
        cal = calibration_from(np.concatenate([p for _, p, _ in even]).tolist(),
                               np.concatenate([t for _, _, t in even]).tolist())
        a, b = cal["slope"], cal["intercept"]

        def shown_of(p, a=a, b=b):
            return np.clip(a * p + b, 1, 10)
        ev_s = np.concatenate([shown_of(p) for _, p, _ in even])
        ev_t = np.concatenate([t for _, _, t in even])
        ev_top = np.concatenate([top_mask(shown_of(p)) for _, p, _ in even])
        qs = {"all": np.array(residual_quantiles(ev_s, ev_t)),
              "top": np.array(residual_quantiles(ev_s[ev_top], ev_t[ev_top]))}
        recs = [(u, shown_of(p), t, top_mask(shown_of(p)), "top") for u, p, t in odd]
        show(f"population, {budget} entries ({len(odd)} users checked, top-of-list quantiles)",
             score(recs, qs.get))
        recs = [(u, shown_of(p), t, top_mask(shown_of(p)), "all") for u, p, t in odd]
        show(f"population, {budget} entries (all-title quantiles)", score(recs, qs.get))
        print(f"   [{time.time() - t0:.0f}s]", flush=True)

    # ---- own-holdout ranges
    for size in (150, 400):
        recs, own_q = [], {}
        urng = np.random.default_rng(bench.SEED + 7 * size)
        rejected = 0
        for u in C:
            if len(recs) >= max_eval:
                break
            es = entries[u]
            scored = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test, cutoff = split_user(scored, urng)
            test_ids = {m for m, _, _ in test}
            history = sorted((e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff),
                             key=lambda e: e["at"])
            if len(history) < size:
                continue
            history = history[-size:]               # the newest `size` entries
            hist_scored = [e for e in history if e["score"] > 0]
            if len(hist_scored) < OWN_HOLDOUT + 30:
                continue
            hold = hist_scored[-OWN_HOLDOUT:]
            inner_cut = hold[0]["at"]
            hold_ids = {e["mal_id"] for e in hold}
            inner = [e for e in history if e["mal_id"] not in hold_ids and e["at"] < inner_cut]
            p_in = fit_predict(inner, inner_cut, [e["mal_id"] for e in hold])
            if p_in is None:
                continue
            t_in = np.array([float(e["score"]) for e in hold])
            ok = ~np.isnan(p_in)
            cal = calibration_from(p_in[ok].tolist(), t_in[ok].tolist())
            if "rmse_raw" not in cal or cal["slope"] < 0.5:
                rejected += 1                        # production falls back to the population line
                continue
            a, b = cal["slope"], cal["intercept"]
            own_q[u] = np.array(residual_quantiles(np.clip(a * p_in[ok] + b, 1, 10), t_in[ok]))
            p = fit_predict(history, cutoff, [m for m, _, _ in test])
            if p is None:
                continue
            t = np.array([s for _, s, _ in test])
            ok = ~np.isnan(p)
            if ok.sum() < 4:
                continue
            s = np.clip(a * p[ok] + b, 1, 10)
            recs.append((u, s, t[ok], top_mask(s), u))
        show(f"own holdout, {size} entries ({len(recs)} users; {rejected} fell back)",
             score(recs, own_q.get))
        print(f"   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=300)
    a = ap.parse_args()
    main(a.users)
