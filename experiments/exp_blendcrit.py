"""How a long list's personal share is best chosen (audit 2026-10-23, P1-11).

From 300 ratings hybrid._choose_lambda picks the share on the user's newest
fifth: fit both sides on the older ratings, try the mixes. By squared error
("mse") it also scores the level, which the stacker's intercept carries; by
correlation ("corr") only the order, which is what the lists use - the shown
score gets its own line (the calibration). This measures the choice itself,
without the lists: on the benchmark's held-out users (frozen snapshot, its
population model), each with their whole history before the cutoff and the
newest 20 % of their ratings as the future,

  rho     rank correlation of the served mix with the future ratings
  rmse    its error after the line the calibration would fit on the user's
          history (here: on the newest fifth before the cutoff)

for each rule (criterion x floor, the 0.5 ceiling as served), next to fixed
shares and the best share in hindsight (oracle, on the future itself). The
choice on one newest fifth is noisy, so two choices on more of the list are
tried too ("<criterion>-<folds>"): t3 the newest three tenths, each against
everything before it; r5 five random fifths (hybrid.unit_hash, as
holdout="random"); squared error pooled, correlation averaged over folds.

  docker compose --profile lab run --rm -T --user root lab python -u -m experiments.exp_blendcrit
"""
from __future__ import annotations

import sys
import time

import numpy as np

from experiments import bench
from malrec.eval import spearman
from malrec.recsys import hybrid, ltr
from malrec.recsys.hybrid import fit_user, split_user

MIN_SCORED = 375                 # 80 % of it is still >= chosen_blend_min (300)
MAX_USERS = 600
GRID = np.round(np.linspace(0, 0.5, 6), 2)      # what serving can use (lam_max 0.5)
RULES = (("mse", 0.3), ("corr", 0.3), ("mse", 0.0), ("corr", 0.0))
FOLDS = ("t3", "r5")
BUCKETS = ((375, 749), (750, 1499), (1500, 10 ** 9))


def memo_fits():
    """hybrid.fit_user with the fits of one call to _choose_lambda kept, so
    both criteria are decided on the same two models (and the calibration
    line is fitted on what they predict for the newest fifth)."""
    real, last = hybrid.fit_user, {}

    def fit(pop, stacker, store, scored, implicit=None, mode="hierarchical", **kw):
        if mode not in last:
            last[mode] = real(pop, stacker, store, scored, implicit, mode=mode, **kw)
        return last[mode]
    return fit, last


def folds(sc: list, im: list, kind: str) -> list[tuple]:
    """(train, implicit, validation) splits of a user's ratings (FOLDS)."""
    out = []
    if kind == "t3":
        ordered = sorted(sc, key=lambda e: (e.at is None, e.at))
        n = len(ordered)
        for lo, hi in ((0.7, 0.8), (0.8, 0.9), (0.9, 1.0)):
            a, b = int(n * lo), int(n * hi)
            first = ordered[a].at
            out.append((ordered[:a], [e for e in im if e.at is None or first is None
                                      or e.at < first], ordered[a:b]))
        return out
    ordered = sorted(sc, key=lambda e: hybrid.unit_hash("blend", e.mal_id))
    n = len(ordered)
    for k in range(5):
        a, b = n * k // 5, n * (k + 1) // 5
        held = {e.mal_id for e in ordered[a:b]}
        out.append((ordered[:a] + ordered[b:], [e for e in im if e.mal_id not in held],
                    ordered[a:b]))
    return out


def fold_preds(pop, stacker, store, splits: list[tuple]) -> list[tuple]:
    """Per split, (personal, stack, truth) on its validation ratings."""
    out = []
    for tr, imp, va in splits:
        inner = fit_user(pop, stacker, store, tr, imp, mode="personal")
        stk = fit_user(pop, stacker, store, tr, imp, mode="stack")
        items = [e.mal_id for e in va]
        a, b = inner.predict(items), stk.predict(items)
        y = np.array([e.score for e in va])
        ok = ~(np.isnan(a) | np.isnan(b))
        out.append((a[ok], b[ok], y[ok]))
    return out


def choose(preds: list[tuple], criterion: str) -> float | None:
    """The share on the grid 0..1 that _choose_lambda's criterion picks over
    all folds (the lowest of equals)."""
    best, best_v = None, -float("inf")
    for lam in np.linspace(0, 1, 11):
        if criterion == "mse":
            v = -float(np.mean(np.concatenate([(lam * a + (1 - lam) * b - y) ** 2
                                               for a, b, y in preds])))
        else:
            rs, ws = [], []
            for a, b, y in preds:
                mix = lam * a + (1 - lam) * b
                if len(y) >= 5 and mix.std() > 1e-9 and y.std() > 1e-9:
                    rs.append(float(np.corrcoef(mix, y)[0, 1]))
                    ws.append(len(y))
            if not rs:
                continue
            v = float(np.average(rs, weights=ws))
        if v > best_v + 1e-9:
            best, best_v = float(lam), v
    return best


def score(lam: float, future: tuple, newest: tuple) -> tuple[float, float]:
    """(rho, rmse) of the mix lam * personal + (1 - lam) * stack on the future
    ratings, the rmse after the line fitted on the newest fifth before the
    cutoff (each a (personal, stack, truth) triple of arrays)."""
    a, b, y = future
    va, vb, vy = newest
    p = lam * a + (1 - lam) * b
    slope, icept = np.polyfit(lam * va + (1 - lam) * vb, vy, 1) if len(vy) >= 5 else (1.0, 0.0)
    return float(spearman(p, y)), float(np.sqrt(np.mean((slope * p + icept - y) ** 2)))


def main() -> int:
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    bench.GROUPS = bench.groupings(store)
    rng = np.random.default_rng(bench.SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    n_a, n_b = int(len(users) * 0.72), int(len(users) * 0.13)
    (pop, stacker), how = bench.population(entries, store, users[:n_a], users[n_a:n_a + n_b])
    long_ = [u for u in users[n_a + n_b:] if entries._n_scored.get(u, 0) >= MIN_SCORED]
    print(f"{len(long_)} held-out users with >= {MIN_SCORED} ratings, population model {how}"
          f" [{time.time() - t0:.0f}s]", flush=True)
    urng = np.random.default_rng(bench.SEED + 1)
    rows = []
    for u in long_[:MAX_USERS]:
        es = entries[u]
        scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
        _, test_s, cutoff = split_user(scored_all, urng)
        test_ids = {m for m, _, _ in test_s}
        hist = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
        sc, im, li = ltr.history_examples(hist, cutoff)
        if len(sc) < 300:
            continue
        fit, last = memo_fits()
        hybrid.fit_user = fit
        try:
            chosen = {c: hybrid._choose_lambda(pop, stacker, store, sc, im, "personal", None,
                                               40.0, 40, False, "newest", c)
                      for c in ("mse", "corr")}
        finally:
            hybrid.fit_user = fit_user
        # the newest fifth before the cutoff, for the calibration line
        ordered = sorted(sc, key=lambda e: (e.at is None, e.at))
        va = ordered[int(len(ordered) * 0.8):]
        if "personal" not in last or "stack" not in last:
            continue
        inner = fit_user(pop, stacker, store, sc, im, mode="personal", listed=li)
        stk = fit_user(pop, stacker, store, sc, im, mode="stack", listed=li)
        items = [m for m, _, _ in test_s]
        truth = np.array([s for _, s, _ in test_s])
        a, b = inner.predict(items), stk.predict(items)
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 10 or truth[ok].std() == 0:
            continue
        va_items = [e.mal_id for e in va]
        va_truth = np.array([e.score for e in va])
        future = (a[ok], b[ok], truth[ok])
        va_a, va_b = last["personal"].predict(va_items), last["stack"].predict(va_items)
        vok = ~(np.isnan(va_a) | np.isnan(va_b))
        newest = (va_a[vok], va_b[vok], va_truth[vok])
        r: dict = {"n": len(sc)}
        for g in GRID:
            r[f"fix{g}"] = score(float(g), future, newest)
        for c, floor in RULES:
            lam = min(max(chosen[c], floor), 0.5)
            r[f"{c}/{floor}"] = score(lam, future, newest)
            r[f"lam {c}/{floor}"] = lam
        for kind in FOLDS:
            preds = fold_preds(pop, stacker, store, folds(sc, im, kind))
            for c in ("mse", "corr"):
                got = choose(preds, c)
                for floor in (0.3, 0.0):
                    lam = min(max(got if got is not None else len(sc) / (len(sc) + 40), floor), 0.5)
                    r[f"{c}-{kind}/{floor}"] = score(lam, future, newest)
                    r[f"lam {c}-{kind}/{floor}"] = lam
        r["oracle"] = max((r[f"fix{g}"] for g in GRID), key=lambda x: x[0])
        rows.append(r)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} users [{time.time() - t0:.0f}s]", flush=True)

    arms = ([f"{c}/{f}" for c, f in RULES]
            + [f"{c}-{k}/{f}" for k in FOLDS for c in ("mse", "corr") for f in (0.3, 0.0)]
            + [f"fix{g}" for g in GRID] + ["oracle"])
    base = "mse/0.3"
    names = [f"{lo}-{hi}" if hi < 10 ** 9 else f"{lo}+" for lo, hi in BUCKETS] + ["all"]
    for k, name in enumerate(names):
        sel = [r for r in rows if k == len(BUCKETS) or BUCKETS[k][0] <= r["n"] <= BUCKETS[k][1]]
        if not sel:
            continue
        print(f"\n== ratings {name} ({len(sel)} users); change vs {base} (today), +-2 SE paired")
        print(f"   {'arm':<14}{'rho':>8}{'change':>9}{'±2se':>7}{'rmse':>8}{'change':>9}{'±2se':>7}"
              f"{'mean lam':>10}")
        for arm in arms:
            out = f"   {arm:<14}"
            for j in (0, 1):
                v = np.array([r[arm][j] for r in sel])
                d = v - np.array([r[base][j] for r in sel])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
                out += f"{v.mean():8.4f}{d.mean():+9.4f}{2 * se:7.4f}"
            lam = [r[f"lam {arm}"] for r in sel if f"lam {arm}" in r]
            out += f"{np.mean(lam):10.2f}" if lam else ""
            print(out)
        for arm in arms:
            if f"lam {arm}" not in sel[0]:
                continue
            lam = np.array([r[f"lam {arm}"] for r in sel])
            print(f"   share {arm}: " + " ".join(f"{g:g}:{np.mean(np.isclose(lam, g)):.2f}"
                                             for g in GRID))
    print(f"\n[{time.time() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
