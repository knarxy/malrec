"""When should a user's own display calibration be trusted?

Each app user's shown score is mapped through a line fitted on their own
newest ratings (train_hybrid -> calibration_from). For one new user that
holdout carried no signal - a list imported in one go, newest 25 nearly all
8s, rho 0.03 - and the fit (slope 0.23) squeezed every prediction to ~8.2,
so the bonuses alone decided the order. Thin lists already use the line
measured on population users of the same list size (size_calibration).

Per sampled user, oldest first:   train | calibrate (40) | future (25)
The calibrate block is predicted by a model fitted on the training part -
the production holdout - and gives the own line and its rho. The future
block is predicted by a model fitted on everything before it, then mapped by

  OWN         the own line, always (production today)
  POP         the population line for this list size
  GATE r s    OWN when the calibrate block's rho >= r and slope >= s, else POP

Scored on the future block: RMSE and mean bias of the shown score, the share
of users the gate sends to POP, and RMSE within that share. The population
model saw these users' lists (in-sample for every arm alike).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from experiments.exp_retrieval import load_entries
from malrec.config import settings
from malrec.eval import spearman
from malrec.recsys.hybrid import Example, fit_user, recency_weight
from malrec.recsys.service import (
    active_global,
    blend_mode,
    blend_opts,
    calibration_from,
    item_store,
    size_calibration,
)

CFG = settings()
CAL_N, FUTURE_N, MIN_TRAIN = 40, 25, 60
GATES = [(0.2, 0.4), (0.3, 0.5), (0.4, 0.5), (0.3, 0.7)]


def examples(seen, cutoff, hl=0.5):
    scored = [Example(e["mal_id"], float(e["score"]), recency_weight(e["at"], cutoff, hl), e["at"])
              for e in seen if e["score"] > 0]
    if len(scored) < 3:
        return scored, []
    m = float(np.average([x.score for x in scored], weights=[x.weight for x in scored]))
    implicit = [Example(e["mal_id"], float(min(max(m + CFG.implicit_offsets[e["status"]], 1), 10)),
                        CFG.implicit_weight * recency_weight(e["at"], cutoff, hl), e["at"])
                for e in seen if e["score"] == 0 and e["status"] in CFG.implicit_offsets]
    return scored, implicit


def predict(gm, store, seen, cutoff, items, n_total):
    sc, im = examples(seen, cutoff)
    um = fit_user(gm.pop, gm.stacker, store, sc, im, mode=blend_mode(n_total),
                  blend_of="personal", **{**blend_opts(), "size_n": n_total},
                  listed={e["mal_id"]: 1.0 for e in seen})
    return um.predict(items)


def main(n_users: int, seed: int):
    t0 = time.time()
    gm, store = active_global(), item_store()
    table = gm.meta.get("size_calibration")
    entries = load_entries()
    rng = np.random.default_rng(seed)
    users = [u for u, es in entries.items()
             if sum(e["score"] > 0 for e in es) >= MIN_TRAIN + CAL_N + FUTURE_N]
    rng.shuffle(users)
    rows = []
    for u in users[:n_users]:
        es = sorted(entries[u], key=lambda e: e["at"])
        scored = [e for e in es if e["score"] > 0]
        future, cal = scored[-FUTURE_N:], scored[-FUTURE_N - CAL_N:-FUTURE_N]
        n = len(scored)
        # calibrate block, predicted from what came before it
        cut_c = cal[0]["at"]
        seen_c = [e for e in es if e["at"] < cut_c]
        pc = predict(gm, store, seen_c, cut_c, [e["mal_id"] for e in cal], n)
        yc = np.array([float(e["score"]) for e in cal])
        okc = ~np.isnan(pc)
        if okc.sum() < 20:
            continue
        own = calibration_from(list(pc[okc]), list(yc[okc]))
        rho_c = spearman(pc[okc], yc[okc]) if np.std(yc[okc]) > 0 else 0.0
        # future block, predicted from everything before it
        cut_f = future[0]["at"]
        seen_f = [e for e in es if e["at"] < cut_f]
        pf = predict(gm, store, seen_f, cut_f, [e["mal_id"] for e in future], n)
        yf = np.array([float(e["score"]) for e in future])
        okf = ~np.isnan(pf)
        if okf.sum() < 10:
            continue
        pop = size_calibration(table, n) or (1.0, 0.0)
        rows.append({"pf": pf[okf], "yf": yf[okf], "own": (own["slope"], own["intercept"]),
                     "own_ok": "rmse_raw" in own, "pop": pop, "rho": rho_c,
                     "slope": own["slope"], "cal_sd": float(np.std(yc[okc]))})
    print(f"{len(rows)} users ({time.time() - t0:.0f}s)\n")

    def score(choose):
        err, bias = [], []
        for r in rows:
            a, b = choose(r)
            shown = np.clip(a * r["pf"] + b, 1, 10)
            err.append(float(np.sqrt(np.mean((shown - r["yf"]) ** 2))))
            bias.append(float(np.mean(shown - r["yf"])))
        return np.mean(err), np.mean(bias)

    own = lambda r: r["own"] if r["own_ok"] else r["pop"]
    pop = lambda r: r["pop"]
    print(f"{'arm':<14}{'rmse':>8}{'bias':>8}{'to POP':>9}{'rmse(own|gated)':>17}{'rmse(pop|gated)':>17}")
    for name, fn in (("OWN", own), ("POP", pop)):
        e, b = score(fn)
        print(f"{name:<14}{e:8.3f}{b:+8.3f}")
    for r_min, s_min in GATES:
        gated = [r for r in rows if r["own_ok"] and (r["rho"] < r_min or r["slope"] < s_min)]
        fn = (lambda r, r_min=r_min, s_min=s_min:
              r["own"] if r["own_ok"] and r["rho"] >= r_min and r["slope"] >= s_min else r["pop"])
        e, b = score(fn)
        sub_own = np.mean([np.sqrt(np.mean((np.clip(r["own"][0] * r["pf"] + r["own"][1], 1, 10)
                                           - r["yf"]) ** 2)) for r in gated]) if gated else np.nan
        sub_pop = np.mean([np.sqrt(np.mean((np.clip(r["pop"][0] * r["pf"] + r["pop"][1], 1, 10)
                                           - r["yf"]) ** 2)) for r in gated]) if gated else np.nan
        print(f"GATE {r_min} {s_min:<5}{e:8.3f}{b:+8.3f}{len(gated) / len(rows):9.1%}"
              f"{sub_own:17.3f}{sub_pop:17.3f}")
    slopes = np.array([r["slope"] for r in rows if r["own_ok"]])
    print(f"\nown slopes: <0.5 {np.mean(slopes < 0.5):.1%}, <0.3 {np.mean(slopes < 0.3):.1%}; "
          f"rho<0.3 {np.mean([r['rho'] < 0.3 for r in rows]):.1%}")
    print(f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=400)
    ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    main(a.users, a.seed)
