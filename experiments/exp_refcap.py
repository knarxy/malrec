"""The reference profile at its real size: personal-share cap x population model.

The benchmark evaluates the reference profile's 8 splits at the size of each
training window (92-127 of its 152 ratings), where the size curve keeps the
personal share far below the cap - so blend_lam_max cannot show there. The
gate and production use the full count (temporal_holdout's size_n), which
puts this profile at the cap. This is the gate's own measurement, per stored
population model and cap.

  python -m experiments.exp_refcap 11 14 [--caps 0.5,0.6,0.7,0.8,0.9] [--n0 100]
"""
from __future__ import annotations

import argparse

import numpy as np

from malrec.config import settings
from malrec.db import require_one
from malrec.recsys.service import GATE_SPLITS, _half_life, load_global, temporal_holdout


def main(ids: list[int], caps: list[float], n0: float | None):
    user = settings().malrec_user
    uid = require_one("SELECT id FROM app_user WHERE mal_username=%s", (user,))["id"]
    hl = _half_life(uid)
    print(f"{'model':>6}{'cap':>6}{'rho':>9}   per split")
    for gid in ids:
        gm = load_global(gid)
        for cap in caps:
            r = temporal_holdout(uid, GATE_SPLITS, gm=gm, half_life=hl, lam_max=cap,
                                 **({"size_n0": n0} if n0 else {}))
            per = [x["spearman"] for x in r["per_holdout"]]
            print(f"{gid:6d}{cap:6.2f}{np.mean(per):9.4f}   " + " ".join(f"{x:.3f}" for x in per),
                  flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", type=int, nargs="+")
    ap.add_argument("--caps", default="0.5,0.6,0.7,0.8,0.9")
    ap.add_argument("--n0", type=float, default=None, help="blend handover (blend_size_n0)")
    a = ap.parse_args()
    main(a.ids, [float(x) for x in a.caps.split(",")], a.n0)
