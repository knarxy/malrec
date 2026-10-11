"""Where does the population fit's memory go? Peak allocation of each stage
(tracemalloc sees numpy's buffers too), on the benchmark snapshot's 72 %.

  python -m experiments.prof_popmem
"""
from __future__ import annotations

import resource
import time
import tracemalloc

import numpy as np

from experiments import bench
from malrec.recsys.cf import CFParams, PopulationModel
from malrec.recsys.hybrid import fit_stacker


def main():
    t0 = time.time()
    entries, static = bench.load_snapshot()
    store = static["store"]
    users = entries.users_with_scored(15)
    np.random.default_rng(bench.SEED).shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B = users[:nA], users[nA:nA + nB]
    u, m, sc = entries.columns(A)
    oa, oi = u.astype(np.int64), m.astype(np.int64)
    r = sc > 0
    print(f"{len(A)} users, {int(r.sum()):,} ratings, {len(oa):,} entries; process "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB so far", flush=True)
    tracemalloc.start()
    pop = PopulationModel(CFParams())

    def stage(name, fn):
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        t = time.time()
        fn()
        cur, peak = tracemalloc.get_traced_memory()
        print(f"  {name:<12} peak +{(peak - before) / 2**20:7.0f} MB, kept +{(cur - before) / 2**20:6.0f}"
              f" MB, {time.time() - t:5.0f}s", flush=True)

    stage("fit (bias, kNN, ALS)", lambda: pop.fit(oa[r], oi[r], sc[r].astype(np.float64)))
    stage("occurrence", lambda: pop.fit_occurrence(oa, oi))
    stage("EASE", lambda: pop.fit_ease(oa, oi))
    import scipy.sparse as sp
    print("  EASE line by line:")
    marks = []

    def mark(name):
        cur, peak = tracemalloc.get_traced_memory()
        marks.append(name)
        print(f"    {name:<28} now {cur / 2**20:7.0f} MB, peak since last {peak / 2**20:7.0f} MB",
              flush=True)
        tracemalloc.reset_peak()
    tracemalloc.reset_peak()
    pairs = np.unique(np.column_stack([oa, oi]), axis=0); mark("unique pairs")
    items, counts = np.unique(pairs[:, 1], return_counts=True)
    keep = items[counts >= 50]; mark(f"keep ({len(keep)} items)")
    mm = np.isin(pairs[:, 1], keep)
    uu = np.unique(pairs[mm, 0], return_inverse=True)[1]
    ii = np.searchsorted(keep, pairs[mm, 1]); mark("indices")
    X = sp.csr_matrix((np.ones(len(uu), dtype=np.float32), (uu, ii)),
                      shape=(uu.max() + 1, len(keep))); mark("X")
    Xc, XT = X.tocsc(), X.T.tocsr()
    G = np.empty((len(keep), len(keep)), dtype=np.float32)
    for c0 in range(0, len(keep), 1024):
        G[:, c0:c0 + 1024] = XT @ Xc[:, c0:c0 + 1024].toarray()
    mark(f"G dense {G.dtype}")
    G[np.diag_indices_from(G)] += 400.0
    from scipy.linalg import inv
    P = inv(G, overwrite_a=True, check_finite=False); mark(f"scipy inv -> {P.dtype}")
    del G; mark("del G")
    held = {x: [(e["mal_id"], float(e["score"]), e["at"]) for e in entries[x] if e["score"] > 0]
            for x in B}
    stage("stacker", lambda: fit_stacker(pop, held, store, seed=bench.SEED))
    print(f"process peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
