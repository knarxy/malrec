"""Two-tower retrieval: does a learned user/item embedding help the lists?

Production decides *which* titles get near the top with a relevance signal:
co-occurrence and EASE over the sampled public lists, z-scored and mixed
(UserModel.relevance). Both only know a title through the lists it is on.
A two-tower model learns one vector per user and per title so that a user's
vector points at titles they go on to like; the title side also reads the
title's tags, genres, studios, year and audience, so it can place a title
few lists contain.

Model (PyTorch, CPU, trained here - nothing ships):
  item     ID embedding (titles on >= 5 training lists; others have none)
           + projection of tag vector, genres, studios, year, size, score;
           the ID part is dropped 20 % of the time in training so the
           content part learns to stand in for it -> MLP -> unit vector
  user     weighted mean of the same item inputs over their list (a title
           they rated above their mean counts more, a dropped one less)
           -> MLP -> unit vector. No per-user parameters: a new list is
           embedded without retraining.
  training predict a title the user liked (>= max(own mean, 7)) from up to
           64 other titles on their list; in-batch softmax with the
           popularity (logQ) correction, so it does not just learn "popular".

Everything else is experiments/bench.py: the frozen snapshot, its split
(population model on 72 % of users, stacker 13 %, 722 held out), the
held-out users at 25 / 150 / 400 entries, the reference profile's 8 splits,
the Safe Bets build through the production code, the same metrics.

Arms, per user, paired with PROD:
  PROD       the shipped Safe Bets build
  TT         the two-tower ranking alone (no rating model, no rerank)
  PROD+TT    production, relevance = the two-tower score (z-scored)
  PROD+MIX   production, relevance = mean of the shipped one and two-tower

Phase "cold": 10 % of the Safe Bets pool is hidden from everything fitted
(population model, stacker, two-tower IDs) - titles nobody in the training
data has listed, like a new season. Reported additionally: recall@50 of
liked future titles among those cold titles (pooled over users).

  python -m experiments.exp_twotower [--phase warm|cold|both] [--epochs 40]
(needs torch on the path: PYTHONPATH=...:/work/experiments/.torchlib)
"""
from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn

from experiments.bench import (
    BUDGETS,
    REF_SPLITS,
    SEED,
    eligible,
    examples,
    load_snapshot,
    population,
    row_of,
    safe_bets,
    user_metrics,
)
from malrec.recsys.scorer import HybridScorer

STATUS_W = {"completed": 0.7, "watching": 0.6, "on_hold": 0.4, "dropped": 0.15,
            "plan_to_watch": 0.3}
HIST = 64
ARMS = ("PROD", "TT", "PROD+TT", "PROD+MIX")
MAX_EVAL = 100


def weight(e: dict, mean: float) -> float:
    if e["score"] > 0:
        return float(np.clip(0.5 + (e["score"] - mean) / 4, 0.1, 1.5))
    return STATUS_W.get(e["status"], 0.3)


def z(v: np.ndarray) -> np.ndarray:
    sd = float(np.nanstd(v))
    return (v - np.nanmean(v)) / sd if sd > 1e-9 else np.zeros_like(v)


# ---------------------------------------------------------------- features --

def side_features(store, items: list[int]) -> np.ndarray:
    rows = [store.rows[m] for m in items]
    tags = np.zeros((len(items), store.vecs.shape[1]), dtype=np.float32)
    for k, m in enumerate(items):
        j = store.vec_index.get(m)
        if j is not None:
            tags[k] = store.vecs[j]
    genres = sorted({g for r in rows for g in (r.get("mal_genres") or [])})
    gi = {g: k for k, g in enumerate(genres)}
    studios = [s for s, _ in Counter(s for r in rows for s in (r.get("mal_studios") or []))
               .most_common(200)]
    si = {s: k for k, s in enumerate(studios)}
    G = np.zeros((len(items), len(genres)), dtype=np.float32)
    S = np.zeros((len(items), len(studios) + 1), dtype=np.float32)
    N = np.zeros((len(items), 6), dtype=np.float32)
    for k, r in enumerate(rows):
        for g in r.get("mal_genres") or []:
            G[k, gi[g]] = 1.0
        st = [si[s] for s in (r.get("mal_studios") or []) if s in si]
        for s in st:
            S[k, s] = 1.0
        if not st:
            S[k, -1] = 1.0
        y = r.get("season_year")
        N[k] = [((y or 2010) - 2010) / 10, float(y is None),
                np.log1p(r.get("mal_num_scoring_users") or 0) / 12,
                ((r.get("mal_mean") or 7.0) - 7.0),
                float(r.get("format_class") == "main"), float(r.get("format_class") == "side")]
    return np.hstack([tags, G, S, N])


class TwoTower(nn.Module):
    def __init__(self, n_ids: int, side: torch.Tensor, ids: torch.Tensor, d: int = 64):
        super().__init__()
        self.side, self.ids = side, ids              # per item: feature row, ID index
        self.id_emb = nn.Embedding(n_ids, d, padding_idx=0)
        self.side_proj = nn.Linear(side.shape[1], d)
        self.item = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.user = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.id_drop = 0.2

    def base(self, items: torch.Tensor) -> torch.Tensor:
        e = self.id_emb(self.ids[items])
        if self.training:
            keep = (torch.rand(e.shape[:-1] + (1,)) > self.id_drop).float()
            e = e * keep
        return torch.cat([e, self.side_proj(self.side[items])], -1)

    def item_vec(self, items: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.item(self.base(items)), dim=-1)

    def user_vec(self, hist: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        b = self.base(hist)                                       # B x H x 2d
        pooled = (b * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp_min(1e-6)
        return nn.functional.normalize(self.user(pooled), dim=-1)


def train_tt(entries, A, store, hidden: set[int], epochs: int, log) -> tuple:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    items = sorted(store.rows)
    pos = {m: k for k, m in enumerate(items)}
    counts = Counter(e["mal_id"] for u in A for e in entries[u] if e["mal_id"] not in hidden)
    id_items = [m for m in items if counts.get(m, 0) >= 5]
    ids = np.zeros(len(items), dtype=np.int64)
    for k, m in enumerate(id_items, 1):
        ids[pos[m]] = k
    side = torch.tensor(side_features(store, items))
    model = TwoTower(len(id_items) + 1, side, torch.tensor(ids))
    # training lists without the hidden titles
    users = []
    for u in A:
        es = [e for e in entries[u] if e["mal_id"] in pos and e["mal_id"] not in hidden]
        sc = [e["score"] for e in es if e["score"] > 0]
        if len(es) < 5 or not sc:
            continue
        mean = float(np.mean(sc))
        liked = [k for k, e in enumerate(es) if e["score"] >= max(mean, 7.0)]
        if liked:
            users.append((np.array([pos[e["mal_id"]] for e in es]),
                          np.array([weight(e, mean) for e in es], dtype=np.float32), liked))
    tq = Counter(int(it[k]) for it, _, liked in users for k in liked)
    logq = torch.zeros(len(items))
    tot = sum(tq.values())
    for k, c in tq.items():
        logq[k] = float(np.log(c / tot))
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    per_user, B, tau = 8, 512, 0.07
    for ep in range(epochs):
        model.train()
        ex = [(ui, int(rng.choice(liked))) for ui, (_, _, liked) in enumerate(users)
              for _ in range(per_user)]
        rng.shuffle(ex)
        loss_sum = 0.0
        for s in range(0, len(ex), B):
            batch = ex[s:s + B]
            H = np.zeros((len(batch), HIST), dtype=np.int64)
            W = np.zeros((len(batch), HIST), dtype=np.float32)
            T = np.zeros(len(batch), dtype=np.int64)
            for r, (ui, t) in enumerate(batch):
                it, wt, _ = users[ui]
                others = np.delete(np.arange(len(it)), t)
                pick = rng.choice(others, min(HIST, len(others)), replace=False)
                H[r, :len(pick)], W[r, :len(pick)] = it[pick], wt[pick]
                T[r] = it[t]
            u = model.user_vec(torch.tensor(H), torch.tensor(W))
            v = model.item_vec(torch.tensor(T))
            logits = u @ v.T / tau - logq[torch.tensor(T)].unsqueeze(0)
            same = torch.tensor(T).unsqueeze(0) == torch.tensor(T).unsqueeze(1)
            logits = logits.masked_fill(same & ~torch.eye(len(T), dtype=torch.bool), -1e9)
            loss = nn.functional.cross_entropy(logits, torch.arange(len(T)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * len(batch)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"    epoch {ep:>2}: loss {loss_sum / len(ex):.3f}")
    model.eval()
    with torch.no_grad():
        table = model.item_vec(torch.arange(len(items))).numpy()

    def user_vector(hist: list[dict]) -> np.ndarray | None:
        sc = [e["score"] for e in hist if e["score"] > 0]
        mean = float(np.mean(sc)) if sc else 7.0
        es = [e for e in hist if e["mal_id"] in pos][-512:]
        if not es:
            return None
        h = torch.tensor([[pos[e["mal_id"]] for e in es]])
        w = torch.tensor([[weight(e, mean) for e in es]], dtype=torch.float32)
        with torch.no_grad():
            return model.user_vec(h, w).numpy()[0]

    def scores(vec: np.ndarray, ids_: list[int]) -> np.ndarray:
        return table[[pos[m] for m in ids_]] @ vec

    return user_vector, scores, len(id_items)


class TTScorer(HybridScorer):
    """Production scorer whose relevance comes (partly) from the two tower."""
    tt_scores = None
    tt_mode = "replace"

    def relevance(self, ids: list[int]) -> np.ndarray:
        t = z(self.tt_scores(ids))
        if self.tt_mode == "replace":
            return t
        return 0.5 * (super().relevance(ids) + t)


# --------------------------------------------------------------- evaluate --

def evaluate(entries, static, C, pop, stacker, tt_user, tt_scores, hidden, log):
    store, derived = static["store"], static["derived"]
    pool_all = [m for m in eligible(store, derived)]
    R = {}

    def genres(ids):
        return {g for m in ids for g in ((store.rows.get(m) or {}).get("mal_genres") or [])}

    def one_user(hist, seen_cutoff, test, listed, liked, disliked, cal, M, tops, cold):
        sc, im, li = examples(hist, seen_cutoff)
        if len(sc) < 5:
            return False
        pool_rows = [row_of(store, m) for m in pool_all if m not in listed]
        fam = genres(e["mal_id"] for e in hist)
        items = [m for m, _ in test]
        truth = np.array([s for _, s in test], dtype=float)
        vec = tt_user(hist)
        from malrec.recsys.hybrid import fit_user
        from malrec.recsys.service import blend_mode, blend_opts
        um = fit_user(pop, stacker, store, sc, im, mode=blend_mode(len(sc)),
                      blend_of="personal", **blend_opts(), listed=li)
        out = {}
        scorer, top = safe_bets(pop, stacker, store, sc, im, li, pool_rows, cal, fam, um=um)
        out["PROD"] = (scorer, top)
        if vec is not None:
            ids_ = [r["mal_id"] for r in pool_rows]
            s = tt_scores(vec, ids_)
            out["TT"] = (scorer, [ids_[k] for k in np.argsort(-s)[:60]])
            def fn(q, v=vec):
                return tt_scores(v, q)
            for arm, mode in (("PROD+TT", "replace"), ("PROD+MIX", "mix")):
                out[arm] = safe_bets(pop, stacker, store, sc, im, li, pool_rows, cal, fam,
                                     scorer_cls=TTScorer, um=um, tt_scores=fn, tt_mode=mode)
        else:
            return False
        for arm, (sc_, top_) in out.items():
            m = user_metrics(sc_, top_, items, truth, liked, disliked)
            if arm == "TT":
                m.pop("rho", None)
                m.pop("rmse", None)
            M[arm].append(m)
            tops[arm].append(top_[:20])
            if hidden:
                lc = liked & hidden
                cold[arm][0] += len(set(top_[:50]) & lc)
                cold[arm][1] += len(lc)
        return True

    for budget in BUDGETS:
        M, tops, cold = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0])
        urng = np.random.default_rng(SEED + budget)
        from malrec.recsys.hybrid import split_user
        n = 0
        for u in C:
            if n >= MAX_EVAL:
                break
            es = entries[u]
            scored_all = [(e["mal_id"], float(e["score"]), e["at"]) for e in es if e["score"] > 0]
            _, test_s, cutoff = split_user(scored_all, urng)
            mean = float(np.mean([s for _, s, _ in scored_all]))
            test_ids = {m for m, _, _ in test_s}
            history = [e for e in es if e["mal_id"] not in test_ids and e["at"] < cutoff]
            if len(history) < budget:
                continue
            seen = [history[i] for i in urng.permutation(len(history))[:budget]]
            listed = {e["mal_id"] for e in es} - test_ids
            liked = {m for m, s, _ in test_s if s >= max(mean, 7.0)}
            disliked = {m for m, s, _ in test_s if s < mean - 1}
            disliked |= {e["mal_id"] for e in es if e["status"] == "dropped" and e["at"] >= cutoff}
            if one_user(seen, cutoff, [(m, s) for m, s, _ in test_s], listed, liked, disliked,
                        (1.0, 0.0), M, tops, cold):
                n += 1
        R[f"sampled {budget}"] = (M, tops, cold)
        log(f"  sampled {budget}: {n} users")

    ref = static["ref"]
    le = ref["le"]
    rated = sorted([r for r in le if r["score"] > 0], key=lambda r: r["at"])
    mean = float(np.mean([r["score"] for r in rated]))
    M, tops, cold = defaultdict(list), defaultdict(list), defaultdict(lambda: [0, 0])
    for h in REF_SPLITS:
        test = rated[-h:]
        cutoff = test[0]["at"]
        test_ids = {r["mal_id"] for r in test}
        hist = [r for r in le if r["mal_id"] not in test_ids and r["at"] < cutoff]
        one_user(hist, cutoff, [(r["mal_id"], float(r["score"])) for r in test],
                 {r["mal_id"] for r in le} - test_ids,
                 {r["mal_id"] for r in test if r["score"] >= max(mean, 7.0)},
                 {r["mal_id"] for r in test if r["score"] < mean - 1}, ref["cal"], M, tops, cold)
    R["reference"] = (M, tops, cold)
    return R


def report(title, R, store, log):
    log(f"\n==== {title}")
    for g, (M, tops, cold) in R.items():
        log(f"\n== {g} ({len(M['PROD'])} {'splits' if g == 'reference' else 'users'})")
        log(f"   {'arm':<10}{'rho':>16}{'r50':>16}{'hit10':>16}{'bad10':>16}{'overlap':>9}"
            f"{'pop#':>7}{'cold r50':>10}")
        base = M["PROD"]
        for arm in ARMS:
            if arm not in M or not M[arm]:
                continue
            cells = []
            for k, higher in (("rho", True), ("r50", True), ("hit10", True), ("bad10", False)):
                pairs = [(a[k], b[k]) for a, b in zip(M[arm], base) if k in a and k in b]
                if not pairs:
                    cells.append(f"{'—':>16}")
                    continue
                now = np.mean([a for a, _ in pairs])
                if arm == "PROD":
                    cells.append(f"{now:16.4f}")
                    continue
                d = np.array([a - b for a, b in pairs])
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("inf")
                good = d.mean() > 0 if higher else d.mean() < 0
                mark = ("+" if good else "-") if abs(d.mean()) > 2 * se else " "
                cells.append(f"{now:9.4f}{d.mean():+6.3f}{mark}")
            sets = [set(t) for t in tops[arm]]
            pairs_ = list(itertools.islice(itertools.combinations(range(len(sets)), 2), 3000))
            ov = np.mean([len(sets[a] & sets[b]) / len(sets[a] | sets[b]) for a, b in pairs_]) \
                if pairs_ else float("nan")
            pp = np.median([store.rows[m].get("mal_popularity") or 99999
                            for t in tops[arm] for m in t]) if tops[arm] else float("nan")
            c = cold.get(arm)
            cr = f"{c[0] / c[1]:10.3f}" if c and c[1] else f"{'':>10}"
            log(f"   {arm:<10}{''.join(cells)}{ov:9.3f}{pp:7.0f}{cr}")
    log("   (+/- : better/worse than PROD by more than two standard errors, paired)")


def main(phase: str, epochs: int):
    t0 = time.time()

    def log(msg):
        print(f"{msg}" if not msg.startswith("  ") else f"{msg}  [{time.time() - t0:.0f}s]",
              flush=True)

    entries, static = load_snapshot()
    store = static["store"]
    rng = np.random.default_rng(SEED)
    users = entries.users_with_scored(15)
    rng.shuffle(users)
    nA, nB = int(len(users) * 0.72), int(len(users) * 0.13)
    A, B, C = users[:nA], users[nA:nA + nB], users[nA + nB:]
    log(f"{len(users)} users ({len(A)} fit / {len(B)} stack / {len(C)} eval); "
        f"torch {torch.__version__}, {torch.get_num_threads()} threads")

    if phase in ("warm", "both"):
        (pop, stacker), how = population(entries, store, A, B)
        log(f"  population model {how}")
        tt_user, tt_scores, n_ids = train_tt(entries, A, store, set(), epochs, log)
        log(f"  two-tower trained ({n_ids} titles with an ID)")
        R = evaluate(entries, static, C, pop, stacker, tt_user, tt_scores, set(), log)
        report("WARM: every title seen in training", R, store, log)

    if phase in ("cold", "both"):
        pool = eligible(store, static["derived"])
        crng = np.random.default_rng(SEED + 99)
        hidden = set(crng.choice(pool, len(pool) // 10, replace=False).tolist())
        (pop, stacker), _ = population(entries, store, A, B, drop=frozenset(hidden))
        log(f"  population model fitted without {len(hidden)} hidden titles")
        tt_user, tt_scores, n_ids = train_tt(entries, A, store, hidden, epochs, log)
        log(f"  two-tower trained ({n_ids} titles with an ID)")
        R = evaluate(entries, static, C, pop, stacker, tt_user, tt_scores, hidden, log)
        report(f"COLD: {len(hidden)} pool titles hidden from all training", R, store, log)
    log("  done")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="both", choices=("warm", "cold", "both"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--users", type=int, default=MAX_EVAL, help="held-out users per size")
    a = ap.parse_args()
    MAX_EVAL = a.users
    main(a.phase, a.epochs)
