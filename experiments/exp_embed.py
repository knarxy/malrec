"""Experiment 5: do synopsis embeddings add anything over tags?

TF-IDF did not (tested earlier). A distilled static embedding model captures
premise and tone rather than vocabulary overlap, so it is a fairer test of
the idea. potion-base-8M is ~30MB and runs on CPU in milliseconds, with no
torch dependency.

Two ways to use it, both measured:
  * kNN taste score - cosine-weighted average rating of the nearest already
    rated anime in synopsis space (one dense feature)
  * reduced embedding - the raw vector compressed by SVD and handed to the
    model as extra columns
"""
from __future__ import annotations

import numpy as np
from exp_implicit import HOLDOUTS, IMPLICIT, MU, ROWS, SCORED, recency, show
from model2vec import StaticModel
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge

from malrec.db import query
from malrec.eval import ndcg_at, spearman
from malrec.features import build_vocabulary, vectorise

OFFSETS = {"dropped": -1.5, "on_hold": -0.5, "completed": 0.0}
IMPLICIT_W = 0.5

ids = list(ROWS)
texts = {r["mal_id"]: (r["synopsis"] or "") for r in query(
    "SELECT mal_id, synopsis FROM anime WHERE mal_id = ANY(%s)", (ids,))}

print("embedding synopses...", flush=True)
model = StaticModel.from_pretrained("minishlab/potion-base-8M")
order = [i for i in ids if len(texts.get(i, "")) > 40]
E = model.encode([texts[i] for i in order], show_progress_bar=False)
E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
EMB = {i: E[k] for k, i in enumerate(order)}
print(f"  {len(EMB)}/{len(ids)} anime have a usable synopsis, dim {E.shape[1]}")

SVD_DIM = 48
SCALE = 0.3
svd = TruncatedSVD(n_components=SVD_DIM, random_state=0).fit(E)
RED = {i: v for i, v in zip(order, svd.transform(E))}
print(f"  SVD to {SVD_DIM} dims, explained variance {svd.explained_variance_ratio_.sum():.2f}")


def training_rows(cutoff, train_s):
    out = []
    for r in train_s:
        if r["mal_id"] in ROWS:
            out.append((r["mal_id"], float(r["score"]), recency(r["at"], cutoff)))
    for r in IMPLICIT:
        off = OFFSETS.get(r["status"])
        if off is None or r["at"] >= cutoff or r["mal_id"] not in ROWS:
            continue
        out.append((r["mal_id"], float(np.clip(MU + off, 1, 10)),
                    IMPLICIT_W * recency(r["at"], cutoff)))
    return out


def M_RED():
    import exp_embed as _m
    return _m.RED


def knn_feature(target: int, train: list[tuple[int, float, float]], tm: float, k=12) -> float:
    """Cosine-weighted deviation of the k nearest rated anime in synopsis space."""
    v = EMB.get(target)
    if v is None:
        return 0.0
    cand = [(float(v @ EMB[i]), y, w) for i, y, w in train if i in EMB and i != target]
    if not cand:
        return 0.0
    cand.sort(reverse=True)
    top = cand[:k]
    num = sum(max(s, 0) ** 2 * w * (y - tm) for s, y, w in top)
    den = sum(max(s, 0) ** 2 * w for s, y, w in top)
    return num / (den + 0.5) if den else 0.0


def evaluate(mode: str, holdouts=HOLDOUTS, alpha=30.0) -> dict:
    rhos, nds, rmses = [], [], []
    for h in holdouts:
        train_s, test_s = SCORED[:-h], SCORED[-h:]
        cutoff = test_s[0]["at"]
        train = training_rows(cutoff, train_s)
        test = [(r["mal_id"], float(r["score"])) for r in test_s if r["mal_id"] in ROWS]
        if len(train) < 20 or len(test) < 5:
            continue
        tm = float(np.average([y for _, y, _ in train], weights=[w for _, _, w in train]))

        vocab = build_vocabulary([ROWS[i] for i, _, _ in train])
        Xtr = vectorise([ROWS[i] for i, _, _ in train], vocab)
        Xte = vectorise([ROWS[i] for i, _ in test], vocab)

        def extra(mal_id, exclude_self):
            if mode == "knn":
                tr = [t for t in train if not (exclude_self and t[0] == mal_id)]
                return [knn_feature(mal_id, tr, tm)]
            if mode == "svd":
                v = M_RED().get(mal_id)
                return list(v * __import__('exp_embed').SCALE) if v is not None else [0.0] * SVD_DIM
            if mode == "both":
                tr = [t for t in train if not (exclude_self and t[0] == mal_id)]
                v = M_RED().get(mal_id)
                return [knn_feature(mal_id, tr, tm)] + (
                    list(v * __import__('exp_embed').SCALE) if v is not None else [0.0] * SVD_DIM)
            return []

        if mode != "none":
            Xtr = np.hstack([Xtr, np.array([extra(i, True) for i, _, _ in train])])
            Xte = np.hstack([Xte, np.array([extra(i, False) for i, _ in test])])

        y = np.array([v for _, v, _ in train]); w = np.array([v for _, _, v in train])
        est = Ridge(alpha=alpha).fit(Xtr, y, sample_weight=w)
        p = est.predict(Xte)
        yte = np.array([v for _, v in test])
        rhos.append(spearman(p, yte))
        nds.append(ndcg_at(p, list(yte), 10))
        rmses.append(float(np.sqrt(np.mean((p - yte) ** 2))))
    return {"rho": float(np.mean(rhos)), "ndcg": float(np.mean(nds)),
            "rmse": float(np.mean(rmses)), "per": rhos}


if __name__ == "__main__":
    print(f"\n{'configuration':<44}{'rho':>7}{'nDCG@10':>8}{'rmse':>8}")
    print("-" * 78)
    base = evaluate("none")
    show("tags only (current)", base)
    show("+ synopsis kNN feature", evaluate("knn"), base)
    show("+ synopsis SVD columns", evaluate("svd"), base)
    show("+ both", evaluate("both"), base)

    # The first pass scaled the embedding columns by 0.3 against alpha=30,
    # which shrinks them to irrelevance. Standardise them and sweep the
    # regularisation so the idea gets a fair test rather than being rejected
    # on a scaling artefact.
    print("\n-- standardised embedding columns, alpha sweep --")
    import exp_embed as M

    mu_v = np.mean(np.stack(list(RED.values())), axis=0)
    sd_v = np.std(np.stack(list(RED.values())), axis=0) + 1e-9
    M.RED = {i: (v - mu_v) / sd_v for i, v in RED.items()}
    M.SCALE = 1.0
    for alpha in (30.0, 100.0, 300.0, 1000.0):
        b = evaluate("none", alpha=alpha)
        print(f"   alpha={alpha:<7} baseline rho {b['rho']:.3f}")
        for mode in ("knn", "svd", "both"):
            show(f"      + {mode}", evaluate(mode, alpha=alpha), b)
