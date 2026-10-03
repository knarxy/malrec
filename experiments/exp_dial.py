"""What the personal relevance dial does to one real Discover list.

Re-ranks the user's stored Discover shortlist (rec_candidate) at several
settings of relevance_weight_personal, novelty at its configured weight,
through the production rerank(), and prints each top 20 with what entered
and left relative to the first setting.
"""
from __future__ import annotations

import argparse

import numpy as np

from malrec.config import settings
from malrec.db import one, query
from malrec.rank import Candidate, relevance_bonus, rerank
from malrec.surfaces import CAND_SQL, SPECS, _Share


def top(uid: int, rwp: float, n: int = 20) -> list[dict]:
    rows = [r for r in query(CAND_SQL, (uid, "discover")) if r["list_status"] is None]
    cands = []
    for r in rows:
        rel = float(relevance_bonus(_Share(r["personal_share"]), np.array([r["relevance_z"]]),
                                    rwp)[0])
        cands.append(Candidate(mal_id=r["mal_id"], title=r["title"], franchise_id=r["franchise_id"],
                               predicted=r["predicted"], novelty=r["novelty"],
                               final=r["predicted"] + rel, relevance=rel,
                               row={"mal_genres": r["mal_genres"], **r}))
    spec = SPECS["discover"]
    return [c.row for c in rerank(cands, spec.novelty_weight, spec.diversity_weight, n)]


def main(user: str, settings_: list[float]):
    uid = one("SELECT id FROM app_user WHERE mal_username=%s", (user,))["id"]
    lists = {w: top(uid, w) for w in settings_}
    base = {r["mal_id"] for r in lists[settings_[0]]}
    for w, rows in lists.items():
        ids = {r["mal_id"] for r in rows}
        pops = [r["mal_popularity"] or 99999 for r in rows]
        print(f"\n== dial {w:g}: median popularity #{int(np.median(pops))}, "
              f"{len(ids & base)}/20 shared with dial {settings_[0]:g} ==")
        for i, r in enumerate(rows, 1):
            mark = " " if r["mal_id"] in base else "+"
            print(f" {mark}{i:>2}. {(r['title_en'] or r['title'])[:48]:<48} "
                  f"pred {r['predicted']:.2f}  pop #{r['mal_popularity']}  "
                  f"rel {r['relevance_z']:+.1f}σ")
        gone = [r for r in lists[settings_[0]] if r["mal_id"] not in ids]
        if gone and w != settings_[0]:
            print("   left the top 20: " + "; ".join(
                (r["title_en"] or r["title"])[:40] for r in gone))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default=settings().malrec_user)
    ap.add_argument("--dial", default="0,0.5,1.0")
    a = ap.parse_args()
    main(a.user, [float(x) for x in a.dial.split(",")])
