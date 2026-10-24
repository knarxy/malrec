"""The in-memory discovery pool (recsys/pool.py) against the SQL rules it
stands in for: prerequisites through any format, main-format ones required,
and Up Next's continuations. No database."""
from __future__ import annotations

from malrec.recsys.pool import Relations, prequel_chains

# 1 -> 2 (OVA) -> 3 -> 4: each the prequel of the next; 5 has no relations;
# 6 is 1's main-format sequel too (a film), 7 a side story's sequel
PRE = {2: {1}, 3: {2}, 4: {3}, 6: {1}}
MAIN = {1, 3, 4, 5, 6}


def entry(m: int, status: str = "completed", score: int = 8) -> dict:
    return {"mal_id": m, "status": status, "score": score}


def test_chains_walk_any_format_but_require_main_ones():
    chains = prequel_chains(PRE, MAIN, [1, 2, 3, 4, 5, 6])
    assert chains[3] == {1}                     # the OVA in between is not required
    assert chains[4] == {1, 3}
    assert chains[6] == {1}
    assert 1 not in chains and 5 not in chains  # nothing before them
    assert 2 not in chains or chains[2] == {1}


def test_a_cycle_never_makes_a_title_its_own_prerequisite():
    chains = prequel_chains({1: {2}, 2: {1}}, {1, 2}, [1, 2])
    assert chains == {1: {2}, 2: {1}}


def test_unmet_prerequisites_and_continuations_leave_the_pool():
    rel = Relations(prequel_chains(PRE, MAIN, [3, 4, 6]), {1: frozenset({6}), 3: frozenset({4})})
    # nothing watched: every sequel waits for its first season
    assert rel.exclusions([]) == {3, 4, 6}
    # 1 completed and liked: 3 and 6 are open, 6 is Up Next's
    out = rel.exclusions([entry(1, score=9)])
    assert out == {4, 6}
    # 1 completed but rated under the bar: 6 stays a discovery
    assert rel.exclusions([entry(1, score=5), entry(5, score=9), entry(5, score=9)]) == {4}
    # 1 only watching: the chain is not met
    assert rel.exclusions([entry(1, "watching", 0)]) == {3, 4, 6}


def test_the_continuation_bar_follows_the_users_mean():
    rel = Relations({}, {1: frozenset({6})})
    # mean 9: the bar is 8, a 7 does not carry the sequel over to Up Next
    assert rel.exclusions([entry(1, score=7), entry(2, score=10), entry(3, score=10)]) == set()
    # never below 6
    assert rel.exclusions([entry(1, score=6), entry(2, score=3)]) == {6}
    assert rel.exclusions([entry(1, score=5), entry(2, score=3)]) == set()
