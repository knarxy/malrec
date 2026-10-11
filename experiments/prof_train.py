"""Where does a user's training time go? cProfile of train_hybrid without
persisting anything (python -m experiments.prof_train USER)."""
import cProfile
import pstats
import sys
import time

from malrec.db import one
from malrec.model import train_hybrid
from malrec.recsys.service import active_global, item_store

name = sys.argv[1]
uid = one("SELECT id FROM app_user WHERE mal_username=%s", (name,))["id"]
n = one("SELECT count(*) AS n FROM list_entry WHERE user_id=%s AND score>0", (uid,))["n"]
t0 = time.time()
active_global(), item_store()                 # warm the caches, as the worker has them
print(f"{name}: {n} ratings, caches warm in {time.time() - t0:.1f}s", flush=True)
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
train_hybrid(uid, persist=False)
pr.disable()
print(f"train_hybrid: {time.time() - t0:.1f}s", flush=True)
st = pstats.Stats(pr)
st.sort_stats("cumulative").print_stats(28)
st.sort_stats("tottime").print_stats(15)
