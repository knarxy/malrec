# How malrec works

The model, the ranking, how each part was measured, and how to run it. The
README has the short version; experiment code and logs are in `experiments/`,
the literature behind the choices in `RESEARCH.md`.

## Services

| service | port | what it is |
|---|---|---|
| `app` | 3000 | React + Vite, served by nginx, which also proxies `/api` |
| `api` | 8000 | FastAPI; ingestion, model, ranking, sign-in |
| `worker` | – | background tasks from a Postgres queue: rebuilds, list syncs, onboarding (`malrec.tasks`) |
| `db` | 5433 | Postgres 17 + pgvector (loopback only) |
| `lab` | – | profile `lab`: experiments and long jobs against the working tree |

The app reaches the API over the internal network through nginx, so the
browser makes only same-origin requests. nginx re-resolves the API through
Docker's DNS, so recreating the API container never leaves the app on 502.

## Sign in with MyAnimeList

OAuth2 authorization code with PKCE, against MAL's own endpoints
(`https://myanimelist.net/v1/oauth2/authorize` and `/v1/oauth2/token` — the
OAuth server is v1 even though the resource API is v2).

**Setup:** at <https://myanimelist.net/apiconfig>, edit your app and set
*App Redirect URL* to exactly the value of `MAL_REDIRECT_URI`
(default `http://localhost:3000/api/auth/callback`). If they differ, MAL
answers with `401 invalid_client` plus `WWW-Authenticate: Basic`, which a
browser shows as a password prompt that can never succeed. `/auth/login`
checks for that first and brings you back to the app with instructions
instead.

Once signed in, on your own profile:

- **+ Plan to watch / ✓ Planned** on every card;
- **rate from the app** — a 1–10 picker on each card and in the details
  panel (which search results open too, so anything you have seen can be
  rated) writes the score to MAL. An unlisted or planned title becomes
  *completed* (with its episode count); a show you are watching, holding or
  dropped keeps its status. The title leaves your lists at once and your
  recommendations are rebuilt in the background (several quick ratings
  cost at most two rebuilds);
- **Sync with MAL** re-reads your whole list and rebuilds. The API allows it
  once per 5 minutes per account (`SYNC_COOLDOWN`, hard-coded) and the button
  counts down; the app stops waiting on a sync after 3 minutes
  (`tasks.UI_TIMEOUT_S`);
- **language (English / Deutsch) and filters** are saved
  to your account (`/me/prefs`) and follow you to any browser. Without
  signing in they are kept in the browser only.

Design choices:

- tokens stay server-side; the browser holds only an opaque HttpOnly cookie,
  and the database stores only its hash; over https the cookie is
  `__Host-malrec_session`, which no other subdomain can set or shadow;
- `state` is single-use, expires after ten minutes and belongs to the browser
  that started the sign-in (the hash of a nonce in a short-lived cookie): a
  state alone, sent to someone else, used to sign them into the account it
  was issued for (login CSRF, audit 2026-10-23); state-changing calls also
  require an `X-Requested-With` header;
- an account is a MAL *user*, found by MAL's numeric id at every sign-in: a
  rename on MAL renames the account, and a name that changes hands gives the
  newcomer a new account (an admin name in another user's hands is refused);
- un-queueing removes an entry **only while it is still Plan to watch** — a
  show you have since started or finished is never deleted from your list;
- signing in also reads **private** lists, which username lookup cannot.

## How it works

### 1. Data

Default rule: nothing is fetched that nobody asked for, and nothing is ever
fetched twice — every item is stamped as it lands and all jobs resume.

| step | cost | when |
|---|---|---|
| your list | 1 request | on sign-in / `sync list` |
| MAL detail for your list | 1 per new entry | first sync, then only new entries |
| AniList for a shortlist | ~10 batched requests | lazily, only what lacks it |

On top of that, **one authorised full fetch** (`malrec sync full`) built the
population layer. Rates come from the providers' documentation:

- **AniList** documents 90 req/min, currently degraded to 30; the client
  reads the live limit from `X-RateLimit-Limit` on every response and paces
  evenly to it, so it speeds up by itself when the limit is restored.
- **MyAnimeList** publishes no limit anywhere (API reference, auth docs,
  License Agreement, response headers); the agreement only forbids
  "unreasonable burden". The job runs at 1 req/s, the one concrete figure in
  circulation. Request *starts* are metered under a lock and a few workers let
  slow responses overlap, so latency never drags the rate below that.

| stage | requests | yields |
|---|---|---|
| MAL catalogue (`/anime/ranking?bypopularity`) | ~65 | all ~30,600 anime |
| AniList catalogue (50 per GraphQL call) | ~600 | tags, staff, drop rates, relations, recs |
| forum user discovery | ~100 | ~6,000 usernames |
| public user lists | ~6,500 | ~5,000 lists, ~4M ratings |
| MAL detail pages (≥5,000 members) | ~9,400 | rec graph, relations, status stats |
| backfill (`--part backfill`) | ~65 + 314 | the same fields for titles enriched on demand before the full fetch existed |

The sample was grown to ~20,000 lists on 2026-10-09/10
(`scripts/expand_population.sh`: forum posters, then their lists, in steps
that hold the shared MAL lock only while they run; one night at 1-1.5 req/s).
The backfill exists because both catalogue stages select what was never
fetched, which skipped the ~3,100 mostly popular titles already enriched by
earlier on-demand requests; their stored payloads predated the new fields.

**Keeping the sample current** (`malrec sync cf-refresh`, then
`malrec population fit --gate-user <MALREC_USER>`; `scripts/monthly_refresh.sh`
runs both on the 1st of each month at 03:30 via `scripts/malrec.cron`). The refit is
gated: the new model is stored inactive and activated only if it passes an
exam on the newest lists and the reference profile's holdout (Evaluation,
The population gate); otherwise the current model stays and users are left
as they are. Since usernames are discarded
after fetching, old lists cannot be re-read; the sample rotates instead: up
to 800 new users (default) are discovered from the *replies* in the newest
anime discussion topics — casual viewers, not only the forum regulars who
start topics — their lists fetched, and as many lists older than a year
retired. About 850 requests, a quarter of an hour at 1 req/s.

Sampled users are stored pseudonymously: the username is kept only until the
list is fetched, then only its hash remains. App users are excluded from the
sample, so their own ratings can never leak into the data used to evaluate
them - also when the account is created after its list was sampled: the
list then leaves the sample with the account's creation
(`store.leave_sample`). Until 2026-10-24 only lists fetched after sign-up
were kept out, and most accounts' own lists were in the sample their model,
holdouts and calibration were fitted on (audit 2026-10-23: holdout rank
correlation +.017 too good on average, up to +.061).

**Series and franchises** (`recsys/series.py`, rebuilt by
`malrec sync franchises` after every catalogue sync). A *series* is one
story: its seasons and films, other versions of it, and the side stories and
recaps that hang off it. A *franchise* adds its spin-offs and alternative
settings. Each pair of related titles gets one kind (continues the story /
same world / retold / hangs off); continuations and retellings merge
outright, and a title that hangs off others joins a group only upward, when
it hangs off exactly one larger group. A crossover short linked to two series
joins the larger one and bridges nothing. Until 2026-10-15 both were plain
connected components of the relation graph, and 45 franchises held unrelated
shows: a crossover special made Monogatari and Nisekoi one franchise, others
joined Detective Conan and Lupin III, Fairy Tail and Rave, Persona 4 and 5.
Only one title per franchise could appear in a list, and "Not for me" on one
removed the other. Now: ~2,900 series and ~2,800 franchises with more than one
title; Fate, Gintama, Attack on Titan and Sword Art Online stay whole.

### 2. Model

**Population layer** (`recsys/cf.py`, fitted by `malrec population fit`):

- *item bias* — how far above their own mean people rate a title (shrunk);
- *item-kNN* — cosine over co-ratings, centred per rater, top-60 neighbours;
- *matrix factorisation* — ALS, rank 32, on what the bias leaves;
- *co-occurrence* — which titles share **lists** regardless of score
  (including dropped and plan-to-watch);
- *EASE* — a closed-form item-item model over the same lists (Steck 2019),
  the second half of the relevance signal.

Fitted on every sampled list with 5+ ratings (model 16, 2026-10-24: 18,690
lists - the sample without the app users' own -, 7.8M ratings, 19,946
titles).

A new user is placed by **fold-in** from their own ratings — no retraining —
so this works from the first handful of ratings. A small *stacker*, fitted on
users the population model never saw, blends these signals with community
scores.

**Personal layer** (`recsys/hybrid.py`, mode `personal`): the ridge regression
this project started with — MAL categorical features, AniList weighted tags,
consensus scores — with recency-weighted samples (taste drifts) and implicit
pseudo-ratings from dropped / on-hold / finished-unrated entries.

**The blend** (mode `sized`, as deployed - `MODEL_MODE` in compose; the code's
own default is `personal`): `lam(n) * personal + (1 - lam(n)) *
population`, with `lam` logistic in the number of scored ratings (50/50 at
150, `blend_size_n0`) and never above 0.5 (`blend_lam_max`). The population
layer is the better predictor for almost everyone; the personal model is a
bounded correction for each person's quirks. Until 2026-10-03 the handover
sat at 80 with no cap, which gave long lists a near-pure personal model:
moving and capping it lifted rank correlation by .02-.03 from 150 entries
up, and the reference profile from .753 to .773
(`experiments/exp_handover.py`).

The ranking dials (relevance pull, Safe Bets popularity prior) follow list
size on the old curve (`ranking_share_n0` 80), not the blend weight, so the
better predictor did not also silently change how mainstream the lists are.

### 3. Rank

```
rank key = calibrated predicted score + relevance bonus + novelty bonus
           - risk + (Safe Bets) popularity prior + the tab's learned order
```

The predicted score says how much someone would like a title *if* they
watched it. On its own that is a trap: devotee classics are watched almost
only by devotees, so their conditional ratings glow, and every newcomer got
the same list of 1970s boxing and ballet dramas (38% of top-20s shared between
unrelated users). The **relevance bonus** — co-occurrence with the user's own
list — asks whether a title is in their world at all. It applies in proportion
to the population share: strongly for thin lists, and through a separate,
bounded dial (`relevance_weight_personal`) for long histories. Its code
default 0 keeps an established profile's lists from drifting toward the
mainstream; the compose file - every deployment, not only dev - sets 0.5.

The **novelty bonus** is the product's "balanced" setting. It is damped only
when a model's predictions are nearly flat, so it can break ties but never
decide a ranking alone — which is how one early user was shown Precure.

**Filters** (format, episode count, year, genres to hide) narrow each
surface's stored shortlist (`rec_candidate`, up to 300 scored titles), which
is re-ranked on read exactly as it was built, so nothing is refitted. Titles
that land on your list — rated from the app, or found by a sync — drop out on
the next read.

Then:
- **prerequisite filtering** — a recursive CTE walks prequel chains, so season 5
  never appears unless you completed the seasons before it. Only main formats
  (tv, movie, ona) are required: MAL lists side OVAs as prequels (One Punch
  Man's is "Road to Hero"), which hid 192 main series — One Punch Man, GTO,
  Assassination Classroom, Nichijou among them — until 2026-09-26. The walk
  stops at the title it starts from: contradicting MAL and AniList edges made
  61 titles their own prerequisite - Danganronpa and Ao Haru Ride among them,
  never recommended - until 2026-10-24
- **franchise collapse** — one entry per franchise (Data, Series and franchises)
- **genre balance** — calibrated re-ranking toward the user's genre mix
  (below, Rating round, ranges, genre balance)
- **age balance** (Safe Bets, Discover, Hidden Gems) — the same toward the
  user's age mix: how old titles were when they watched them (0-1, 2-4,
  5-9, 10-19, 20+ years). The list is then ordered by the key within blocks
  of 20, so the visible top 20 keeps the mix too (below)

**The learned order** (Safe Bets; `recsys/ltr.py`, `order_ranker_weight`
0.5). A LightGBM lambdarank model re-ranks the top 600 of the hand-built key
and is mixed with it half and half (as z-scores, so the shift stays on the
key's scale and rides in the stored bonus - filters and reranking on read
stay consistent). It is trained per population model (`malrec ranker train`,
~25 minutes; the monthly refresh does it after a new model) on ~2,300 of the
stacker's held-out lists, each cut at two of five history sizes (10-400 entries);
a title is a positive if the user later rated it at least their mean (7+),
a strong one at mean + 1.5 (9+). Its inputs are the key's own terms -
prediction and its population / personal parts, relevance, each risk part,
popularity prior, acclaim, genre fit, novelty, list size, the title's age
(years since release, not its release year: an absolute year put every
current title past the end of the training range, where the trees read
"newest" as "best") - plus the user's age taste (how much of their list
they watched at this age, and how they rated it) and long-series taste: the share of started titles that run 50+ episodes,
how they rate those against their mean, and how often they drop them. No
hard length rule: the ranker learns per user whether long series land.
Without a ranker for the active model, Safe Bets uses the hand key alone.

The ranker is trained with the identity calibration, and since 2026-10-24
served on that scale too: a user's own calibration line no longer moves its
inputs. Training it *as served* (`ranker_train_as_served`) - each query on
the pool production shows (`recsys/pool.py`: no sequels with unmet
prerequisites, no Up Next continuations, which are a fifth and an eighth of
the titles people went on to like) and the list at the cutoff, Discover's
against Safe Bets' learned list - was measured on the production-pool
benchmark (2026-10-24, `results_ranker_pool_2026-10-24.txt`): Safe Bets
and Hidden Gems within noise, Discover recall@50 -.005 / -.007 at 150 /
400 entries. No gain anywhere, so it is off, and the rankers keep learning
from the whole eligible pool.

What the learned orders win **on the pool production shows** (audit
2026-10-23, `results_audit_2026-10-23.txt`; the benchmark had counted those
easy sequels as hits until then): Safe Bets' top-10 hit rate +.056 / +.053
/ +.068 at 25 / 150 / 400 entries over the hand key, recall@50 about half
of what the old benchmark said (+.022 to +.031); Hidden Gems' gain holds;
Discover's is smaller (within noise below 400 entries), and the reference
profile's Discover recall is lower with its learned order (-.043, 6 of 8
splits).

**Discover and Hidden Gems** have learned orders of their own
(`discover_ranker_weight`, `gems_ranker_weight` 0.5; since 2026-10-11), from
the same pipeline and inputs, each trained on its tab's own pool: Discover's
without the titles Safe Bets' top 60 shows for that list (ordered with Safe
Bets' learned order, as served), Hidden Gems' its pool (below). Each is
stored per population model and tab
(`malrec ranker train --surface discover`, ~35 minutes each; without
`--surface` every enabled tab is trained) and falls back to the hand key
when missing. A ranker records the settings that shape the key it learned
from (`ltr.FINGERPRINT_SETTINGS`: the model, blend and risk dials); after a
change to any of them it no longer counts as current, and the tab falls back
to the hand key until `malrec ranker train` has run again. The fallback is
logged, and `malrec ranker check` (run by every deploy, which mails the
admin) names a tab that is switched on without a current ranker.

Every card explains itself: "because you rated X" comes from the kNN term
(people who rated X above their usual rated this above theirs too), the rest
from the model's own contributions, reported as what makes *this* title stand
out from the other candidates.

### Surfaces

| surface | what it is |
|---|---|
| `safe_bets` | the primary list: strong fits, novelty ignored. Ordered by predicted score plus the relevance bonus, a model that weighs every rating equally (`safe_bets_memory_mix`) and the learned order, so two neighbours can sit out of score order; "Why this?" shows every term and the resulting order score |
| `discover` | picks up where Safe Bets ends: never repeats it, balanced novelty, its own learned order |
| `hidden_gems` | strong fit among titles with a small audience for their age (the member count a title of its age is on course for below what MAL's #1,200 has today, below #2,000's while under a year old; `recsys/popularity.py`), its own learned order |
| `next_up` | the next full season or film of something you finished |
| `side_stories` | OVAs, specials and shorts from franchises you know |
| `this_season` | currently airing, ranked by fit |
| `plan_to_watch` | your own Plan to Watch, ordered by predicted fit alone |
| `coming_soon` | announced seasons and films of franchises you rated at least your own mean, rounded (min. 7), soonest first; no score shown, since an unaired title has nothing to predict from |

Formats are split into `main` (tv, movie, ona) and `side` (ova, special,
tv_special) by `format_class()` in the schema. MAL labels a four-minute joke
short and a full second season both as "sequel", so without that split
`next_up` fills with specials and buries the thing you actually want next.
Every surface except `side_stories` shows main formats only.

## Evaluation

Two populations, two questions:

1. **Held-out sampled users** (~430 MAL users the population model never saw),
   bucketed by how many list entries the model is given — the "new user"
   question. Their newest ratings are the test set; the model sees a random
   `n` of their older entries.
2. **The reference profile's own 8 temporal splits** (newest 25..60 ratings
   held out) — the "no losses" gate. The blend weight is the one production
   applies at the profile's real size.

Both measure **rating prediction** (rank correlation, nDCG@10, RMSE on the
held-out ratings) *and* **retrieval** (share of held-out titles the user went
on to love that appear in their top 50 / 100 from the whole catalogue).
Rating prediction alone cannot see the "everyone gets the same devotee
classics" failure; retrieval can.

```bash
make dev-lab CMD="python -u experiments/exp_final.py"   # shipped vs today
```

Full sample (4,563 users with ≥15 ratings; 2026-09-26; ~180 held-out users per bucket):

| list entries | rho today -> shipped | recall@50 today -> shipped | top-20 overlap between users | median popularity of top 20 |
|---|---|---|---|---|
| 10 | 0.278 -> **0.464** | 0.027 -> **0.125** | 0.050 -> 0.010 | #5,175 -> #807 |
| 25 | 0.360 -> **0.463** | 0.053 -> **0.145** | 0.045 -> 0.013 | #3,252 -> #694 |
| 60 | 0.407 -> **0.483** | 0.076 -> **0.195** | 0.038 -> 0.017 | #1,976 -> #580 |
| 150 | 0.462 -> **0.497** | 0.108 -> **0.174** | 0.047 -> 0.033 | #1,377 -> #698 |
| 400 | 0.532 -> **0.538** | 0.099 -> **0.128** | 0.047 -> 0.046 | #1,504 -> #1,000 |

Reference profile at its real size (personal share 0.985): rho 0.7625 ->
0.7686 over 8 splits (5 better, 2 identical, 1 lower by 0.0015), nDCG@10
0.761 -> 0.762, RMSE 0.916 -> 0.913. After refitting the production model on
the full sample its top-50 Discover, Hidden Gems and Safe Bets lists were
50/50 unchanged. Raw logs: `experiments/results_*_2026-09-26.txt`.

**Against simple baselines** (external audit, 2026-09-27; `exp_baselines.py`,
`exp_audit.py` on the deployed configuration). Rank correlation on newest
ratings:

| list entries | community score | user mean + item bias | population only | shipped |
|---|---|---|---|---|
| 10 | .438 | .447 | .454 | .454 |
| 25 | .439 | .451 | .463 | .463 |
| 60 | .443 | .450 | .475 | .479 |
| 150 | .493 | .493 | .514 | .512 |
| 400 | .517 | .514 | .539 | .537 (.562 with the own-history blend, used from 300 ratings) |
| reference profile | .750 | .766 | .738 | .779 |

The honest reading: for most people the model beats the community score by
0.02-0.04 rho - with 10-60 ratings there is little signal to find. The large
gains are for long, idiosyncratic histories. Retrieval (recall@50 of liked
future titles; IPS = weighted by 1/P(watched), which removes the head start
people's own popularity-driven choices give a popularity ranking): Safe Bets
beats a plain popularity ranking on IPS recall at every list size and on
plain recall from 25 entries; a fading popularity term for thin lists
(`safe_bets_popularity` 0.5) lifts plain recall at every size (10 entries:
.141 -> .186) without lowering IPS recall. Discover trades recall for novelty
on purpose.

Known optimism in the reference-profile numbers: its half-life was chosen on
the same 8 nested splits it is reported on. Chosen on the 4 older splits and
scored on the 4 newer, 0.35 still wins, by +0.009 rather than +0.015.
A rating without a finish date is dated by its last edit, so re-scoring an
old title makes it "new". Dating it by its start date instead was tested
(2026-09-27): on a fixed test set the last-edit dating trains the better model
(reference profile .796 vs .785, 5 of 8 splits) - the edit is when the opinion
was formed. Kept. `malrec eval-prospective` compares predictions as they were shown with
scores given afterwards - the one test the model cannot have seen.
`malrec eval-lists --cut DATE [--surface S] [--days N]` asks the same of a
ranking change at list level: of a tab's top 10 shown before and after the
date, how many were added, started or rated at least the user's mean
(positive) or dismissed, dropped or rated more than a point under it
(negative) within N days of first being shown - overall, as the mean
over users, and paired: after minus before for each user with titles on both
sides of the date, so a change of who was active is not read as a change of
the list. What happened is read from `list_event`, which records every
change of status or score with its time (since 2026-10-24); titles shown
before that are judged by the list as it is now, timed by its last edit,
which a later change overwrites. Item
consensus features (MAL/AniList means) are today's values, not as of each
cutoff; MAL publishes no history, and it flatters every consensus-based
method equally.

**Handover and cap** (`exp_handover.py`, 2026-10-03, ~50-95 held-out users
per size; Safe Bets keys as built in production):

| list entries | rho: today -> shipped | RMSE | Safe Bets recall@50 | IPS recall@50 |
|---|---|---|---|---|
| 25 | .546 -> .551 | 1.373 -> 1.374 | .211 -> .211 | .111 -> .111 |
| 60 | .524 -> .520 | 1.272 -> 1.276 | .234 -> .231 | .128 -> .124 |
| 150 | .554 -> .580 | 1.224 -> 1.201 | .217 -> .208 | .118 -> .113 |
| 250 | .550 -> .575 | 1.209 -> 1.181 | .214 -> .210 | .125 -> .125 |
| 400 | .577 -> .595 | 1.139 -> 1.123 | .212 -> .206 | .108 -> .112 |
| reference profile | .753 -> .773 | .919 -> .847 | .334 -> .339 | .248 -> .264 |

Leaving the dials coupled to the new blend weight gave the reference profile
more plain recall (.456) but a Safe Bets top 20 with median popularity #35,
in effect the most popular titles; that is a different product, so it is
not shipped.

**The top of the list** (`exp_toplist.py`, 2026-10-03). Rank correlation
and recall are measured on titles people chose to watch, so neither sees a
devotee title at #1 for someone who has never touched its genre - which is
what the later handover produced: Hajime no Ippo (2000, 75 episodes,
boxing) led the reference profile's Safe Bets, population layer 9.07,
personal model 7.91, no rated title of theirs linked to it. The population
layer rates such titles highly because mostly devotees watch them. The
order key now carries a risk term, scaled by the trust in the personal model
(so thin lists are untouched): 0.25 points per point the population layer is
more optimistic than the personal model, 0.3 for a genre absent from the
user's list, 0.3 for a 50+ episode series. Reference profile, two user
splits averaged:

| | top-10 hit rate | recall@50 | IPS recall@50 | long or unfamiliar in top 10 |
|---|---|---|---|---|
| before | .169 | .400 | .314 | ~.15 |
| with the risk term | .244 | .427 | .328 | ~0 |

Held-out users at 150-400 entries stay within noise on every metric; bad
picks (rated more than a point below their mean, or dropped) at 400 entries
fall .015 -> .010. Tried and not shipped: an evidence-weighted blend (shrink
the population weight where the user has no rated neighbours) cost rank
correlation (.781 -> .767); stronger disagreement penalties (1-2) traded
the reference profile's recall@50 for top-10 hits and cost 400-entry users
.03 of hit rate. "Why this?" shows the risk term as its own line. The
length rule applies at every list size and also counts a series that has
been airing for three years or more (MAL gives those 0 episodes; Detective
Conan and Crayon Shin-chan were thin lists' Hidden Gems): long titles in
thin lists' top 10 .13 -> .09, hit rate and recall unchanged.

**Evidence check** (`exp_toplist.py --round 7`). A high personal-model score
is not evidence on its own: the content ridge carries acclaim columns (MAL
and AniList score, popularity, favourites), and for the reference profile
they made up most of the lift on older classics - Cowboy Bebop's personal
8.22 was +0.79 acclaim, +0.04 content, with no rated title linked to it by
co-ratings. The risk term gains a fourth part: 0.6 points per point of the
personal model's lift over the user's average rated title that comes from
the acclaim columns, fading with co-rating evidence from the user's own
rated titles as e0 / (e0 + evidence), e0 = 0.1. Trust-scaled like the others.

| | top-10 hit rate, 400 entries | recall@50, 400 entries | reference profile top-10 hit rate |
|---|---|---|---|
| before (seed 3 / seed 4) | .386 / .379 | .257 / .299 | .375 / .400 |
| with the evidence check | .431 / .411 | .268 / .310 | .412 / .450 |

At 150 entries the gain is small (+.007 / +.011), thin lists are untouched,
and bad picks stay level (.018 -> .022 on one split, .011 on both on the
other). A plain acclaim discount without the evidence fade did little; letting
the content part offset the discount as well was weaker. "Why this?" now
names each part of the risk term on its own line. Round 8 tried stronger
penalties (1.0, 1.5) and faster or slower evidence fades (e0 0.05, 0.2): all
within noise of 0.6 / 0.1 on the second split, and 1.5 adds bad picks, so the
setting stays.

Round 9 tried counting only MAL's genres proper (Action, Sports, ...) as
"unfamiliar", not themes (Medical, Showbiz, Childcare): The Apothecary
Diaries, the reference profile's highest prediction, was held back by the
theme "Medical". Genres-only behaved like switching the rule off - nearly no
list lacks a whole genre - and cost the reference profile top-10 hits
(.412 -> .375 and .450 -> .375 on the two splits), with held-out users level.
Themes new to a list do predict misses, so the rule stays as it is.

Round 10 tried leaving Safe Bets' titles out of Hidden Gems, as Discover
does - for a long, niche list 8 of the gems' top 10 also stood in Safe Bets.
Distinct liked titles across both tabs' top 10 fell instead of rising
(400 entries .260 -> .248 and .248 -> .228; the niche profile level), and
the gems' own hit rate about halved: the shared titles are the best
lesser-known picks, and what moves up in their place rarely lands. Both
tabs keep them.

Safe Bets' long-memory adjustment is capped at +-0.15 points
(`safe_bets_memory_cap`): uncapped, it carried Mahouka (predicted 6.90,
the user's mean 7.51) to #8 on +0.65 alone. Two user splits: reference
profile top-10 hits .225 -> .263 and .275 -> .300, held-out users within
noise. A floor at the user's mean (`safe_bets_floor`) showed no benefit and
is off.

Side stories and recaps (MAL "parent story" / "full story" relations) are no
longer discovery picks: Lord of Mysteries' specials led Hidden Gems, and
Attack on Titan's compilation film was offered to someone who had not seen
the final season. Side stories now appear in Side Stories next to what
they add to, whatever MAL calls their format.

**The population gate** (`malrec population fit --gate-user`, monthly). The
newest batch of sampled lists (one rotation) is the exam: the candidate is
fitted without it, the active model predates it, and both predict those
lists' newest ratings. Every earlier batch the active model has not seen
trains the candidate, so a refit after a failed gate still gains data. The
candidate fails only if it loses more than .002 mean rank correlation *and*
more than one standard error of the paired difference (two equally good
refits differ by about .003). The gate user vetoes a mean loss of more than
.01 on their own 8 splits only while the exam gain is within two standard
errors: on 2026-10-10 a fourfold sample won the exam by .010 and cost the
reference profile .022, which is one taste pulled toward the mainstream, not
a fault, and kept 20,000 lists out. A loss above .05 refuses regardless.
Fewer than 100 lists in the batch:
the older single-user rule. Refits are deterministic per user and title:
who fits the population model or the stacker is a stable hash of the user,
and each title's factorisation start comes from its id (one extra user used
to move the factors by up to 2.2; now .006), so two fits differ by their
data, not by a reshuffle. The gate user's splits are scored at their own half-life
and real list size, i.e. the model they are served. `--dry-run` reports
without activating.

**The fourfold sample and the learned order** (2026-10-10, benchmark on the
expanded snapshot; details and the rejected counterweights in
`docs/RESEARCH.md`). The larger sample predicts unseen lists better (exam
+.010) and the reference profile worse (-.022). The learned order, mixed
50/50, lifts Safe Bets' top-10 hit rate at every size - 25 entries .18 ->
.22, 150 .28 -> .32, 400 .37 -> .46, reference profile .39 -> .50 - and
recall@50 by .02-.03 for sampled lists; part of it is popularity (IPS recall
+.004 to +.011, the reference profile's top-20 median popularity #154 ->
#87).

**Re-measured as served** (2026-10-11, `exp_blend.py`; RESEARCH.md has the
table). The learned order beats the hand key alone at every size (top-10
hit rate -.03 to -.10 without it). Kept as they are: the long-memory mix
(neutral for Safe Bets, a hint of a Discover loss for the reference profile
without it) and the risk term in the key (left to the ranker alone,
recall@50 -.008 at 400 entries). Not shipped: popularity-corrected
training labels, which moved lesser-known hits from Discover into Safe
Bets rather than finding new ones. Shipped: Discover's and Hidden Gems'
own learned orders - Discover recall@50 +.015 to +.024 at every size
(reference profile .038 -> .091), Hidden Gems +.008 / +.015 and top-10 hit
rate +.023 / +.044 at 150 / 400 entries, Safe Bets unchanged.

**The `relevance_weight_personal` dial** (reference profile, same splits; it
changes only the order, so rho is unaffected):

| setting | recall@20 | recall@50 | recall@100 | median popularity of top 20 |
|---|---|---|---|---|
| 0 (default) | 0.071 | 0.122 | 0.142 | #788 |
| 0.25 | 0.107 | 0.149 | 0.240 | #467 |
| 0.5 | 0.112 | 0.177 | 0.369 | #350 |
| 1.0 | 0.112 | 0.230 | 0.423 | #302 |

`experiments/exp_dial.py` shows what it does to a live list.

**Signals from the full fetch, tested and not shipped** (`exp_signals.py`):
staff (director / writer / creator) as tokens or as an affinity signal, drop
rate and score polarisation, tag similarity in the population stacker, and
dropped titles as negative relevance. In the stacker every one was within
noise for 10-60 entries (drop rate + polarisation slightly worse). In the
personal model the audience signals helped 150-entry users a little (+0.005
rho) but cost the reference profile 0.031 rho (7 of 8 splits worse); staff
tokens were +0.0015 there with one split -0.013. All remain behind flags,
off.

**Displayed scores for small accounts** now use a calibration fitted per list
size on held-out population users (`global_model.meta.size_calibration`):
at 5 ratings slope 0.69 (RMSE 1.64 -> 1.56), at 10 slope 0.77 (1.55 -> 1.50),
converging to identity by 150. It is used whenever an account has no usable
holdout of its own. Since 2026-10-24 each size's line is fitted on the blend
an account of that size is served; before, it was fitted with the old
handover at 80 and no cap (lines at 60 and 400 ratings moved, the lists did
not measurably).

**Top-of-list calibration** (`exp_topcal.py`). The worry was that list tops
overstate (the winner's curse). Measured on held-out users, the opposite: the
general line *understates* titles that make a user's top 60 by 0.16-0.44
points at every list size. A second line per size, fitted only on those
titles, removes the bias (top-of-list RMSE 1.71 -> 1.61 at 5 ratings,
1.68 -> 1.61 at 10) but is worse for arbitrary titles (1.60 -> 1.85), so it
is used only for the number on list cards and a list-opened "Why this?";
search and ranking keep the general line, so no list order changes.

Notes on method, each learned the hard way:

- **Spearman is tie-aware.** Scores are integers, so ties are the norm. An
  earlier version ranked by array position, which gave a constant prediction
  rho 0.12 and understated real models (the reference profile read 0.770
  instead of 0.790).
- **One consistent path.** Evaluation builds features in memory
  (`recsys/items.py`), verified to reproduce the production SQL features to
  4e-6, with the taste vector and every recency weight computed from the
  cutoff so nothing after it leaks in.
- **Temporal, not random, splits** for anything time-dependent (below).

Earlier, per-user-only results (before the population layer) follow.

### What each feature group is worth

Measured by zeroing one block at a time on the temporal holdout. Every group
has to justify itself with a number:

| configuration | ρ | nDCG@10 |
|---|---|---|
| everything | 0.728 | 0.749 |
| − MAL categorical | 0.700 | 0.717 |
| − consensus scores | **0.551** | 0.709 |
| − AniList tags | 0.739 | 0.756 |
| − affinity (graph + franchise) | 0.746 | 0.794 |
| + implicit signal (shipped) | **0.766** | **0.818** |
| baseline: MAL community score | 0.687 | 0.647 |

Two uncomfortable results, both acted on:

- **Consensus scores carry most of the signal.** Removing them costs 0.177 —
  more than everything else combined. Personalisation is real but it is a
  refinement on top of "is this broadly good", not a replacement for it.
- **The affinity block was net-negative**, so it is off by default
  (`use_affinity_features`). The recommendation graph averages 0.082 across a
  user's own rated anime but 0.0014 across candidates — a 50x shift, so the
  learned weight is calibrated to a distribution that never occurs at scoring
  time. The graph stays where it is genuinely good: retrieval and explanations.

### What was tried and rejected

These ideas were implemented, measured and thrown away. They live in
[`experiments/`](experiments/) so the reasoning is reproducible rather than
folklore.

| idea | result |
|---|---|
| LightGBM regression | ρ 0.454 vs ridge 0.728 |
| Pairwise ranking (RankNet-style) | ρ 0.730, lost all 8 splits |
| LambdaMART as the per-user rating model | ρ 0.547, nDCG 0.530 (180 rows per user; as a global ranker over thousands of lists it is now the Safe Bets order) |
| Synopsis embeddings, kNN feature | ±0.003, no consistent direction |
| Synopsis embeddings, SVD columns | −0.05 to −0.15 at every alpha |
| Two-tower retrieval with content features (`exp_twotower.py`, 2026-10-09) | recall@50 −.02 to −.03 and top-10 hits −.05 to −.10 vs production, alone or mixed in; helps only titles on no list |
| Like-minded lists (user kNN on 13k lists, `exp_uknn.py`) | ρ −.005 to −.037 for the reference profile, −.002 to −.023 sampled; lists unchanged. Left in the errors: .10 on 25-entry lists, .03 on long ones; for the reference profile its neighbours point the wrong way (ρ −.24 with the later ratings) |
| The Up Next sequel anchor in every prediction (`exp_seqall.py`) | sampled sequels RMSE 1.13 -> 1.04 at 400 entries, overall ρ +.003 at most; the reference profile −.012 to −.022 at every weight and carry-over (flat, the population's .68, its own): its sequel ratings sit near its mean whatever the first season got. Up Next keeps its own 50/50 rule |

The pattern in all of them is sample size. At ~180 training rows, ridge's
shrinkage is worth more than a better-matched loss or a richer representation.
Optimising the ranking objective directly is the textbook move and it loses
here; it is worth revisiting at a few thousand ratings, not before.

Embeddings lose for a different reason: AniList's ~22 human-curated weighted
tags per anime already describe premise directly, so a synopsis vector is a
noisier route to information the model already has.

### Calibration

Ridge shrinks toward the training mean, so the raw prediction is a good
*ranking* and a poor *score*: on this profile it predicted a spread of 0.50
where the real spread was 1.27, and sat +0.64 high. A linear map fitted on
out-of-sample holdout predictions fixes the displayed number without touching
the order:

| | RMSE |
|---|---|
| raw model output | 1.15 |
| MAL community score | 0.95 |
| **after calibration** | **0.89** |

### Rating round, ranges, genre balance

**Rating round** (signed-in, "Rate titles", only below 60 ratings, where a
banner also suggests it; since 2026-10-14 it is hidden from longer lists,
whose owners skipped its popular titles card after card): one card at a time, 1-10 / "haven't seen it" / skip, most-listed
titles first; ratings go to MAL like any other. Simulated on held-out users
(`exp_elicit.py`): 15 ratings on top of 10 known ones double recall@50
(.120 -> .239; IPS .079 -> .115) at ~33 cards. A "smart" pick (P(seen) x
spread of opinion) did worse on every measure (recall .178), so it is not used.

**Likely range and chance of 9+** on every prediction: from the user's own
past errors (their temporal holdout) or, for small accounts, those of
population users with a list the same size (top-of-list errors, as list
cards show the top). Shown is the shortest run of whole ratings around the
likeliest one that holds half of the error distribution - "most likely 7-8",
or one number: strictly out of time it held 56-66 % of later ratings for both
sources and the top of a list (`exp_range.py`, 2026-10-20). Until 2026-10 the
5-95 % band was shown ("likely 5-8"), which a user read as "you might rate it
a 5"; the 25-75 % band held only 45-55 % and was a single number half the
time. The bands stay internal (`malrec report`).

**Genre balance** is calibrated re-ranking (Steck 2018): each list is built
greedily, trading relevance against the KL divergence between the user's
genre mix and the list's (strength 0.7; genres hidden by a filter are left
out of the target). `exp_calib.py`: recall@50 up in Safe Bets and Discover at
every list size (reference profile Safe Bets .336 -> .368, Discover .078 ->
.100) with the genre mix closer to the user's; the old overlap penalty is the
fallback.

**Age balance** (2026-10-13, after a user who watches a lot of older anime
found each tab's top 20 dominated by the latest releases).
Three causes: the learned orders read the absolute release year, and the
current year lies past their training range; the future ratings they learn
from are recency-heavy by nature (people rate what just aired); and
Hidden Gems' pool past popularity #1,200 is mostly new titles. Fix: the
rankers see a title's age instead of its year, plus the user's age taste;
the calibrated re-ranking adds a KL term for the user's age mix
(`age_calibration` 1; the age is counted at watch time, finished or last
updated, so someone who watches each season as it airs keeps getting new
titles); and the calibrated list is ordered by the key within blocks of 20
(`calibration_block`), since sorting all 60 by the key put the newest at the
top again. `exp_blend.py` (`results_age_2026-10-13.txt`), against the
production of that day, at 25 / 150 / 400 entries:

| | Safe Bets recall@50 | hit@10 | age gap top 20 | Discover age gap | Hidden Gems age gap |
|---|---|---|---|---|---|
| before | .196 / .240 / .236 | .23 / .32 / .46 | .32 / .29 / .33 | .32 / .31 / .37 | .37 / .37 / .40 |
| shipped | +.023+ / +.009+ / +.005 | +.01 / +.03+ / +.02+ | -.15+ / -.14+ / -.17+ | -.13+ / -.16+ / -.18+ | -.16+ / -.18+ / -.19+ |

(age gap: total-variation distance of the top 20's age mix from the user's
whole history; lower is closer.) Discover's and Hidden Gems' recall did not
move beyond noise (Discover +.007+ at 150 entries). The age features alone
raise recall but make the lists newer still (age gap up to +.05); the
calibration alone, with the old year feature, costs a little recall
(Discover -.007- at 25 entries, Safe Bets -.004 at 400). Strength 2 gains a little more balance and gives part of
the recall back. Reference profile: Safe Bets recall .477 -> .487, hit@10
.50 -> .54+, Discover recall .091 -> .062 (8 splits, within noise at
strength 1).

**Sequels** in Up Next are predicted half by the model, half by "your score
for the previous season plus how raters of both typically move"
(`exp_sequel.py`, 3,000 held-out sequel ratings: RMSE 1.187 -> 1.052, rho .749
-> .800). The Promised Neverland's second season, for instance, is rated 3.8
points below the first by people who saw both. (Up Next had also read the
sequel relation backwards and listed prequels; fixed 2026-09-27.)

### Explanations

Cards explain themselves in up to three plain sentences (English/German),
built only from the reasons the model produced: what on your list the title
is tied to, which of your tastes it matches, and context (broad acclaim, a
little-known find, or an honest "no direct link to your list"). "Your
tastes" and the card's "from your list" bars are the attributes the user's
own ratings clearly rate above their average (the `evidence` reason, the
same lines as in "Why this?", attached at list build); the community bars
are the model's consensus terms, without how well known a title is. **Why this?** in the details panel
shows the whole derivation (`GET /explain/{mal_id}`): your average, what the
population layer and your personal model each predict and how they are
weighted, the calibration to the shown score, every driver pushing the
prediction up or down, the rated titles that vouch for it through real
co-ratings, and the relevance and novelty bonuses that placed it. Its
personal side ("your ratings say") lists what the user's own ratings clearly
say about the title's genres, themes, studio, format, era, length and source
(`taste.evidence_for`, the profile page's rule: series-level means, enough
series, a clear difference), each with its distance from the user's average
and the number of series behind it - not the personal model's largest
terms, of which only 15 % were backed by the user's ratings and 13 % said
the opposite (`exp_evidence.py`). When everything clear is against the
title, the panel says that it is recommended for what similar viewers say.
How well known a title is stays out of both sides. A tag on a card or in
the panel opens the catalogue's titles with it (`GET /tag/{tag}`), the ones
not on the list best predicted first.

**Profile page** ("Your profile" in the account menu; `GET /profile/taste`,
`taste.py`). One card per question, each led by a sentence chosen from a fixed
set (a key plus arguments, worded in the app): how the user's scale compares
with MAL's on the same titles, the genres and AniList themes they rate above
or below their own average, where they disagree most with MAL on titles at
least 20,000 people rated, their best and worst release years and how old
titles were when they watched them, studios, and how well the personal model
predicts their newest ratings. Only what the list supports is said: each
series counts once (the mean of its rated entries), a genre, studio or era
needs five different series and a difference of at least two standard
errors from the user's series average, a theme eight series and 2.5 (hundreds
of tags are tested). Below 15 rated series the taste cards ask for more
ratings; a card with nothing clear to say is left out. Labels that describe
a title rather than a taste, or that a page should not announce about
someone (adult genres, tags about sex, gender, orientation, suicide,
religion and the like), never appear. Tried on 3,400 sampled lists of every
size and the app's users (`exp_profile.py`): no list fails, every sentence
occurs; with 100-200 entries a typical list gets two clear genres, one or two
themes and no studio; from 800 entries on almost every card fills.

### Learning from what you do

Every build logs what was shown with its ranking inputs (`rec_log`).
`malrec learn-weights` joins the discovery tabs' impressions (not your own
Plan to Watch, continuations or announcements) with what you did afterwards,
in the app or on MAL itself once a sync has seen it: queued, put on Plan to
Watch, started, completed unscored or rated at or above your mean (positive);
"Not for me", dropped, or rated more than a point below your mean (negative).
It estimates what a
standard deviation of relevance or full novelty is actually worth in rating
points, with bootstrap intervals. Ignored titles are not counted as
negatives: most of a list is never scrolled to. It refuses to suggest anything
below 200 events with at least 30 of each outcome; the weights are changed by
hand, and re-validated with `exp_final`, only when an interval excludes the
current value.

## Why Postgres does the heavy lifting

- **recursive CTEs** resolve prerequisite chains (`prerequisites()`)
- **pgvector** stores a tag-profile vector per anime and your taste vector, so
  "more like this" is an index lookup (`<=>`)
- **GIN + trigram** indexes back search across titles and synopses
- **arrays and JSONB** hold genres, studios and recommendation reasons
- affinity aggregates run **once per query set**, not once per candidate

`eligible_candidates()` is the single definition of "could this be recommended
to this user", so the API, the CLI and the evaluation harness cannot disagree
about it.

## Configuration

Everything tunable lives in `src/malrec/config.py` and can be overridden by
environment variable. The values that matter:

| setting | default | note |
|---|---|---|
| `model_mode` | `personal` (`sized` in compose) | `personal` = the original per-user model only |
| `relevance_weight` | 0.5 | relevance pull for thin lists (population share) |
| `relevance_weight_personal` | 0.0 (0.5 in compose) | the dial for long histories; >0 trades list character for recall |
| `content_staff` / `content_audience` / `content_staff_aff` / `stack_extras` | off | tested, not shipped (see Evaluation) |
| `novelty_weight` | 1.0 | "balanced"; damped only for near-flat models (`novelty_ref_sd` 0.30) |
| `recency_half_life_years` | 0.5 | tuned; see `malrec tune-recency` |
| `implicit_weight` / `implicit_offsets` | 0.8 / dropped -2, on-hold -1 | see `malrec tune-implicit` |
| `mal_rps` | 1.0 | MAL publishes no limit; see Data |
| `anilist_rpm` | 30 | starting guess; the live header wins |
| `mal_redirect_uri` | `http://localhost:3000/api/auth/callback` | must match MAL's app config |
| `min_scoring_users` | 2000 | floor of the candidate pool where a surface sets none of its own; Safe Bets and Discover fix 2,000 in `surfaces.SPECS` |
| `blend_size_n0` / `blend_lam_max` | 150 / 0.5 | where the personal model takes over, and how far (`exp_handover.py`) |
| `ranking_share_n0` | 80 | list size the ranking dials follow; 0 = the blend weight |
| `disagreement_penalty` / `unfamiliar_genre_penalty` / `long_series_penalty` | 0.25 / 0.3 / 0.3 | risk term in the order key (`exp_toplist.py`) |
| `acclaim_penalty` / `acclaim_evidence_e0` | 0.6 / 0.1 | evidence check: lift from acclaim alone, fading with co-rating evidence (`exp_toplist.py` round 7) |
| `safe_bets_memory_cap` / `safe_bets_floor` | 0.15 / -99 (off) | Safe Bets guards |
| `safe_bets_memory_mix` | 0.0 (0.5 in compose) | Safe Bets' share of the every-rating-equal model |
| `order_ranker_weight` | 0.0 (0.5 in compose) | Safe Bets' learned order; needs `malrec ranker train` |
| `discover_ranker_weight` / `gems_ranker_weight` | 0.0 (0.5 in compose) | Discover's and Hidden Gems' own learned orders |
| `age_calibration` / `calibration_block` | 0 / 0 (1 / 20 in compose) | the three tabs held to the user's age mix, ordered within blocks of 20 (2026-10-13) |
| `age_fresh_weight` | 0.5 | the aim's weight on titles at most a year old: a seasonal watcher's tabs lean toward the back catalogue, This Season lists the new (`results_fresh_2026-10-22.txt`); acts only with `age_calibration` > 0 |
| `chosen_blend_min` / `blend_lam_min` / `blend_holdout` | 300 / 0.3 / newest | from this many ratings the personal share is chosen on the user's newest fifth, never below 0.3 (`results_blendchoice_2026-10-22.txt`; on model 16, without the app users' own lists, three long lists still choose 0-0.2) |
| `safe_bets_key_risk` | 1.0 | share of the risk term in Safe Bets' hand key (0 = left to the ranker; measured worse) |
| `order_ranker_ips` | 0.0 | popularity correction of the rankers' training labels (tested, not shipped) |
| `series_weight_power` | 0.0 | each rating counts 1 / n^power as taste evidence, n = the user's entries in its series; 0.5 and 1 cost Discover and Hidden Gems accuracy on long lists (`results_series_2026-10-15.txt`) |
| `not_interested_offset` / `_weight` / `_foldin` / `_franchise` | -1 / 0.4 / on / on | "Not for me" as an implicit rating under the user's mean, also in the fold-in; the franchise leaves the pool (`exp_notforme.py`) |

## Notes on the MAL API

- Reading a **public** list needs only the `X-MAL-CLIENT-ID` header. No OAuth,
  no token store, no refresh loop. OAuth is only required for `/anime/suggestions`,
  `@me` and writes, none of which this project uses.
- `limit` maxes at 1000 on the list endpoint, 500 on rankings, 100 on search.
- `/anime/{id}` returns at most 10 recommendations regardless of how many exist.
- The full OpenAPI spec is embedded in the Redoc page at
  `myanimelist.net/apiconfig/references/api/v2` as `__redoc_state`.

## Operations

### Local development

```bash
make db-up                    # just Postgres
make install && make init     # venv + migrations
make serve                    # API on :8000 (ENABLE_DOCS=true for /docs)
make worker                   # background tasks - or TASKS_INLINE=true for `serve` alone
make app-dev                  # Vite dev server on :5173, proxying to :8000
```

### Background work and memory

The API only enqueues rebuilds, list syncs and onboarding; the `worker`
container runs them, one task per user at a time (`WORKER_THREADS` users at
once), so a rebuild never slows anyone else's requests. A request that
arrives while the same task waits is absorbed by it; a task a dead worker
left behind is requeued at its next start. The API keeps at most
`SCORER_CACHE_SIZE` (16) users' loaded models in memory, ~20 MB each, and
reloads one as soon as that user's list, model run or the population model
changes. Measured on the reference profile: a warm rebuild of all eight lists
took ~5 s once a quadratic lookup in the risk term was fixed (10.7 s before).

`docker-compose.yml` defines everything; the `app` binds to `BIND_ADDR`
(loopback by default), the API and the database to loopback only - the API
is reached through the app's nginx, which decides the client address the
rate limits key on: the one a reverse proxy on a Docker network
(172.16.0.0/12) names, or the peer's own for a direct connection. It sends
the API that single address, replacing whatever X-Forwarded-For a client
sent (audit 2026-10-23: a VPN peer on the published port could pick a fresh
budget per request). IPv6 clients count by their /64. Services are
CPU/memory capped so they never compete with anything else on the host; the
API, worker and lab run without any Linux capabilities and none of the
services can gain privileges. The database image is pinned by digest. The
`lab` container mounts the working tree read-only except `experiments/`,
since the host runs scripts, compose files and `.env` from it as root. Put a
TLS-terminating reverse proxy in front of the `app` container for public use.

Tests: `make test-db` starts a throwaway Postgres (the stack's image, no
volume, port 5434) and migrates it; `make check` lints and runs the suite
against it, where a test that cannot reach the database fails instead of
skipping. `make dev-test` runs the suite on the server against a copy of
last night's backup, never the live database.

Migrations (`db/migrations`, applied by `malrec init` at every API start)
run once each: `schema_migrations` records each file with a hash of its
text, and a file runs again only when it changes - under an advisory lock
and a 15-second lock timeout, so a start that would wait on a busy table
fails (and the deploy rolls back) instead of stalling the database.

`make audit` (after `make audit-install`) runs the code-quality and security
checks - ruff's bug and security rules, bandit, vulture, pip-audit, npm
audit, semgrep and gitleaks (pinned containers, removed afterwards) - and
writes the findings to `audit/`, which is not tracked.

### Scheduled jobs (`scripts/malrec.cron`)

| when | job | cost |
|---|---|---|
| nightly 03:15 | `malrec sync users`: re-read every approved account's list (with their MAL token when signed in, so private lists work), rebuild those that changed or whose last rebuild did not finish - scores and status changes made on MAL itself reach the model within a day | 1-2 MAL requests per user, plus new titles' details |
| nightly 04:15 | `scripts/backup.sh`: `pg_dump -Fc` to a root-only backup directory, read back before older ones rotate out, 7 kept, skipped below 20 GB free | ~340 MB each |
| Mondays 03:45 | `malrec sync upcoming`: rankings + season lists, relations for new announcements (>=1,000 members), AniList for what is airing, announced or started/finished in the last 120 days (>=1,000 members or on a stored list), rebuild `coming_soon` and `this_season` | ~30-60 MAL requests, one AniList request per 50 titles due |
| 1st of month 03:30 | `scripts/monthly_refresh.sh`: sample rotation, AniList for the whole catalogue (`malrec sync anilist-refresh`, before the fit so it trains on current scores, tags and recommendations), gated refit (exam + `MALREC_USER`'s holdout), the learned orders (Safe Bets, Discover, Hidden Gems) for a new model, rebuilds, `learn-weights` report | ~850 MAL requests, ~600 AniList (~20 min) |

The MAL jobs share a lock, so they never run at once. AniList data is otherwise
fetched once per anime, on demand; these two passes are what keep it current.

The dumps are not encrypted and stay on the same host; only the MAL tokens
inside are sealed, with `TOKEN_KEY` from `.env`, which the dumps do not
contain - keep a copy of it elsewhere, or a restore on a new host signs
everyone out. Restore a dump with:

```bash
docker compose stop api worker
docker exec -i malrec-db pg_restore -U malrec -d malrec --clean --if-exists \
    < backups/malrec-YYYY-MM-DD.dump
docker compose start api worker
```

`scripts/deploy.sh` (run by `make dev-up` on a remote host set in `DEV`)
runs the migrations against a copy of the last backup first, takes a fresh
dump of the live database (`backups/predeploy-*`, the newest three kept),
keeps the running images as a rollback point, and restores them if the new
containers fail to start or are not all healthy within three minutes. The
cron jobs run in the `lab` container on the synced tree, so after a rollback
they run the new code until the old one is synced again. Experiments and
long jobs run in the `lab` service:

```bash
make dev-lab CMD="python -u -m experiments.exp_toplist --round 7"
```

Move the database with `pg_dump -Fc` / `pg_restore`, not by copying files.

### Admin e-mails

With `SMTP_HOST` and `ADMIN_EMAIL` set (STARTTLS on `SMTP_PORT`, login with
`SMTP_USER` / `SMTP_PASSWORD`), `malrec.notify` mails the admin, in the admin
account's app language and in the app's look:

| when | from |
|---|---|
| an account asks for approval (once per request) | sign-in |
| background tasks failed - the first at once, then bundled, at most every 30 minutes | the worker |
| the monthly refresh ran: new model or kept, with the gate's numbers; or it stopped on an error | `monthly_refresh.sh` |
| the weekly refresh (Coming Soon, This Season) ran, or failed | `weekly_upcoming.sh` |
| the nightly list sync or backup failed (no mail when they succeed) | `nightly_sync.sh`, `backup.sh` |

Every mail is recorded in the `notification` table, which also keeps one
event from being mailed twice. `malrec notify test --all` sends a test mail
and one sample of every kind. Sending never fails the job or request that
triggered it, and the test suite never sends.
