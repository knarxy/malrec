# Research notes: rating models, drift, and what "state of the art" means here

Written 2026-09-26 before changing how ratings are learned; brought up to date
2026-10-13. Each section: what the literature says, where malrec stands, and
what (if anything) is left to do. The dated results sections below are a log
and stay as written; **Open questions** at the end is the current agenda.
Sources at the end.

## 0. Frame: what kind of problem this is

malrec predicts **explicit 1-10 ratings** for a handful of users with long,
drifting histories, backed by ~18,700 sampled MAL lists with 15+ ratings
(~7.8M ratings; ~4,800 and ~2M until 2026-10-09). That is
the Netflix-Prize setting (explicit ratings, rating prediction + top-N), not
the setting most 2020s papers target (implicit clicks, billions of events,
next-item prediction). Two consequences:

- The methods proven for *explicit ratings with per-user bias and drift* are
  the relevant state of the art - chiefly Koren's baseline + factor models
  with temporal dynamics, and ordinal rating models.
- Large sequential / generative models (SASRec, HSTU) show their gains at
  scale on implicit next-action prediction. Meta's HSTU needs billions of
  users to pay off; at our scale the reproducibility literature finds tuned
  simple models (item-kNN, EASE, iALS, MF) match or beat them.
- What *does* pay at this scale is learning the **order of a list** from
  outcomes: a gradient-boosted ranker over the model's own signals, trained
  on thousands of held-out lists (shipped 2026-10-10, see below).

## 1. Temporal dynamics: model drift, don't just forget the past

**Literature.** Koren, *Collaborative Filtering with Temporal Dynamics*
(KDD 2009 / CACM 2010), the core of the Netflix-Prize-winning models:

- *Instance decay* (down-weighting old ratings) was tested extensively and
  **hurt**: accuracy improved as the decay was weakened and was best with no
  decay, "despite users changing taste and rating scale", because old ratings
  still carry persistent signal and cross-user patterns.
- Instead, time enters as **parameters**: a drifting user bias
  `b_u(t) = b_u + alpha_u * dev_u(t) + b_{u,t}` with
  `dev_u(t) = sign(t - t_u) * |t - t_u|^0.4` (days from the user's mean
  rating date; 0.4 by cross-validation), a spline alternative, a **per-day
  term** `b_{u,t}` for ratings given in one session, a per-user **rating-scale
  factor** `c_u(t)` multiplying item bias, time-binned item biases, and
  drifting user factors in timeSVD++.
- Netflix baseline RMSE: static .9799 -> linear drift .9731 -> + day effect
  .9605; adding `c_u(t)` .9555. timeSVD++ beat SVD++ at every rank; a rank-10
  timeSVD++ beat rank-200 SVD++.

**malrec today.** Recency *instance weighting* (half-life 0.5 years, floor
0.05) - the approach Koren found inferior - but measured here to help a lot
(reference profile rho 0.654 -> 0.765). Not a contradiction: our personal model is a
small per-user content ridge, not a global factor model, so old ratings carry
less shared signal. The level drift is *not* modelled; the display map
absorbs it after the fact, which is why a rating can move a prediction the
"wrong" way (see the Amagi Brilliant Park case).

**Status (2026-10-11).** Done and settled. The explicit level drift was
measured and rejected (Results 2026-09-27); the half-life stays at 0.5 years,
chosen per user from {0.35, 0.5} on their own splits. The **day effect** is
dropped from the agenda: it is a per-day shift of the user's level, our
metrics rank titles within one user, and the level-drift test - the same kind
of term over a longer horizon - moved them by at most .003.

## 2. Ratings are ordinal, and each person uses the scale differently

**Literature.** OrdRec (Koren & Sill, RecSys 2011) treats ratings as ordered
categories with **user-specific thresholds**, and outputs a full probability
distribution per item (mean, spread, P(>=9)). Ordinal models consistently
beat regression on explicit ratings and give principled uncertainty.

**malrec today.** Least-squares regression on 1-10 treated as interval data;
a linear map per user for display. Option 1 below is **shipped**
(`uncertainty.py`): every prediction carries a split-conformal range
("likely 7-9", ~80-85 % out-of-sample coverage) and a "chance of 9+", both
from the distribution of the user's own past errors (or, for small accounts,
those of population users with a list the same size).

1. ~~Conformal intervals~~ - shipped 2026-09-27.
2. An ordinal head (per-user thresholds) would give the same two numbers
   from a model rather than from past errors. Not worth it now: the
   conformal version already answers "gamble or safe bet", and Safe Bets'
   order is now learned from outcomes (section 6), which is the direct way to
   prefer titles that land. Revisit only if the shown ranges prove badly
   calibrated in `malrec report`.

## 3. Showing a predicted score changes the rating you give

**Literature.** Adomavicius, Bockstedt, Curley, Zhang (ISR 2013 and follow-ups):
a displayed rating - personalised prediction or community average - acts as an
**anchor**. Users rate items higher after seeing a higher number, even when
the number was deliberately perturbed. Showing both kinds does not add up;
the effect is about the size of either alone.

**malrec today.** Every card shows a predicted score; MAL shows the community
mean. Ratings given after seeing ours flow back into training and into the
evaluation.

**To do.** Nothing drastic. Worth knowing when reading "the model predicted
X and I gave X+1". The coarse band ("likely 7-9") is shown next to the score
since 2026-09-27. The prospective check (`malrec eval-prospective`, and the
gate's `prospective_compare`) scores predictions *as they were shown* against
ratings given afterwards; it is the one test the model cannot have seen, but
it is also the one the anchoring effect inflates - read its absolute level
with that in mind, its comparisons between models less so.

## 4. Selection bias: ratings exist only for what people chose to watch

**Literature.** Ratings are **missing not at random** (Marlin & Zemel;
Schnabel et al., *Recommendations as Treatments*, ICML 2016). Estimates of
error on observed ratings are biased; inverse-propensity weighting (IPS) and
doubly robust estimators correct training and evaluation. Active area
(stabilised DR 2022; SIGIR 2026 on positivity violations).

**malrec today.** Every metric (rho, RMSE, calibration, the top-of-list test)
is computed on ratings of titles users chose. Example: `exp_topcal.py` found
top-of-list predictions *under*-state actual ratings - partly because people
choose what they expect to like.

**Status (2026-10-11).** IPS-weighted recall (`ips50`: popularity-based
propensities per list size) is part of every benchmark run and of the
audits; no shipped change has lowered it. It matters most for the learned
order: trained on what people went on to watch and like, it learns that
popular titles get watched. Its lists are markedly more popular (the
reference profile's top-20 median popularity #154 -> #66), and its IPS gain
is a quarter to a half of its plain gain (Results 2026-10-10). **Training**
with propensities (labels weighted by 1/P, tempered 0.5 and 1) was tested on
2026-10-11 and not shipped: Safe Bets gained IPS recall exactly where
Discover lost it - the corrected ranker takes the lesser-known hits Discover
already showed, it does not find new ones (Results 2026-10-11).

**The same bias in time** (2026-10-13). What people rate next is mostly
what just aired, so labels and held-out ratings are recency-heavy as they
are popularity-heavy, and the benchmark rewards newness the way it rewards
popularity. A user who watches a lot of older anime reported it: each
tab's top 20 was dominated by the latest releases. Measured as the
gap between the top 20's age mix and the user's own, and corrected by
calibration rather than by reweighting labels (Results 2026-10-13).

## 5. Which population model: the reproducibility evidence

**Literature.** Ferrari Dacrema et al. (RecSys 2019, TOIS 2021): 11 of 12
reproducible neural methods lost to tuned kNN / linear baselines. Rendle et
al. (RecSys 2022): tuned iALS is competitive with or beats newer methods.
**EASE** (Steck, WWW 2019): a closed-form item-item linear model, often the
strongest top-N baseline. For sequential models, SASRec properly trained
(gSASRec, RecSys 2023) matches later models.

**malrec today.** Shrunk item-kNN + ALS + co-occurrence + EASE + a
stacker - the tuned-simple family that holds up best - fitted on ~18,700
lists since 2026-10-10. EASE shipped as a relevance signal on 2026-09-27.

Tried and rejected since (MODEL.md, "What was tried and rejected"):
- a **two-tower** retrieval model with content features and the logQ
  correction (`exp_twotower.py`, 2026-10-09): worse recall and top-10 hits
  than production at every list size, alone or mixed in; it only helps
  titles no list contains, which Safe Bets never shows;
- **user-kNN** ("like-minded lists", `exp_uknn.py`, 2026-10-10): adds
  nothing the item-based layer lacks; for the reference profile its
  neighbours point the wrong way.

Sequential and generative models: still not at this data size.

## 6. Lists, not just scores: calibration and diversity

**Literature.** Steck, *Calibrated Recommendations* (RecSys 2018): the genre
mix of a list should match the user's history (KL divergence, greedy
re-ranking); survey in ACM TORS 2025.

**malrec today.** Calibrated re-ranking (strength 0.7) replaced the
overlap penalty on 2026-09-27 (`exp_calib.py`: recall up in Safe Bets and
Discover at every size, genre KL down); the penalty remains the fallback when
no genre profile is known.

**Learning to rank.** Burges' LambdaMART optimises a list metric (nDCG)
directly with gradient-boosted trees; it is the standard learned ranker.
Rejected in 2026-09 *as the rating model* (180 rows per user - too few),
it is now **shipped as the Safe Bets order** (2026-10-10, `recsys/ltr.py`):
one global LightGBM lambdarank model, trained on ~2,300 held-out stacker
lists at five history sizes, re-ranks the top 600 of the hand-tuned key and
is mixed with it 50/50. Its inputs are the model's own terms (prediction,
relevance, risk parts, popularity, acclaim, genre fit, novelty, list size)
plus the user's long-series taste. Since 2026-10-11 **Discover and Hidden
Gems** have their own rankers, trained on each tab's pool (Discover without
the titles Safe Bets shows, Hidden Gems past popularity #1,200). Unbiased-LTR
work (Joachims et al. 2017) corrects such a ranker for the position and
popularity bias in its labels; the popularity correction was tried and not
kept (section 4).

Calibration is not only for genres. Since 2026-10-13 the three discovery
tabs are calibrated to the user's **age mix** too - how old titles were
when they watched them - and ordered by the key within blocks of 20, since a
calibrated set sorted as a whole by the key put its newest titles on top
again. The rankers see a title's age, not its release year (Results
2026-10-13).

## 7. LLMs

**Literature (2024-26).** On rating prediction, classic CF beats LLMs (lower
RMSE/MAE). On ranked lists, strong LLMs are competitive; hybrids feeding CF
signals into an LLM report sizeable gains on sparse or cold-start items.

**For malrec.** Not for scoring. Explanations are written from the
model's own reasons by templates (English/German) and need no LLM; Coming
Soon shows no score. Nothing here is on the agenda.

## 8. Evaluation methodology

**Literature.** Time-aware evaluation should split by time; leave-one-out
and random splits leak the future (Campos et al. 2014; RecSys 2025
"Time to Split").

**malrec today.** Per-user temporal holdouts with cutoff-consistent
features, plus strictly out-of-time nested tests (exp_displaycal). Since
2026-10: a frozen **benchmark** (`experiments/bench.py`, paired per user,
+-2 SE) that every ranking change must pass, a population **exam** on the
newest batch of lists neither model has seen, and the **prospective** check
on ratings given after recommendations.

Since 2026-10-11 the benchmark builds every tab as served: the
long-memory model when `safe_bets_memory_mix` is on, each tab's learned
order, the key's settings, and it records the settings it ran with. It
also reports IPS recall for all three discovery tabs. A real-use check
next to the prospective one: `malrec eval-lists --cut DATE` compares the
titles shown before and after a change by what users did with them
(added, rated well / dropped, rated badly) within a fixed window. One known
gap remains: the reference profile is scored on nested splits of one list
(8 splits, 26 sequel ratings were 6-8 titles), so its +-2 SE flags
overstate certainty. Since 2026-10-13 it reports the **age gap** of each
tab's top 20 (total-variation distance from the user's age mix): the
benchmark's labels favour new titles (section 4), so recall alone would
never flag a list that is all this year's.

## Results so far (2026-09-27)

**Level drift (section 1) - rejected.** `exp_signals.py --part level`: a
learned rating-level trend (linear capped at 3 years, log, and Koren's
`|t - t_u|^0.4`) changed held-out users by at most +-0.003 rho at 60/150/400
ratings, and cost reference profile 0.003-0.006 rho (Koren form: 2 of 8 splits
better, 6 worse). Strictly out of time, the display map still leaves the same
bias with or without it.

**Less decay (Koren's "no instance weighting") - rejected for this model.**
`--part grid`, half-life 0.5 / 1 / 2 years / none, with and without drift:
reference profile falls from 0.7666 to 0.73 / 0.69 / 0.73 rho (7 of 8 splits worse
each); held-out users are mixed (150 ratings: +0.007 at 2 years; 400: -0.007).
Koren's result belongs to a global factor model where old ratings carry
cross-user signal; our personal content model needs the recency weighting.
The half-life of 0.5 years stays.

**One lead:** with no decay, the reference profile's recall@50 of titles he later loved
rises from 0.200 to 0.348 while rating order gets worse - long memory finds
more, short memory orders better. Tested next: blending a no-decay model into
the discovery ranking only (`--part mix`), displayed scores unchanged.

Raw logs: `experiments/results_{level,grid}_2026-09-27.txt`.

**Shipped 2026-09-27 (gate passed; logs `results_{mix,combo,final2,prod}_2026-09-27.txt`):**

1. **EASE as a second relevance signal** (50/50 with co-occurrence, top-100
   weights per item). Recall@50 of liked future titles, today -> EASE:
   10 entries .125 -> .134, 25: .151 -> .175, 60: .214 -> .232,
   150: .216 -> .239, 400: .181 -> .204, reference profile .187 -> .245. Median
   popularity of the top 20 unchanged (reference profile #328 -> #339). Ranking only;
   displayed scores unchanged.
2. **Per-user recency half-life** chosen from {0.35, 0.5} on the user's own 8
   splits (>= 100 ratings). 0.30 / 0.35 / 0.40 all beat 0.5 for reference profile
   (smooth curve; 0.35: rho .765 -> .780, 7 of 8 splits); held-out users lose
   .001-.003 on average, hence per user, not global. reference profile -> 0.35.

3. **Long memory in Safe Bets only** (`safe_bets_memory_mix` = 0.5 on dev,
   `exp_tabs.py`). Safe Bets' order blends in a model with every rating
   weighted equally; Discover unchanged but skips Safe Bets' titles as before.
   Share of liked future titles caught by either tab: 25 entries .242 -> .260,
   150: .327 -> .341, 400: .280 -> .284, reference profile .529 -> .547. Safe Bets
   becomes more familiar (his top-20 median popularity #162 -> #81), as the
   tab promises. Displayed scores unchanged.

**Earlier variant, superseded:** blending a no-decay model into the discovery
order. With EASE it nearly doubles the reference profile's recall@50 (.187 -> .354) but
makes his top 20 more mainstream (median #328 -> #201); for 400-entry users it
slightly lowers recall without EASE. Same trade-off as the relevance dial.

**Audit follow-up (2026-09-27, `results_audit_2026-09-27.txt`):** own-history
blend weight from 300 ratings (+0.025 rho at 400; costs 0.023 on the
143-rating reference profile, hence the threshold); Safe Bets popularity
prior 0.5 (plain recall up at every size, IPS level or better); the audit's
"popularity beats our lists" measured Discover without EASE - Safe Bets beats
popularity on IPS recall everywhere. Tried and reverted: fitting the size
calibration on users disjoint from the stacker's (the smaller stacker cost
the reference profile 0.002 rho on 4 of 8 splits), and dating unfinished
ratings by start date (a worse training signal: .785 vs .796).

## Handover and cap (2026-10-03)

The size-weighted blend handed over to the personal model at 80 ratings
(~1.0 by 140). Three runs agreed that this was too early: the 2026-10-01
audit ("LATE", handover at 150), a curve on the reference profile (n0 80 /
100 / 120 / 150 / 200 at his real size: .753 / .763 / .785 / .792 / .788)
and `exp_handover.py` (table in README, Evaluation). With a ceiling of 0.5
on the personal share, 250- and 400-entry users gain a further .02. Once
capped, choosing the weight on the user's own newest ratings (from 300) adds
nothing over the size curve (.595 vs .594); it is kept as it costs nothing.

The ranking dials were keyed to the blend weight, so the change also turned
up the relevance pull and popularity prior for long lists. Decoupling them
(list size on the old curve) keeps recall and IPS recall at today's level.
Coupled is a trade worth knowing about: reference profile recall@50 .456
and IPS .325, but a top 20 of the most popular titles (median #35).

Also fixed: the gate and the stored holdout metrics scored the blend weight
implied by the truncated training window, not the one served (reference
profile: reported .782, served .753), and the gate ran at the default
half-life instead of the user's own. The gate now decides on an exam of the
newest sampled lists, which neither model has seen.

First dry run (2026-10-03): the candidate scored the reference profile
better (.792 -> .803) and the 247 unseen lists worse by .0029 (standard
error .0031, 108 better / 132 worse) and was refused. It held 5 more lists
than the active model - the same data refitted, so the difference was refit
noise. Three fixes followed: the exam is the newest batch only (before, a
failed gate would have kept every new list out of every later candidate);
roles and random starts are stable per user and title; and the tolerance
allows one standard error.

## The top of the list (2026-10-03, overnight)

The owner flagged Hajime no Ippo as Safe Bets #1 after the handover change.
An audit of every tab found the same pattern throughout: the population
layer 0.7-1.3 points above the personal model on titles no rated title
vouches for (Ashita no Joe, 1970 boxing, in Hidden Gems), plus two
structural leaks (side stories and recaps in discovery tabs).

`exp_toplist.py` adds metrics for the top of the list: top-10 hit rate, bad
picks, and the share of the top 10 that has no evidence from the user's own
ratings, an unfamiliar genre, 50+ episodes, or predates 2005. Four rounds:

1. Disagreement penalty (1/2), evidence-weighted blend, unfamiliar-genre
   penalty. Thin lists unaffected (trust scaling). The evidence-weighted
   blend cost rank correlation; strong disagreement penalties cost 400-entry
   users hit rate; the genre penalty lifted the reference profile's recall
   (.427 -> .518) at no cost elsewhere - the owner's red flag agrees with
   the data.
2. Weights and combinations, 150-400 entries: genre + long-series penalty
   ties or improves every metric everywhere.
3. Confirmation on another user split (seed 4): same ordering. Shipped:
   disagreement 0.25, genre 0.3, long 0.3.
4. Thin lists (10-60 entries): the length rule applied at every size and
   counting series airing for 3+ years (MAL gives them 0 episodes) cuts
   long titles in the top 10 (.13 -> .09 at 10 entries, .15 -> .10 at 60)
   with hit rate and recall unchanged. Shipped that way. No age rule: one
   thin account's list is mostly 1986-2008 classics, and the old titles it
   is offered fit it; "old" is the owner's red flag, not a universal one.
5. Safe Bets guards, after Mahouka (predicted 6.90, the owner's mean 7.51)
   reached #8 on a +0.65 long-memory adjustment alone: a floor at the
   user's mean showed no benefit and cost 250-entry users recall
   (.251 -> .243), not shipped; capping the long-memory adjustment at
   +-0.15 lifted the reference profile (top-10 hits .225 -> .263, recall@50
   .432 -> .460), held-out users within noise.

## Not for me, two-tower (2026-10-09)

**"Not for me"** (`exp_notforme.py`, simulated dismissals on the benchmark):
beyond hiding the title, a dismissal now counts as an implicit rating one
point under the user's mean at weight 0.4, in the personal model and the
population fold-in, and the rest of its franchise leaves the pool. Recall of
the rebuilt list up .005-.008 at 150-400 entries (top-10 hits +.024 at 150)
for .003-.009 of rank correlation; heavier weights cost more (-.012 to
-.029). **Two-tower retrieval**: rejected (section 5).

## The fourfold sample and the learned order (2026-10-10)

**Sample.** 5,129 -> 20,374 lists (`scripts/expand_population.sh`, 1-1.5 req/s,
one night: 8 hours). The refit won the exam on 3,000 unseen lists by +.010 rank
correlation (clearly beyond noise) and cost the reference profile .022 on
its 8 splits; the gate vetoed it. Diagnosis (`exp_source`, `exp_models`,
`exp_params`, `exp_refitems`): not the stacker, the factor settings, a leak
or the current season - the many new lists rate mainstream titles the
reference profile rated low (isekai sequels, shounen) higher, and pull its
predictions with them. The owner chose the larger sample; the gate now lets a
clearly better exam outweigh the gate user (MODEL.md, The population gate).
Model 15 (18,699 lists, 7.8M ratings) is active.

**Where the lists miss** (`bench_misses.py`): of liked future titles outside
the Safe Bets top 50, thin lists lose most to relevance and popularity, long
lists to the prediction itself; ~25 % are filtered by design (side stories,
formats, NSFW, < 2,000 scorers); the scored misses sit far down (median place
~400), out of reach of small re-weightings.

**Learned order** (section 6; `exp_ltr.py`, `exp_blend.py`), top-10 hit rate
/ recall@50, today -> mixed 50/50:

| list entries | hit10 | r50 | IPS r50 | pop# top 20 |
|---|---|---|---|---|
| 25 | .179 -> .222 | .176 -> .197 | +.011 | 159 -> 112 |
| 150 | .277 -> .317 | .211 -> .241 | +.011 | 222 -> 183 |
| 400 | .369 -> .455 | .206 -> .234 | +.007 | 374 -> 282 |
| reference profile | .388 -> .500 | .477 -> .477 | +.004 | 154 -> 87 |

The pure learned order (weight 1) gains more for sampled users but costs the
reference profile recall (-.029) and makes its top 20 the most popular
titles (#59); the mix keeps both. The long-series taste inputs (share of
started titles that are long, rating lift on them, drop rate) add a little
for sampled lists, none for the reference profile, and make its top 20 less
mainstream (#59 -> #87).

**Counterweights for the reference profile, all rejected:** more personal
weight (`exp_refcap`: at its real 152 ratings any share above 0.5 is worse,
.798 -> .746 at 1.0); like-minded lists (section 5); the Up Next sequel rule
in every prediction (`exp_seqall.py`: sampled sequels RMSE 1.13 -> 1.04, but
the reference profile -.012 to -.022 overall at every weight and
carry-over - its sequel ratings sit near its mean whatever the first season
got). The common thread: this taste departs from the crowd in ways other
lists cannot reveal; its own ratings and "Not for me" are what move it.

**More lists** were not pursued: the 5k -> 20k step gained .010 on unseen
lists, a further doubling would likely gain less, and the shared 15 GB host
already needs 27 minutes and most of the lab's 5 GB for one fit.

## The agenda worked through (2026-10-11)

One benchmark run, eight arms, each paired with the first - production as
served: long-memory mix 0.5, learned Safe Bets order 0.5
(`exp_blend.py`, `results_points_2026-10-11.txt`). Changes against it,
+/- = beyond two standard errors:

| arm | 25 entries | 150 | 400 | reference profile |
|---|---|---|---|---|
| long-memory mix off | 0 | 0 | 0 | Safe Bets 0, Discover r50 -.011 |
| risk out of the key | 0 | 0 | r50 -.008-, IPS -.008- | r50 +.011 |
| both | 0 | 0 | hit10 -.017-, Discover +.005+ | hit10 -.038- |
| hand key only (no ranker) | r50 -.019-, hit10 -.047- | r50 -.030-, hit10 -.032- | r50 -.028-, hit10 -.066- | hit10 -.100- |
| IPS labels 0.5 | hit10 -.013- | Safe Bets IPS +.011+, Discover IPS -.005- | Safe Bets IPS +.007+, Discover IPS -.008- | r50 +.011, Discover IPS -.010 |
| IPS labels 1 | 0 | Safe Bets IPS +.006+, Discover IPS -.006- | Safe Bets IPS +.008+, Discover IPS -.011-, Discover hit10 +.017+ | Discover r50 -.017 |
| Discover + Hidden Gems rankers | Discover r50 +.015+ | Discover r50 +.015+, hit10 +.034+; Gems r50 +.008+, hit10 +.023+ | Discover r50 +.024+, hit10 +.035+; Gems r50 +.015+, hit10 +.044+ | Discover r50 .038 -> .091+ |

**A1, long memory: kept.** With the learned order on, the
every-rating-equal model no longer moves Safe Bets for anyone (all
differences within noise). But Safe Bets' order decides what Discover may
show, and without it the reference profile's Discover recall@50 fell both
times it was measured (-.011 without, -.013 with the new Discover ranker;
same splits, so one finding, within noise). Dropping it would save one user
model per rebuild and gain nothing measured, so it stays at 0.5 until the
real-use check says otherwise.

**A2, risk in the key: kept.** Leaving the four risk parts to the ranker
alone costs long lists recall (-.008 at 400 entries); it gains the
reference profile .011 that is within its noise. The key's risk term and
the ranker's risk features are not redundant: the key sets which 600
titles the ranker sees.

**B1, popularity-corrected labels: not shipped.** At both strengths Safe
Bets gained IPS recall exactly where Discover lost it - the corrected ranker
pulls the lesser-known hits that Discover already showed into Safe Bets
instead of finding new ones - and plain recall did not move. For a tab
meant to be backed by the user's own ratings that is the wrong trade.

**B2, learned orders for Discover and Hidden Gems: shipped** (0.5 each). The
largest gain of the day: Discover's recall@50 up by a fifth to a third at
every size and more than doubled for the reference profile, Hidden Gems' up
at 150 and 400 entries, Safe Bets untouched. The gain is not bought with
popularity: IPS recall rises with plain recall (Discover +.008 / +.010,
Hidden Gems +.007 / +.015 at 150 / 400 entries).

**A3, real use.** `malrec eval-lists` (MODEL.md, Evaluation) compares a
tab's top 10 before and after a date. Baseline for the Safe Bets order
(cut 2026-10-10, 7-day windows): before, 112 titles, 5.4 % acted on
positively, 4.5 % negatively (user means 3.5 % / 3.3 %); the first 7-day
windows after the cut close on 2026-10-17, the 28-day ones on 2026-11-07.
Discover and Hidden Gems: cut 2026-10-11, same dates a day later. Small
numbers - seven active users - so read the user means and expect wide
intervals.

## Recency bias (2026-10-13)

A user who watches a lot of older anime found Safe Bets', Discover's and
Hidden Gems' top 20 dominated by the latest releases. The hand key alone
gave far fewer of them: most of it came from the learned orders, which read the absolute release year (the current
one lies past their training range, where trees continue the last trend)
and learn from future ratings, which are mostly of new titles. The rest:
the calibrated selection of 60 had roughly the user's age mix but was then
sorted by the key as a whole, and Hidden
Gems' pool is mostly new titles.

One benchmark run, six arms (`exp_blend.py`,
`results_age_2026-10-13.txt`), paired with production as served; the
rankers retrained per feature set. Age gap = distance of the top 20's age
mix from the user's whole history, lower is better.

| arm | Safe Bets r50 (25/150/400) | Safe Bets age gap | Discover / Gems age gap | reference profile |
|---|---|---|---|---|
| calibration 1 + blocks of 20 | +.003 / -.001 / -.004 | -.15+ / -.15+ / -.18+ | -.13 to -.21+ | r50 -.005, Discover r50 -.019 |
| blocks of 20 alone | 0 | -.01 / 0 / -.02+ | -.01 to -.03+ | 0 |
| age features (age, user's age share and lift) | +.025+ / +.010+ / +.009+ | 0 / +.03- / +.02- | 0 to +.05- | r50 +.010, Discover r50 -.026 |
| age features + calibration 1 + blocks | +.023+ / +.009+ / +.005 | -.15+ / -.14+ / -.17+ | -.13+ to -.19+ | r50 +.010, hit10 +.038+, Discover r50 -.028 |
| age features + calibration 2 + blocks | +.023+ / +.007 / +.003 | -.17+ / -.18+ / -.22+ | -.15+ to -.25+ | r50 +.016+, Discover r50 -.028- |

**Shipped: age features, calibration 1, blocks of 20.** The age features are
the better ranker input on their own (recall up at every size) but, read
honestly, they learn recency more cleanly than the year did; with the
calibration on top the lists halve their age gap and keep the recall gain.
Discover's recall did not fall for sampled users (+.007+ at 150 entries);
Hidden Gems' did not move. Strength 2 balances further and gives part
of the recall back - and a recency-heavy benchmark undercounts what balance
is worth, so 1 is the cautious choice, not the measured optimum. The
reference profile's Discover recall fell .091 -> .062 in every arm with the
age features (8 splits, flagged only at strength 2); it came from .038 two
days earlier, and the real-use check will say more than its splits can.
The age is counted at watch time (finished, else last updated), so someone
who watches each season as it airs keeps getting new titles; the
alternative, the release-era mix of the list, would push everyone toward
older titles and is not implemented.

## Open questions (2026-10-13)

**A3. Read the real-use check** (`malrec eval-lists --cut 2026-10-10`, then
`--surface discover|hidden_gems --cut 2026-10-11`) on 2026-10-17 and
2026-11-07. Note that the monthly refresh on 2026-11-01 may activate a new
population model (and retrain the rankers) inside the 28-day window. The
age balance changed all three tabs three days into those windows, so
their "after" side now reads the learned orders and the age balance
together; A4 separates the two.

**A4. Real use of the age balance.** `malrec eval-lists --cut 2026-10-14`
(the rebuilt lists went live in the night of 2026-10-13/14), for all three
tabs; first 7-day windows close 2026-10-21, 28-day ones 2026-11-11.

**B3. The Discover and Hidden Gems ranker weights.** Only 0.5 was measured
(as for Safe Bets, where 1.0 cost the reference profile); 0.75 / 1 may gain
more in tabs that are about breadth. One `exp_blend.py` run with cached
rankers, ~30 minutes.

**C1. Remove flags that were tested and stayed off for weeks:**
`use_affinity_features`, `content_staff`, `content_audience`,
`content_staff_aff`, `stack_extras`, `safe_bets_floor`, and now
`order_ranker_ips`. Their results stay in MODEL.md and
`experiments/`; the code paths go.

**D. Off the agenda** (measured, or not worth it at this scale): level drift
and day effect, less recency decay, an ordinal head, sequential / generative
models, two-tower retrieval, user-kNN, the sequel rule everywhere, more
personal weight for long lists, a nightly fetch beyond ~20k lists, LLMs for
scoring, popularity-corrected ranker labels, the risk term only as ranker
features, manga lists (decided 2026-10-13: the project stays with anime).

## Sources

- Koren, Collaborative Filtering with Temporal Dynamics (KDD 2009):
  https://faculty.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf ;
  CACM version: https://cacm.acm.org/research/collaborative-filtering-with-temporal-dynamics/
- Koren & Sill, OrdRec (RecSys 2011): https://dl.acm.org/doi/10.1145/2043932.2043956 ;
  Koren & Sill, Collaborative Filtering on Ordinal User Feedback (IJCAI 2013):
  https://www.ijcai.org/Proceedings/13/Papers/449.pdf
- Adomavicius et al., Do Recommender Systems Manipulate Consumer Preferences?
  (ISR 2013): https://pubsonline.informs.org/doi/10.1287/isre.2013.0497 ;
  Recommender systems, ground truth, and preference pollution (AI Magazine 2022):
  https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12055
- Schnabel et al., Recommendations as Treatments (ICML 2016):
  https://arxiv.org/abs/1602.05352 ; StableDR: https://arxiv.org/pdf/2205.04701
- Ferrari Dacrema et al., A Troubling Analysis of Reproducibility and Progress:
  https://arxiv.org/abs/1911.07698
- Rendle et al., Revisiting the Performance of iALS (RecSys 2022):
  https://arxiv.org/abs/2110.14037v1
- Steck, Embarrassingly Shallow Autoencoders (WWW 2019): https://arxiv.org/abs/1905.03375
- Petrov & Macdonald, gSASRec (RecSys 2023): https://arxiv.org/pdf/2308.07192 ;
  A Reproducible Analysis of Sequential Recommender Systems: https://arxiv.org/html/2408.03873v1
- Zhai et al., Actions Speak Louder than Words (HSTU, 2024): https://arxiv.org/abs/2402.17152
- Steck, Calibrated Recommendations (RecSys 2018); survey (ACM TORS):
  https://dl.acm.org/doi/10.1145/3789266
- LLMs vs CF on ratings vs ranked lists: https://journals.sagepub.com/doi/10.3233/FAIA260019
- Conformal prediction for rating intervals: https://arxiv.org/pdf/2412.12110
- Burges, From RankNet to LambdaRank to LambdaMART: An Overview (MSR-TR-2010-82):
  https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
- Joachims, Swaminathan, Schnabel, Unbiased Learning-to-Rank with Biased
  Feedback (WSDM 2017): https://arxiv.org/abs/1608.04468
- Yi et al., Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
  Recommendations (RecSys 2019, the logQ correction):
  https://dl.acm.org/doi/10.1145/3298689.3346996
- Time-aware evaluation: Campos et al. survey
  https://www.researchgate.net/publication/257671605 ; Time to Split (RecSys 2025):
  https://arxiv.org/pdf/2507.16289
