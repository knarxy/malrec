# Research notes: rating models, drift, and what "state of the art" means here

Written 2026-09-26 before changing how ratings are learned. Each section: what
the literature says, where malrec stands, and what (if anything) to do. Sources
at the end.

## 0. Frame: what kind of problem this is

malrec predicts **explicit 1-10 ratings** for a handful of users with long,
drifting histories, backed by ~4,800 sampled MAL lists (~2M ratings). That is
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

**To do.** Model the level explicitly (running: `exp_signals.py --part level`,
variants lin3 / log / koren). If it passes, re-tune the recency half-life with
the drift term in place - with the level handled explicitly, less decay may be
optimal, as Koren found. Also worth a test: a **day effect** - MAL imports
often put dozens of ratings on one day, exactly the "session" pattern
`b_{u,t}` absorbs.

## 2. Ratings are ordinal, and each person uses the scale differently

**Literature.** OrdRec (Koren & Sill, RecSys 2011) treats ratings as ordered
categories with **user-specific thresholds**, and outputs a full probability
distribution per item (mean, spread, P(>=9)). Ordinal models consistently
beat regression on explicit ratings and give principled uncertainty.

**malrec today.** Least-squares regression on 1-10 treated as interval data;
a linear map per user for display.

**To do.** Two options, in order of cost:
1. Keep the regression but report **uncertainty**: split-conformal intervals
   per user from the temporal holdout residuals ("likely 7-9", ~80% coverage),
   distribution-free and cheap.
2. An ordinal head (per-user thresholds on the current score): gives
   "chance you'd rate it 9+". That answers the "is 8.0 a masterpiece?" question
   directly - a title with expected 8.0 and wide spread is a gamble; one with
   8.0 and narrow spread is a safe bet. It could define Safe Bets properly
   (high lower bound, not just a high mean).

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
X and I gave X+1". Option: show a coarse band ("likely 7-9") instead of 7.84,
which also fits section 2. Keep evaluation on ratings given before the app
existed where possible.

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

**To do.** Low cost, useful: popularity-based propensities (P(watched)
~ popularity, per list size) and IPS-weighted versions of the key metrics, to
check that no conclusion flips. Training with IPS is a later step.

## 5. Which population model: the reproducibility evidence

**Literature.** Ferrari Dacrema et al. (RecSys 2019, TOIS 2021): 11 of 12
reproducible neural methods lost to tuned kNN / linear baselines. Rendle et
al. (RecSys 2022): tuned iALS is competitive with or beats newer methods.
**EASE** (Steck, WWW 2019): a closed-form item-item linear model, often the
strongest top-N baseline. For sequential models, SASRec properly trained
(gSASRec, RecSys 2023) matches later models.

**malrec today.** Shrunk item-kNN + ALS + co-occurrence + a stacker - i.e.
the tuned-simple family that holds up best. Good.

**To do.** EASE as one more population signal for **retrieval** (it's
excellent at "what will they watch"): one ~15k x 15k solve, cheap to test in
`exp_final`. Sequential and generative models: not at this data size.

## 6. Lists, not just scores: calibration and diversity

**Literature.** Steck, *Calibrated Recommendations* (RecSys 2018): the genre
mix of a list should match the user's history (KL divergence, greedy
re-ranking); survey in ACM TORS 2025.

**malrec today.** A hand-set genre-overlap penalty (`diversity_weight`).

**To do.** Replace it with calibrated re-ranking, and measure recall and
list-level genre KL. A principled version of the same idea.

## 7. LLMs

**Literature (2024-26).** On rating prediction, classic CF beats LLMs (lower
RMSE/MAE). On ranked lists, strong LLMs are competitive; hybrids feeding CF
signals into an LLM report sizeable gains on sparse or cold-start items.

**For malrec.** Not for scoring. Plausible later uses: natural-language
explanations from the existing breakdown, or cold-start content for titles
with no ratings (Coming Soon) - both optional.

## 8. Evaluation methodology

**Literature.** Time-aware evaluation should split by time; leave-one-out
and random splits leak the future (Campos et al. 2014; RecSys 2025
"Time to Split").

**malrec today.** Per-user temporal holdouts with cutoff-consistent
features, plus strictly out-of-time nested tests (exp_displaycal). In line
with best practice; a global time split for the population sample would be
stricter still.

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

## Recommended order

1. Finish the level-drift experiment; if it passes, add a day effect and
   re-tune the recency half-life with the drift term in place (section 1).
2. Conformal intervals, then ordinal probabilities; define Safe Bets by the
   lower bound (sections 2-3).
3. IPS-weighted sanity check of the key conclusions (section 4).
4. EASE as a retrieval signal; calibrated re-ranking in place of the genre
   penalty (sections 5-6).

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
- Time-aware evaluation: Campos et al. survey
  https://www.researchgate.net/publication/257671605 ; Time to Split (RecSys 2025):
  https://arxiv.org/pdf/2507.16289
