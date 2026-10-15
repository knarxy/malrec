from __future__ import annotations

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Containers get every setting as an environment variable and may not be
    # allowed to read a host-owned .env (it holds the client secret), so the
    # file is optional rather than a hard requirement.
    model_config = SettingsConfigDict(
        env_file=".env" if os.access(".env", os.R_OK) else None, extra="ignore")

    database_url: str = "postgresql://malrec:malrec@localhost:5433/malrec"

    mal_client_id: str = ""
    mal_client_secret: str = ""
    # Must match the "App Redirect URL" registered for this client at
    # https://myanimelist.net/apiconfig exactly, or MAL rejects the login.
    mal_redirect_uri: str = "http://localhost:3000/api/auth/callback"
    # Where the browser is sent after signing in.
    app_base_url: str = "http://localhost:3000"
    # Set true behind HTTPS so the session cookie is never sent in the clear.
    session_cookie_secure: bool = False
    session_days: int = 30
    malrec_user: str = ""
    # Comma-separated MAL usernames with admin rights (approve users, admin
    # panel, viewing any profile). Config, not a database flag, on purpose.
    admin_users: str = ""
    # Admin e-mails (malrec.notify): off unless SMTP_HOST and ADMIN_EMAIL are
    # set. STARTTLS on SMTP_PORT; SMTP_FROM defaults to SMTP_USER. Times in
    # the mails are shown in NOTIFY_TIMEZONE.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    admin_email: str = ""
    notify_timezone: str = "UTC"
    # Test phase: approve new accounts automatically while fewer than this many
    # accounts are approved; after that they wait for the admin again. 0 = off.
    auto_approve_limit: int = 0
    # How users reach whoever runs this instance (a URL or an address), shown
    # on the privacy page. Empty: the page says "the operator of this site".
    operator_contact: str = ""
    # Background work (rebuilds, syncs, onboarding) runs in `malrec worker`.
    # tasks_inline runs it in a thread of the API instead, for local
    # development without a worker; worker_threads > 1 runs several users'
    # tasks at once (one per user at a time either way).
    tasks_inline: bool = False
    worker_threads: int = 1
    # Users whose loaded model the API keeps in memory (~20 MB each).
    scorer_cache_size: int = 16
    # Fernet key for MAL tokens at rest (malrec.tokenbox); generate with
    # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    token_key: str = ""
    # Interactive API docs (/docs, /openapi.json); off unless developing.
    enable_docs: bool = False
    # Where the scheduled jobs write their logs (mounted read-only into the
    # API container for the admin panel's job status).
    logs_dir: str = "/app/logs"

    # Catalog ingest
    ranking_types: tuple[str, ...] = (
        "all", "tv", "movie", "ova", "special", "bypopularity", "favorite", "airing", "upcoming",
    )
    ranking_pages: int = 2           # 2 x 500 per ranking type
    # MyAnimeList publishes no rate limit (not in the API reference, the auth
    # docs or the API License Agreement, and no rate-limit response headers).
    # The agreement only forbids "unreasonable burden". 1 req/s is the one
    # concrete figure in circulation; ~3/s sustained triggers 307 load
    # shedding. Chosen conservatively because bulk jobs run for hours.
    mal_rps: float = 1.0
    anilist_batch: int = 50          # GraphQL Page(perPage:50)
    # Starting guess only: the client re-reads the live limit from the
    # X-RateLimit-Limit header on every response (documented 90/min, currently
    # degraded to 30/min - https://docs.anilist.co/guide/rate-limiting).
    anilist_rpm: int = 30

    # Recency: taste drifts, so recent ratings weigh more. Both values are
    # tuned by `malrec tune-recency`; these defaults come from a temporal
    # holdout on the reference profile (rho 0.654 -> 0.765 vs no weighting).
    recency_half_life_years: float = 0.5
    recency_floor: float = 0.05

    # The recommendation graph is a strong retriever and the best source of
    # explanations, but as a model *feature* it does not survive contact with
    # the candidate set: affinity averages 0.082 across the user's own rated
    # anime and 0.0014 across candidates, a 50x shift, so the learned weight is
    # calibrated to a distribution that never occurs at scoring time. Measured
    # over 8 temporal holdouts, switching it off improved Spearman in 6 of them
    # (mean +0.015). Re-check with `malrec ablate` if the feature set changes.
    use_affinity_features: bool = False

    # Content signals from the full fetch, each off until it has shown a
    # measured gain (experiments/exp_signals.py):
    #   content_staff      director / writer / creator / composer as tokens
    #   content_audience   drop rate and score polarisation as numeric columns
    #   content_staff_aff  how the user rated other work by the same people
    #   stack_extras       the same signals offered to the population stacker
    content_staff: bool = False
    content_audience: bool = False
    content_staff_aff: bool = False
    stack_extras: tuple[str, ...] = ()
    # The user's rating level as a slow trend in time, learned by the personal
    # model next to their taste, with predictions at today's level: "" (off),
    # "lin3" (linear in years, capped at 3), "log" (log1p of years) or
    # "koren" (sign(t - t_u)|t - t_u|^0.4 in days, Koren KDD 2009).
    # See experiments/exp_signals.py (level variants).
    level_trend: str = ""
    # Rating round: how the next card is chosen. Simulated on held-out users
    # (experiments/exp_elicit.py, 2026-09-27): after 15 ratings "popular"
    # doubled recall@50 (.120 -> .239) and beat "smart" (P(seen) x spread of
    # opinion: .178) on rho, recall and IPS recall alike.
    quiz_strategy: str = "popular"
    # From this many ratings the personal/population blend weight is chosen
    # on the user's own newest ratings instead of from list size
    # (experiments/exp_audit.py: +0.025 rho at 400 ratings; at 60-150 it
    # does not help, and on the 143-rating reference profile it costs 0.023).
    chosen_blend_min: int = 300
    # Centre of the size-weighted handover from the population stack to the
    # personal model (50/50 at this many ratings, logistic with scale 15),
    # and a ceiling on the personal share. experiments/exp_handover.py
    # (2026-10-03): moving 80 -> 150 and capping at 0.5 lifts rank correlation
    # at 150 entries .554 -> .580, at 250 .550 -> .575, at 400 .577 -> .595,
    # and the reference profile .753 -> .773 (6 of 8 splits); at 60 entries
    # -.004, within noise. The population layer is the better predictor for
    # almost everyone; the personal model is a bounded correction.
    blend_size_n0: float = 150.0
    blend_lam_max: float = 0.5
    # What the ranking dials (relevance pull, Safe Bets popularity prior)
    # treat as "how established is this list". 0: the blend's personal share,
    # as before. > 0: list size on a logistic centred here (scale 15), so
    # moving the blend handover does not silently re-tune the dials, which
    # were set against the n0 = 80 curve. Measured (exp_handover.py): left
    # coupled, the new handover would push the reference profile's Safe Bets
    # top 20 to a median popularity rank of #35; decoupled at 80, recall@50
    # and its IPS version stay at or above today's (.334 -> .339, .248 -> .264)
    # with the list's character kept (#96 -> #196).
    ranking_share_n0: float = 80.0
    # Risk in the order key. The first two are scaled by the trust in the
    # user's own model (personal share / blend_lam_max), so thin lists are
    # left alone; the length rule applies at every size (it also counts a
    # series airing for 3+ years - exp_toplist.py round 4: long titles in
    # thin lists' top 10 .13 -> .09, hit rate and recall unchanged):
    #   disagreement_penalty      points per point by which the population
    #                             layer is more optimistic than the personal
    #                             model (devotee titles: Hajime no Ippo led the
    #                             reference profile's Safe Bets at 9.07 vs 7.91)
    #   unfamiliar_genre_penalty  a genre absent from the user's whole list
    #   long_series_penalty       a 50+ episode series
    # experiments/exp_toplist.py, two user splits: reference profile top-10
    # hit rate .169 -> .244, recall@50 .400 -> .427, long / unfamiliar titles
    # in the top 10 ~.15 -> ~0; held-out users within noise (bad picks at 400
    # entries .015 -> .010). 0 = off.
    disagreement_penalty: float = 0.25
    unfamiliar_genre_penalty: float = 0.3
    long_series_penalty: float = 0.3
    # Evidence check (exp_toplist.py round 7, two user splits): points off per
    # point of the personal model's lift that comes from acclaim alone (MAL /
    # AniList score, popularity), fading with co-rating evidence from the
    # user's own rated titles, e0 / (e0 + evidence). Cowboy Bebop's personal
    # 8.22 was +0.79 acclaim, +0.04 content, no co-rating link. Top-10 hit
    # rate at 400 entries .386 -> .431 / .379 -> .411, recall@50 +.011 on
    # both; reference profile .375 -> .412 / .400 -> .450, pre-2005 picks to
    # 0; thin lists untouched (trust-scaled). 0 = off.
    acclaim_penalty: float = 0.6
    acclaim_evidence_e0: float = 0.1
    # Safe Bets: popularity prior for thin lists, weight * (1 - personal share)
    # in standard deviations of the Safe Bets key (exp_audit.py: recall@50 up
    # at every list size, IPS-weighted recall level or better; 1.0 hurt IPS).
    safe_bets_popularity: float = 0.5
    # Safe Bets' order blends in a model with every rating weighted equally
    # (0 = off); displayed scores are unaffected. See experiments/exp_tabs.py.
    safe_bets_memory_mix: float = 0.0
    # Safe Bets guards (experiments/exp_toplist.py rounds 5-6): the
    # long-memory adjustment is clipped to +-cap points (0 = no cap), and
    # nothing the current model predicts more than `floor` below the user's
    # own mean is a safe bet (-99 = no floor). Mahouka (6.90, mean 7.51)
    # reached #8 on a +0.65 memory adjustment alone. Cap 0.15, two user
    # splits: reference profile top-10 hits .225 -> .263 / .275 -> .300,
    # held-out users within noise. The floor showed no benefit: off.
    safe_bets_memory_cap: float = 0.15
    safe_bets_floor: float = -99.0
    # Safe Bets' order mixed with a learned one (recsys/ltr.py), this share
    # of it (0 = off, 0.5 = the MIX of experiments/exp_ltr.py: recall@50
    # +0.020-0.027 and hit@10 +0.04-0.09 at 25/150/400 entries, reference
    # profile hit@10 +0.11, lists more popular - median rank #154 -> #66 for
    # the reference profile). Needs a ranker for the active population
    # model (malrec ranker train); without one the hand-tuned key applies.
    order_ranker_weight: float = 0.0
    # The learned order's training labels weighted against popularity
    # (recsys/ltr.py, inverse_propensity): 0 = plain labels. 0.5 and 1 lifted
    # Safe Bets' IPS recall@50 (+.006-.011 at 150/400 entries) by taking the
    # lesser-known hits Discover showed (its IPS recall -.005 to -.011): a
    # shift between tabs, not a gain (exp_blend.py, 2026-10-11). Off.
    order_ranker_ips: float = 0.0
    # Learned orders for Discover and Hidden Gems (each its own ranker); 0 = off.
    # 0.5 (exp_blend.py, 2026-10-11): Discover recall@50 +.015/+.015/+.024 at
    # 25/150/400 entries, reference profile .038 -> .091; Hidden Gems recall@50
    # +.008/+.015 and hit@10 +.023/+.044 at 150/400; Safe Bets untouched.
    discover_ranker_weight: float = 0.0
    gems_ranker_weight: float = 0.0
    # Share of the risk term in Safe Bets' hand-built key; the learned order
    # reads the risk parts as features whatever this is. 0 (risk left to the
    # ranker alone) cost recall@50 -.008 at 400 entries (exp_blend.py,
    # 2026-10-11): the key keeps it.
    safe_bets_key_risk: float = 1.0
    # EASE as a second relevance signal (fitted with the population model)
    relevance_ease: bool = True

    # Which user model ships - see recsys/hybrid.py for the modes and
    # experiments/exp_arch.py for how they compare. Set from the evaluation.
    model_mode: str = "personal"

    # Implicit signal: the unscored half of a list is still an opinion.
    # Offsets are relative to the user's recency-weighted mean and are global
    # rather than per-user, because a single profile has far too few scored
    # examples per status to fit them (n=1 for dropped and on_hold on the
    # reference profile). Fitted on the temporal holdout with
    # `malrec tune-implicit`: Spearman 0.756 -> 0.769, RMSE 1.146 -> 1.111,
    # nDCG@10 0.797 -> 0.810, improving on 7 of 8 splits. Every weight between
    # 0.3 and 0.8 beat the baseline, so the exact value is not delicate.
    # plan_to_watch is deliberately absent: it is an interest signal, not a
    # quality one, and including it cost nDCG.
    implicit_weight: float = 0.8
    implicit_offsets: dict[str, float] = Field(
        default_factory=lambda: {"dropped": -2.0, "on_hold": -1.0, "completed": 0.0})
    # "Not for me" beyond hiding the one title (experiments/exp_notforme.py):
    # it becomes an implicit rating `not_interested_offset` under the user's
    # mean (weight relative to a real rating; 0 turns it off), and the rest of
    # its franchise leaves the discovery tabs unless the user has something
    # of that franchise on their list.
    # Simulated (2 rounds): -1 / 0.4 in the personal model *and* the fold-in,
    # plus the franchise rule, beat hiding alone at every list size - hit@10
    # +0.024 at 150 entries, recall@50 +0.008 at 400 - for a rank-correlation
    # cost inside the noise (<= 0.009); -2 / 0.8 and stronger cost 0.03-0.04.
    not_interested_offset: float = -1.0
    not_interested_weight: float = 0.4
    not_interested_franchise: bool = True
    not_interested_foldin: bool = True       # also in the population fold-in

    # Ranking
    min_scoring_users: int = 2000
    shortlist_size: int = 600
    # Rank = predicted score + relevance bonus + novelty, in rating points per
    # standard deviation of co-occurrence relevance (see rank.relevance_bonus).
    # relevance_weight applies in proportion to the population share (thin
    # lists); relevance_weight_personal to long histories, bounded at +-2 sd.
    # The latter defaults to 0 so an established profile's lists do not shift
    # toward the mainstream unless its owner opts in.
    relevance_weight: float = 0.5
    relevance_weight_personal: float = 0.0
    novelty_weight: float = 1.0      # "balanced", chosen by the user
    # Prediction spread (sd of shown scores across a candidate list) at which
    # novelty gets its full weight; flatter models get proportionally less, so
    # novelty can break ties but never decide a ranking on its own. Measured
    # spreads: 0.34-0.48 for every current user (0.344 for the reference
    # profile), ~0 for the collapsed legacy model this guards against. Kept
    # below all of them so a working model always gets the full setting.
    novelty_ref_sd: float = 0.30
    diversity_weight: float = 0.07    # fallback when no genre profile is known
    # How much a series' entries count as evidence of taste: each entry
    # weighs 1 / n ** this, n the user's entries in that series
    # (recsys/series.py). Thirteen seasons of one show rated 10 say "loves this
    # show", not thirteen times "loves its genres". 0 = every entry counts
    # fully. Applies to the user model's ratings and listed titles and the
    # population stacker's training lists. Off: 0.5 and 1 cost Discover and
    # Hidden Gems accuracy on long lists (2026-10-15). The genre profile
    # weights series on its own (series.PROFILE_SERIES_POWER).
    series_weight_power: float = 0.0
    # Calibrated re-ranking strength (Steck 2018); 0 = the old overlap penalty
    genre_calibration: float = 0.7
    # The same for the age mix (rank.age_mix: how old titles were when the
    # user watched them): the weight of its KL term in Safe Bets, Discover and
    # Hidden Gems; 0 = off. Shipped at 1 with calibration_block 20 and the
    # ranker's age features (exp_blend.py, 2026-10-13): the distance of the
    # top 20's age mix from the user's own fell by about half on every tab
    # (Safe Bets .29-.33 -> .15-.18 at 25/150/400 entries) while recall@50
    # rose (+.023/+.009/+.005); at 2, recall gives part of that back.
    age_calibration: float = 0.0
    # Calibrated lists are ordered by the ranking key within blocks of this
    # many titles (0 = the whole list, as measured in exp_calib.py). Without
    # blocks the calibrated set is right but its newest titles still sort to
    # the top: the visible top 20 hardly changed.
    calibration_block: int = 0

    @property
    def conninfo(self) -> str:
        return self.database_url


@lru_cache
def settings() -> Settings:
    return Settings()
