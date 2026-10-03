-- Core catalog, graph and user-list schema.
-- Everything an ingest writes lands here; the recommender only ever reads from it.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------- catalog --

CREATE TABLE IF NOT EXISTS anime (
    mal_id                  integer PRIMARY KEY,
    title                   text        NOT NULL,
    title_en                text,
    title_ja                text,
    synopsis                text,
    media_type              text,
    status                  text,
    source                  text,
    rating                  text,
    nsfw                    text,
    num_episodes            integer,
    avg_episode_seconds     integer,
    start_date              date,
    end_date                date,
    season_year             smallint,
    season                  text,
    picture_medium          text,
    picture_large           text,

    mal_mean                real,
    mal_rank                integer,
    mal_popularity          integer,
    mal_num_list_users      integer,
    mal_num_scoring_users   integer,
    mal_genres              text[]      NOT NULL DEFAULT '{}',
    mal_studios             text[]      NOT NULL DEFAULT '{}',

    anilist_id              integer,
    al_average_score        smallint,
    al_mean_score           smallint,
    al_popularity           integer,
    al_favourites           integer,
    al_genres               text[]      NOT NULL DEFAULT '{}',

    -- taste/content embedding derived from AniList weighted tags (see 003)
    tag_vec                 vector(256),

    mal_fetched_at          timestamptz,
    al_fetched_at           timestamptz,
    raw_mal                 jsonb,
    raw_al                  jsonb,

    search_tsv              tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('simple', coalesce(title, '')),    'A') ||
        setweight(to_tsvector('simple', coalesce(title_en, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(synopsis, '')), 'D')
    ) STORED
);

CREATE INDEX IF NOT EXISTS anime_search_idx      ON anime USING gin (search_tsv);
CREATE INDEX IF NOT EXISTS anime_genres_idx      ON anime USING gin (mal_genres);
CREATE INDEX IF NOT EXISTS anime_studios_idx     ON anime USING gin (mal_studios);
CREATE INDEX IF NOT EXISTS anime_title_trgm_idx  ON anime USING gin (title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS anime_popularity_idx  ON anime (mal_popularity) WHERE mal_popularity IS NOT NULL;
CREATE INDEX IF NOT EXISTS anime_season_idx      ON anime (season_year, season);
CREATE UNIQUE INDEX IF NOT EXISTS anime_anilist_idx ON anime (anilist_id) WHERE anilist_id IS NOT NULL;

-- AniList weighted tags, kept long-form so they can be aggregated and filtered.
CREATE TABLE IF NOT EXISTS anime_tag (
    mal_id      integer  NOT NULL REFERENCES anime (mal_id) ON DELETE CASCADE,
    tag         text     NOT NULL,
    rank        smallint NOT NULL,          -- AniList relevance, 0..100
    category    text,
    PRIMARY KEY (mal_id, tag)
);
CREATE INDEX IF NOT EXISTS anime_tag_tag_idx ON anime_tag (tag) INCLUDE (rank);
CREATE INDEX IF NOT EXISTS anime_tag_rank_idx ON anime_tag (mal_id, rank DESC);

-- ------------------------------------------------------------------ graph --

-- "people who liked src also recommend dst" - stored BOTH directions so a
-- lookup on either endpoint is a single index hit.
CREATE TABLE IF NOT EXISTS rec_edge (
    src        integer NOT NULL,
    dst        integer NOT NULL,
    provider   text    NOT NULL CHECK (provider IN ('mal', 'anilist')),
    votes      integer NOT NULL DEFAULT 0,
    weight     real    NOT NULL DEFAULT 0,
    PRIMARY KEY (src, dst, provider)
);
CREATE INDEX IF NOT EXISTS rec_edge_dst_idx ON rec_edge (dst, provider);

-- Franchise structure: sequels, prequels, side stories, alternative versions.
CREATE TABLE IF NOT EXISTS relation (
    src            integer NOT NULL,
    dst            integer NOT NULL,
    relation_type  text    NOT NULL,        -- normalised: sequel/prequel/side_story/...
    provider       text    NOT NULL CHECK (provider IN ('mal', 'anilist')),
    PRIMARY KEY (src, dst, relation_type, provider)
);
CREATE INDEX IF NOT EXISTS relation_dst_idx  ON relation (dst, relation_type);
CREATE INDEX IF NOT EXISTS relation_type_idx ON relation (relation_type);

-- Connected components over `relation`, maintained by refresh_franchises().
-- Lets a query collapse "all of Monogatari" to one row in a single join.
CREATE TABLE IF NOT EXISTS franchise (
    mal_id        integer PRIMARY KEY,
    franchise_id  integer NOT NULL
);
CREATE INDEX IF NOT EXISTS franchise_fid_idx ON franchise (franchise_id);

-- ------------------------------------------------------------------ users --

CREATE TABLE IF NOT EXISTS app_user (
    id            serial PRIMARY KEY,
    mal_username  text   NOT NULL UNIQUE,
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_sync_at  timestamptz
);

CREATE TABLE IF NOT EXISTS list_entry (
    user_id           integer  NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    mal_id            integer  NOT NULL,
    status            text     NOT NULL,   -- watching/completed/on_hold/dropped/plan_to_watch
    score             smallint NOT NULL DEFAULT 0,
    episodes_watched  integer  NOT NULL DEFAULT 0,
    is_rewatching     boolean  NOT NULL DEFAULT false,
    started_at        date,
    finished_at       date,
    updated_at        timestamptz,
    PRIMARY KEY (user_id, mal_id)
);
CREATE INDEX IF NOT EXISTS list_entry_scored_idx ON list_entry (user_id) WHERE score > 0;
CREATE INDEX IF NOT EXISTS list_entry_status_idx ON list_entry (user_id, status);

-- Explicit signals from the web app. Append-only; latest row per (user, anime,
-- action) wins so a user can undo by posting the inverse action.
CREATE TABLE IF NOT EXISTS feedback (
    id          bigserial PRIMARY KEY,
    user_id     integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    mal_id      integer NOT NULL,
    action      text    NOT NULL CHECK (action IN
                  ('not_interested', 'queued', 'hidden', 'seen_it', 'liked', 'clicked')),
    surface     text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS feedback_user_idx ON feedback (user_id, mal_id, created_at DESC);

-- --------------------------------------------------------------- modelling --

CREATE TABLE IF NOT EXISTS model_run (
    id             serial PRIMARY KEY,
    user_id        integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    algo           text    NOT NULL,        -- 'ridge' | 'lgbm'
    params         jsonb   NOT NULL DEFAULT '{}',
    metrics        jsonb   NOT NULL DEFAULT '{}',
    feature_names  text[]  NOT NULL DEFAULT '{}',
    n_train        integer,
    artifact       bytea,
    trained_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS model_run_user_idx ON model_run (user_id, trained_at DESC);

CREATE TABLE IF NOT EXISTS recommendation (
    user_id          integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    surface          text    NOT NULL,
    mal_id           integer NOT NULL,
    rank             integer NOT NULL,
    predicted_score  real    NOT NULL,
    final_score      real    NOT NULL,
    novelty          real,
    reasons          jsonb   NOT NULL DEFAULT '[]',
    model_run_id     integer REFERENCES model_run (id) ON DELETE SET NULL,
    generated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, surface, mal_id)
);
CREATE INDEX IF NOT EXISTS recommendation_rank_idx ON recommendation (user_id, surface, rank);

-- Bookkeeping so an incremental refresh knows what is already fresh.
CREATE TABLE IF NOT EXISTS ingest_log (
    id          bigserial PRIMARY KEY,
    job         text NOT NULL,
    status      text NOT NULL,
    detail      jsonb NOT NULL DEFAULT '{}',
    started_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);
CREATE INDEX IF NOT EXISTS ingest_log_job_idx ON ingest_log (job, started_at DESC);
