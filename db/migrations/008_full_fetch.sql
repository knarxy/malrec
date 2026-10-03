-- One-time full fetch: catalogue-wide enrichment and a collaborative-filtering
-- sample of public MyAnimeList lists.
--
-- Everything the fetch touches is persisted as it arrives and stamped, so an
-- interrupted run resumes where it stopped and nothing is ever requested twice.

-- ------------------------------------------------------------ enrichment --

ALTER TABLE anime ADD COLUMN IF NOT EXISTS al_status_dist  jsonb;
ALTER TABLE anime ADD COLUMN IF NOT EXISTS al_score_dist   jsonb;
ALTER TABLE anime ADD COLUMN IF NOT EXISTS mal_status_dist jsonb;
-- Share of everyone who started a show and then dropped it. A quality signal
-- that a mean score hides: a 7.5 that half its audience abandons is not the
-- same show as a 7.5 people finish.
ALTER TABLE anime ADD COLUMN IF NOT EXISTS al_drop_rate    real;
ALTER TABLE anime ADD COLUMN IF NOT EXISTS mal_drop_rate   real;

CREATE TABLE IF NOT EXISTS anime_staff (
    mal_id    integer NOT NULL REFERENCES anime (mal_id) ON DELETE CASCADE,
    staff_id  integer NOT NULL,           -- AniList staff id
    name      text    NOT NULL,
    role      text    NOT NULL,
    PRIMARY KEY (mal_id, staff_id, role)
);
CREATE INDEX IF NOT EXISTS anime_staff_staff_idx ON anime_staff (staff_id);

-- ------------------------------------------- collaborative filtering data --

-- Public lists of MyAnimeList users found through the forums. The username is
-- only kept until the list has been fetched; after that the row is identified
-- by a hash, which is all deduplication needs. Nothing else about the person
-- is stored.
CREATE TABLE IF NOT EXISTS cf_user (
    id            serial PRIMARY KEY,
    name_hash     text        NOT NULL UNIQUE,
    name          text,
    source        text        NOT NULL DEFAULT 'mal_forum',
    state         text        NOT NULL DEFAULT 'pending'
                  CHECK (state IN ('pending', 'done', 'private', 'error', 'excluded')),
    n_entries     integer,
    n_scored      integer,
    discovered_at timestamptz NOT NULL DEFAULT now(),
    fetched_at    timestamptz
);
CREATE INDEX IF NOT EXISTS cf_user_state_idx ON cf_user (state);

CREATE TABLE IF NOT EXISTS cf_rating (
    user_id     integer  NOT NULL REFERENCES cf_user (id) ON DELETE CASCADE,
    mal_id      integer  NOT NULL,
    score       smallint NOT NULL DEFAULT 0,
    status      text     NOT NULL,
    finished_at date,
    updated_at  timestamptz,
    PRIMARY KEY (user_id, mal_id)
);
CREATE INDEX IF NOT EXISTS cf_rating_item_idx ON cf_rating (mal_id) WHERE score > 0;

-- Item-item similarity learned from cf_rating, top-K neighbours per item.
CREATE TABLE IF NOT EXISTS item_sim (
    a        integer NOT NULL,
    b        integer NOT NULL,
    sim      real    NOT NULL,
    support  integer NOT NULL,            -- users who rated both
    PRIMARY KEY (a, b)
);
CREATE INDEX IF NOT EXISTS item_sim_b_idx ON item_sim (b);

-- Learned latent factors (matrix factorisation) and population item biases.
CREATE TABLE IF NOT EXISTS item_factor (
    mal_id    integer PRIMARY KEY,
    bias      real    NOT NULL,           -- how far above their own mean people rate it
    n_ratings integer NOT NULL,
    factors   real[]  NOT NULL
);

-- --------------------------------------------------------- resumability --

-- Cursor per fetch stage (e.g. how far through a forum board), so a restarted
-- run continues rather than starting over.
CREATE TABLE IF NOT EXISTS fetch_state (
    stage       text PRIMARY KEY,
    cursor      jsonb NOT NULL DEFAULT '{}',
    done        boolean NOT NULL DEFAULT false,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
