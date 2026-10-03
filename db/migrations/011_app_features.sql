-- Account preferences, live re-ranking, learning from in-app actions.

-- Language, discovery slider and filters, saved for a signed-in account.
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS prefs jsonb NOT NULL DEFAULT '{}';

-- The scored shortlist behind each surface, kept so the list can be re-ranked
-- per request (discovery slider, filters) without refitting anything. The
-- `recommendation` table keeps the default ranking.
CREATE TABLE IF NOT EXISTS rec_candidate (
    user_id       integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    surface       text    NOT NULL,
    mal_id        integer NOT NULL,
    franchise_id  integer NOT NULL,
    predicted     real    NOT NULL,     -- calibrated, as shown
    relevance_z   real    NOT NULL DEFAULT 0,
    personal_share real   NOT NULL DEFAULT 1,
    novelty       real    NOT NULL DEFAULT 0,
    base_rank     integer NOT NULL,     -- order before re-ranking
    reasons       jsonb   NOT NULL DEFAULT '[]',
    generated_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, surface, mal_id)
);

-- What was shown, with the ranking inputs at the time. Joined later with
-- feedback and list changes to learn how much relevance and novelty should
-- weigh (`malrec learn-weights`).
CREATE TABLE IF NOT EXISTS rec_log (
    id            bigserial PRIMARY KEY,
    user_id       integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    surface       text    NOT NULL,
    mal_id        integer NOT NULL,
    rank          integer NOT NULL,
    predicted     real    NOT NULL,
    relevance_z   real    NOT NULL DEFAULT 0,
    novelty       real    NOT NULL DEFAULT 0,
    personal_share real   NOT NULL DEFAULT 1,
    shown_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS rec_log_user_idx ON rec_log (user_id, mal_id, shown_at);

-- 'rated' records a score given from the app (the score itself is on the
-- list); the allowed actions are set once, further down, with the rating
-- round's answers included - migrations replay on every start, so an
-- earlier, narrower version of the check would fail once those rows exist.

-- the sample is refreshed by rotation: old lists are retired, new ones added
ALTER TABLE cf_user DROP CONSTRAINT IF EXISTS cf_user_state_check;
ALTER TABLE cf_user ADD CONSTRAINT cf_user_state_check CHECK (state IN
    ('pending', 'done', 'private', 'error', 'excluded', 'retired'));

-- cooldown for the manual "sync with MyAnimeList" button
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS last_manual_sync_at timestamptz;

-- the novelty scale the build used (it depends on the whole scored pool, not
-- just the kept shortlist), so the default re-ranking reproduces it exactly
ALTER TABLE rec_candidate ADD COLUMN IF NOT EXISTS novelty_scale real NOT NULL DEFAULT 1;
-- and the list length it cut at, which decides how far down rerank() looks
ALTER TABLE rec_candidate ADD COLUMN IF NOT EXISTS build_limit integer NOT NULL DEFAULT 60;
-- the number a list card shows, when it differs from the ranking value
-- (population top-of-list calibration for small accounts)
ALTER TABLE rec_candidate ADD COLUMN IF NOT EXISTS shown real;
-- extra ordering term (Safe Bets' long-memory mix), kept for re-ranking
ALTER TABLE rec_candidate ADD COLUMN IF NOT EXISTS bonus real NOT NULL DEFAULT 0;
-- rating-round answers that are not ratings, so a title is not asked twice
ALTER TABLE feedback DROP CONSTRAINT IF EXISTS feedback_action_check;
ALTER TABLE feedback ADD CONSTRAINT feedback_action_check CHECK (action IN
    ('not_interested', 'queued', 'unqueued', 'hidden', 'seen_it', 'liked', 'clicked', 'rated',
     'quiz_unseen', 'quiz_skip'));
