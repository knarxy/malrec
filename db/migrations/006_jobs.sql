-- Background job tracking.
--
-- Onboarding a new user takes a few minutes (their list, then one MAL detail
-- call per entry). The web app needs to show progress rather than a spinner
-- with no end in sight, so each run records its current step here.

CREATE TABLE IF NOT EXISTS job (
    id          bigserial PRIMARY KEY,
    kind        text NOT NULL,
    username    text,
    state       text NOT NULL DEFAULT 'running'
                CHECK (state IN ('running', 'done', 'failed')),
    step        text,
    step_index  smallint NOT NULL DEFAULT 0,
    step_total  smallint NOT NULL DEFAULT 1,
    detail      jsonb NOT NULL DEFAULT '{}',
    error       text,
    started_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

CREATE INDEX IF NOT EXISTS job_lookup_idx ON job (username, kind, started_at DESC);
-- At most one live job per user, so a double-click cannot start two crawls.
CREATE UNIQUE INDEX IF NOT EXISTS job_one_running_idx
    ON job (username, kind) WHERE state = 'running';
