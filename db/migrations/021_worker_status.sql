-- The worker's heartbeat, for the admin panel: one row, updated every 30 s
-- while the worker runs. A stalled worker (alive but stuck) is otherwise
-- invisible - Docker restarts a crashed container, not a hung one.
CREATE TABLE IF NOT EXISTS worker_status (
    id          smallint    PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    started_at  timestamptz NOT NULL,
    beat_at     timestamptz NOT NULL,
    threads     smallint    NOT NULL DEFAULT 1
);
