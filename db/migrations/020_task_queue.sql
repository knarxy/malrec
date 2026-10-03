-- Background work (rebuilds, list syncs, onboarding) runs in a separate worker
-- process (`malrec worker`) instead of the API's request threads, so a rebuild
-- never slows the app for everyone else. The API only enqueues.
--
-- At most one queued task per user and kind: a second request while one is
-- waiting is absorbed by it (five quick ratings cost at most two rebuilds - the
-- running one and one queued). The worker runs one task per user at a time.
CREATE TABLE IF NOT EXISTS task (
    id           bigserial   PRIMARY KEY,
    kind         text        NOT NULL
                 CHECK (kind IN ('rebuild', 'sync', 'login_sync', 'onboard')),
    user_id      integer     NOT NULL REFERENCES app_user(id) ON DELETE CASCADE,
    priority     smallint    NOT NULL DEFAULT 0,
    state        text        NOT NULL DEFAULT 'queued'
                 CHECK (state IN ('queued', 'running', 'done', 'failed')),
    attempts     smallint    NOT NULL DEFAULT 0,
    error        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    started_at   timestamptz,
    finished_at  timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS task_one_queued ON task (user_id, kind) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS task_queue_idx ON task (priority, id) WHERE state = 'queued';
CREATE INDEX IF NOT EXISTS task_user_idx ON task (user_id, id DESC);
