-- Admin e-mails (malrec.notify): one row per mail sent or attempted, so the
-- same event is never mailed twice and task failures can be bundled (at most
-- one mail per 30 minutes).
CREATE TABLE IF NOT EXISTS notification (
    id       bigserial   PRIMARY KEY,
    kind     text        NOT NULL,
    key      text        NOT NULL,
    sent_at  timestamptz NOT NULL DEFAULT now(),
    ok       boolean     NOT NULL DEFAULT false,
    error    text,
    UNIQUE (kind, key)
);
CREATE INDEX IF NOT EXISTS notification_kind_idx ON notification (kind, sent_at DESC);
