-- What a signed-in user actually saw and opened in the app (malrec.usage):
-- the app opened, a tab viewed, a card on screen for a second, a title's
-- details opened. One row per thing per day with a counter, so a reload or a
-- scroll back adds nothing but a count. Actions (ratings, Plan to Watch,
-- Not for me) stay in feedback. Kept 180 days; deleted with the account.
CREATE TABLE IF NOT EXISTS app_event (
    user_id   integer     NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    kind      text        NOT NULL,
    surface   text        NOT NULL DEFAULT '',
    mal_id    integer     NOT NULL DEFAULT 0,
    day       date        NOT NULL DEFAULT current_date,
    rank      integer,
    n         integer     NOT NULL DEFAULT 1,
    first_at  timestamptz NOT NULL DEFAULT now(),
    last_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, kind, surface, mal_id, day)
);
CREATE INDEX IF NOT EXISTS app_event_day_idx ON app_event (day);
