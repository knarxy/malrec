-- What happened to a title on a user's list, and when (audit 2026-10-23,
-- P1-7). The real-use check (malrec eval-lists) judged a shown title by the
-- list as it is today, timed by list_entry.updated_at, which every later
-- change overwrites: a title rated well in its window and dropped weeks
-- later counted as a drop outside it. This records every change of status
-- or score from any path - the list sync, an in-app rating or queue - with
-- its time, so an outcome is the one inside the window.
CREATE TABLE IF NOT EXISTS list_event (
    id       bigserial   PRIMARY KEY,
    user_id  integer     NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    mal_id   integer     NOT NULL,
    status   text,                          -- NULL: taken off the list
    score    smallint    NOT NULL DEFAULT 0,
    at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS list_event_user_title ON list_event (user_id, mal_id, at);

CREATE OR REPLACE FUNCTION record_list_event() RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        -- an account being deleted takes its list along: nothing to record
        IF NOT EXISTS (SELECT 1 FROM app_user WHERE id = OLD.user_id) THEN
            RETURN NULL;
        END IF;
        INSERT INTO list_event (user_id, mal_id, status, score)
             VALUES (OLD.user_id, OLD.mal_id, NULL, 0);
    ELSIF TG_OP = 'INSERT' OR OLD.status IS DISTINCT FROM NEW.status
          OR OLD.score IS DISTINCT FROM NEW.score THEN
        INSERT INTO list_event (user_id, mal_id, status, score)
             VALUES (NEW.user_id, NEW.mal_id, NEW.status, NEW.score);
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS list_entry_event ON list_entry;
CREATE TRIGGER list_entry_event AFTER INSERT OR UPDATE OR DELETE ON list_entry
    FOR EACH ROW EXECUTE FUNCTION record_list_event();
