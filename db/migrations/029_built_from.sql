-- The list a user's recommendations were last built from (refresh.fingerprint),
-- set by surfaces.build_all when it finishes. A sync rebuilds when the list
-- differs from it, not only when the sync itself changed something: a
-- rebuild that failed after the list was stored used to be forgotten, and the
-- lists stayed stale until the next change (audit 2026-10-23, ING-4).
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS built_from text;
