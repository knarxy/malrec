-- Format classes.
--
-- A "sequel" on MAL covers both "season 2" and "a 4-minute joke short", which
-- made the continuations surface fill up with specials. Splitting formats into
-- main works and side material keeps the two apart everywhere at once.
--
--   main  tv, movie, ona     - a full work, watchable on its own terms
--   side  ova, special, tv_special - supplementary to a main work
--   noise music, cm, pv      - never recommendable

CREATE OR REPLACE FUNCTION format_class(p_media_type text)
RETURNS text AS $$
    SELECT CASE lower(coalesce(p_media_type, ''))
        WHEN 'tv'         THEN 'main'
        WHEN 'movie'      THEN 'main'
        WHEN 'ona'        THEN 'main'
        WHEN 'ova'        THEN 'side'
        WHEN 'special'    THEN 'side'
        WHEN 'tv_special' THEN 'side'
        WHEN 'music'      THEN 'noise'
        WHEN 'cm'         THEN 'noise'
        WHEN 'pv'         THEN 'noise'
        ELSE 'main'
    END;
$$ LANGUAGE sql IMMUTABLE;

CREATE INDEX IF NOT EXISTS anime_format_class_idx ON anime (format_class(media_type));

-- Replace the 'noise' exclusion in eligible_candidates with the shared
-- definition so there is only one list of junk formats in the codebase.
-- The return type gains a column, so the old signature must go first.
DROP FUNCTION IF EXISTS eligible_candidates(integer, integer, boolean);
CREATE OR REPLACE FUNCTION eligible_candidates(
    p_user            integer,
    p_min_scorers     integer DEFAULT 2000,
    p_allow_nsfw      boolean DEFAULT false
)
RETURNS TABLE (
    mal_id           integer,
    franchise_id     integer,
    title            text,
    mal_mean         real,
    mal_popularity   integer,
    known_franchise  boolean,
    format_class     text
) AS $$
    WITH user_franchises AS (
        SELECT DISTINCT f.franchise_id
          FROM list_entry le JOIN franchise f ON f.mal_id = le.mal_id
         WHERE le.user_id = p_user
    )
    SELECT a.mal_id,
           coalesce(f.franchise_id, a.mal_id),
           a.title,
           a.mal_mean,
           a.mal_popularity,
           uf.franchise_id IS NOT NULL,
           format_class(a.media_type)
      FROM anime a
      LEFT JOIN franchise f ON f.mal_id = a.mal_id
      LEFT JOIN user_franchises uf ON uf.franchise_id = f.franchise_id
     WHERE NOT EXISTS (SELECT 1 FROM list_entry le
                        WHERE le.user_id = p_user AND le.mal_id = a.mal_id)
       AND NOT EXISTS (SELECT 1 FROM feedback fb
                        WHERE fb.user_id = p_user AND fb.mal_id = a.mal_id
                          AND fb.action IN ('not_interested', 'hidden', 'seen_it'))
       AND (p_allow_nsfw OR a.nsfw IS NULL OR a.nsfw = 'white')
       AND format_class(a.media_type) <> 'noise'
       AND coalesce(a.mal_num_scoring_users, 0) >= p_min_scorers
       AND coalesce(a.status, '') <> 'not_yet_aired'
       AND prereqs_satisfied(p_user, a.mal_id);
$$ LANGUAGE sql STABLE;
