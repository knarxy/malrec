-- MAL flips a title from not_yet_aired to currently_airing days after its
-- premiere, so the status alone hid shows that had started (2026-10-03, three
-- days into the fall season: 24 started fall shows still read not_yet_aired
-- and This Season showed one title). A start date in the past counts as aired.
-- Same signature and return type as 005, so a plain replace suffices.
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
       AND (coalesce(a.status, '') <> 'not_yet_aired' OR a.start_date <= current_date)
       AND prereqs_satisfied(p_user, a.mal_id);
$$ LANGUAGE sql STABLE;
