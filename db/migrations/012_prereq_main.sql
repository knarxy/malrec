-- Only main-format prequels (tv / movie / ona) are prerequisites.
--
-- The chain is still walked through OVAs and specials, so a later season
-- stays gated on the ones before it, but side works are not required
-- themselves: MAL lists e.g. "One Punch Man: Road to Hero" (OVA) as the
-- prequel of One Punch Man, which hid the main series from anyone who had not
-- seen the OVA. (Replaces the definition in 002_logic.sql, which cannot use
-- format_class() because that is defined in 005.)
CREATE OR REPLACE FUNCTION prereqs_satisfied(p_user integer, target integer)
RETURNS boolean AS $$
    SELECT NOT EXISTS (
        SELECT 1
          FROM prerequisites(target) p
          JOIN anime a ON a.mal_id = p.mal_id
          LEFT JOIN list_entry le
                 ON le.user_id = p_user AND le.mal_id = p.mal_id
         WHERE le.status IS DISTINCT FROM 'completed'
           AND format_class(a.media_type) = 'main'
    );
$$ LANGUAGE sql STABLE;
