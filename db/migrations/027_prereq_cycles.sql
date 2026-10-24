-- A title is never its own prerequisite (audit 2026-10-23, ING-16).
--
-- `relation` holds MAL's and AniList's edges, and a few contradict each
-- other: two titles each listed as the other's prequel (Ao Haru Ride and
-- its OVA, the Monogatari specials). The walk then came back to where it
-- started, prereqs_satisfied asked for the title itself to be completed -
-- which no unlisted title is - and 61 titles with 300+ scorers, Danganronpa
-- and Ao Haru Ride among them, could never be recommended. The walk now
-- stops at the target. Two main-format titles that name each other still
-- wait for one another: the data says each comes first. (Replaces the
-- definition in 002_logic.sql; recsys/pool.py walks the same way.)
CREATE OR REPLACE FUNCTION prerequisites(target integer)
RETURNS TABLE (mal_id integer, depth integer) AS $$
    WITH RECURSIVE chain AS (
        SELECT r.dst AS mal_id, 1 AS depth
          FROM relation r
         WHERE r.src = target AND r.relation_type = 'prequel' AND r.dst <> target
        UNION
        SELECT r.dst, c.depth + 1
          FROM chain c
          JOIN relation r ON r.src = c.mal_id AND r.relation_type = 'prequel'
         WHERE c.depth < 12 AND r.dst <> target
    )
    SELECT chain.mal_id, min(chain.depth)::integer FROM chain GROUP BY chain.mal_id;
$$ LANGUAGE sql STABLE;
