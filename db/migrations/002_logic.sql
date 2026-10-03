-- Recommendation logic that belongs in the database: franchise components,
-- prerequisite chains, and the eligibility view the ranker retrieves from.

-- ------------------------------------------------- franchise components --
-- Iterative min-label propagation over `relation`. Converges in a handful of
-- passes because franchise graphs are shallow, and it keeps the whole
-- connected-component computation server-side.

CREATE OR REPLACE FUNCTION refresh_franchises() RETURNS integer AS $$
DECLARE
    changed bigint := 1;
    passes  integer := 0;
BEGIN
    CREATE TEMP TABLE IF NOT EXISTS _fr (mal_id integer PRIMARY KEY, fid integer NOT NULL)
        ON COMMIT DROP;
    TRUNCATE _fr;

    -- every node that appears anywhere starts as its own component
    INSERT INTO _fr (mal_id, fid)
    SELECT mal_id, mal_id FROM anime
    UNION
    SELECT src, src FROM relation
    UNION
    SELECT dst, dst FROM relation
    ON CONFLICT (mal_id) DO NOTHING;

    -- only structural relations merge a franchise; 'other' and 'character'
    -- links would chain unrelated shows together.
    WHILE changed > 0 AND passes < 40 LOOP
        WITH edges AS (
            SELECT src AS a, dst AS b FROM relation
             WHERE relation_type IN ('sequel', 'prequel', 'side_story', 'parent_story',
                                     'alternative_version', 'alternative_setting',
                                     'full_story', 'summary', 'spin_off')
            UNION ALL
            SELECT dst, src FROM relation
             WHERE relation_type IN ('sequel', 'prequel', 'side_story', 'parent_story',
                                     'alternative_version', 'alternative_setting',
                                     'full_story', 'summary', 'spin_off')
        ),
        prop AS (
            SELECT e.a AS mal_id, min(f.fid) AS fid
              FROM edges e JOIN _fr f ON f.mal_id = e.b
             GROUP BY e.a
        )
        UPDATE _fr t SET fid = LEAST(t.fid, p.fid)
          FROM prop p
         WHERE t.mal_id = p.mal_id AND p.fid < t.fid;
        GET DIAGNOSTICS changed = ROW_COUNT;
        passes := passes + 1;
    END LOOP;

    TRUNCATE franchise;
    INSERT INTO franchise (mal_id, franchise_id) SELECT mal_id, fid FROM _fr;
    RETURN passes;
END;
$$ LANGUAGE plpgsql;

-- -------------------------------------------------- prerequisite chains --
-- Walks prequel edges transitively. This is what stops the recommender from
-- offering season 5 of a show whose season 1 the user has never touched, and
-- it works even when the user has no entry from that franchise at all.

CREATE OR REPLACE FUNCTION prerequisites(target integer)
RETURNS TABLE (mal_id integer, depth integer) AS $$
    WITH RECURSIVE chain AS (
        SELECT r.dst AS mal_id, 1 AS depth
          FROM relation r
         WHERE r.src = target AND r.relation_type = 'prequel'
        UNION
        SELECT r.dst, c.depth + 1
          FROM chain c
          JOIN relation r ON r.src = c.mal_id AND r.relation_type = 'prequel'
         WHERE c.depth < 12
    )
    SELECT chain.mal_id, min(chain.depth)::integer FROM chain GROUP BY chain.mal_id;
$$ LANGUAGE sql STABLE;

-- True when every prequel in the chain is completed by this user.
CREATE OR REPLACE FUNCTION prereqs_satisfied(p_user integer, target integer)
RETURNS boolean AS $$
    SELECT NOT EXISTS (
        SELECT 1
          FROM prerequisites(target) p
          LEFT JOIN list_entry le
                 ON le.user_id = p_user AND le.mal_id = p.mal_id
         WHERE le.status IS DISTINCT FROM 'completed'
    );
$$ LANGUAGE sql STABLE;

-- --------------------------------------------------------- taste vectors --
-- A user's taste as a weighted average of the tag vectors of what they rated,
-- centred on their personal mean so "loved" pulls and "disliked" pushes.

CREATE OR REPLACE FUNCTION user_taste_vector(p_user integer)
RETURNS vector AS $$
    WITH m AS (
        SELECT avg(score)::real AS mu
          FROM list_entry WHERE user_id = p_user AND score > 0
    ),
    w AS (
        SELECT a.tag_vec, (le.score - m.mu)::float8 AS wt
          FROM list_entry le
          JOIN anime a ON a.mal_id = le.mal_id
          CROSS JOIN m
         WHERE le.user_id = p_user AND le.score > 0 AND a.tag_vec IS NOT NULL
    ),
    -- pgvector 0.8 has no vector*scalar operator, so scale elementwise against
    -- a constant vector built from the weight.
    scaled AS (
        SELECT tag_vec * array_fill(wt, ARRAY[vector_dims(tag_vec)])::vector AS v,
               abs(wt) AS aw
          FROM w
    )
    -- L2-normalised, which is what the <=> cosine operator wants anyway
    SELECT CASE WHEN count(*) = 0 OR sum(aw) = 0 THEN NULL
                ELSE l2_normalize(sum(v))
           END
      FROM scaled;
$$ LANGUAGE sql STABLE;

-- ------------------------------------------------------- eligibility view --
-- One place that defines "could this ever be recommended to this user".
-- Parameterised by user through the function below rather than a plain view so
-- the planner can still use indexes.

-- A later migration widens the return type, and migrations are replayed in
-- order on every `malrec init`, so this must drop before it creates.
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
    known_franchise  boolean
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
           uf.franchise_id IS NOT NULL
      FROM anime a
      LEFT JOIN franchise f ON f.mal_id = a.mal_id
      LEFT JOIN user_franchises uf ON uf.franchise_id = f.franchise_id
     WHERE NOT EXISTS (SELECT 1 FROM list_entry le
                        WHERE le.user_id = p_user AND le.mal_id = a.mal_id)
       AND NOT EXISTS (SELECT 1 FROM feedback fb
                        WHERE fb.user_id = p_user AND fb.mal_id = a.mal_id
                          AND fb.action IN ('not_interested', 'hidden', 'seen_it'))
       AND (p_allow_nsfw OR a.nsfw IS NULL OR a.nsfw = 'white')
       AND coalesce(a.media_type, '') NOT IN ('music', 'cm', 'pv')
       AND coalesce(a.mal_num_scoring_users, 0) >= p_min_scorers
       AND coalesce(a.status, '') <> 'not_yet_aired'
       AND prereqs_satisfied(p_user, a.mal_id);
$$ LANGUAGE sql STABLE;

-- --------------------------------------------------------------- helpers --

-- Affinity of one anime to a user, over a chosen recommendation provider.
-- Score-weighted and shrunk so a single strong edge cannot dominate.
CREATE OR REPLACE FUNCTION rec_affinity(p_user integer, p_provider text, p_shrink real DEFAULT 1.5)
RETURNS TABLE (mal_id integer, affinity real) AS $$
    WITH m AS (
        SELECT avg(score)::real AS mu FROM list_entry WHERE user_id = p_user AND score > 0
    )
    SELECT e.src,
           (sum(e.weight * (le.score - m.mu)) / (sum(e.weight) + p_shrink))::real
      FROM rec_edge e
      JOIN list_entry le ON le.mal_id = e.dst AND le.user_id = p_user AND le.score > 0
      CROSS JOIN m
     WHERE e.provider = p_provider
     GROUP BY e.src;
$$ LANGUAGE sql STABLE;

-- Best score the user gave anywhere in the same franchise.
CREATE OR REPLACE FUNCTION franchise_affinity(p_user integer)
RETURNS TABLE (mal_id integer, best_delta real, has_history boolean) AS $$
    WITH m AS (
        SELECT avg(score)::real AS mu FROM list_entry WHERE user_id = p_user AND score > 0
    ),
    rated AS (
        SELECT f.franchise_id, max(le.score - m.mu)::real AS best
          FROM list_entry le
          JOIN franchise f ON f.mal_id = le.mal_id
          CROSS JOIN m
         WHERE le.user_id = p_user AND le.score > 0
         GROUP BY f.franchise_id
    )
    SELECT f.mal_id, coalesce(r.best, 0)::real, r.franchise_id IS NOT NULL
      FROM franchise f LEFT JOIN rated r ON r.franchise_id = f.franchise_id;
$$ LANGUAGE sql STABLE;
