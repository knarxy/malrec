-- Fix target leakage in the franchise features.
--
-- franchise_affinity() aggregated the user's ratings over a franchise and then
-- joined that back onto every anime in the franchise - including the rated
-- anime itself. So when the model trained on an anime the user had scored, the
-- feature already contained that anime's own score.
--
-- Two consequences, both measured:
--   * offline metrics were inflated, because the target was partly an input;
--   * has_history was literally 1.0 for every training row (a rated anime is
--     always in a franchise containing a rated anime - itself) and ~0.03 for
--     real candidates, so the model learned nothing from it and the constant
--     got absorbed into the intercept.
--
-- Excluding the anime itself makes the training-time feature mean the same
-- thing as the inference-time one: "how do you rate the REST of this
-- franchise".

CREATE OR REPLACE FUNCTION franchise_affinity(p_user integer, p_half_life real DEFAULT 0.5)
RETURNS TABLE (mal_id integer, best_delta real, has_history boolean) AS $$
    WITH m AS (SELECT user_mean(p_user, p_half_life) AS mu),
    rated AS (
        SELECT f.franchise_id,
               le.mal_id AS rated_id,
               ((le.score - m.mu)
                * recency_weight(le.finished_at, le.updated_at, p_half_life))::real AS d
          FROM list_entry le
          JOIN franchise f ON f.mal_id = le.mal_id
          CROSS JOIN m
         WHERE le.user_id = p_user AND le.score > 0
    )
    SELECT f.mal_id,
           coalesce(max(r.d) FILTER (WHERE r.rated_id <> f.mal_id), 0)::real,
           coalesce(bool_or(r.rated_id <> f.mal_id), false)
      FROM franchise f
      LEFT JOIN rated r ON r.franchise_id = f.franchise_id
     GROUP BY f.mal_id;
$$ LANGUAGE sql STABLE;
