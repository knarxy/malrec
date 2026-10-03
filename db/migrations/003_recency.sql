-- Recency weighting.
--
-- Taste drifts. A 10 given eight years ago says less about what the user wants
-- tonight than a 7 given last month, so every place that aggregates over the
-- user's ratings is weighted by an exponential decay on when the rating was
-- made. `finished_at` is preferred because it is when they actually watched it;
-- `updated_at` is the fallback and is always present.
--
-- The floor keeps old ratings contributing something rather than deleting the
-- user's history outright. Half-life is a parameter, tuned in eval/tune.py.

CREATE OR REPLACE FUNCTION rating_age_years(p_finished date, p_updated timestamptz)
RETURNS real AS $$
    SELECT greatest(
        extract(epoch FROM (now() - coalesce(p_finished::timestamptz, p_updated, now())))
            / 31557600.0,
        0.0
    )::real;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION recency_weight(
    p_finished   date,
    p_updated    timestamptz,
    p_half_life  real DEFAULT 0.5,
    p_floor      real DEFAULT 0.05
) RETURNS real AS $$
    SELECT CASE
        WHEN p_half_life IS NULL OR p_half_life <= 0 THEN 1.0::real
        ELSE (p_floor + (1 - p_floor)
              * pow(0.5, rating_age_years(p_finished, p_updated) / p_half_life))::real
    END;
$$ LANGUAGE sql STABLE;

-- Convenience view: every scored entry with its decay weight already applied.
CREATE OR REPLACE VIEW scored_entry AS
SELECT le.user_id,
       le.mal_id,
       le.score,
       le.status,
       le.finished_at,
       le.updated_at,
       rating_age_years(le.finished_at, le.updated_at) AS age_years,
       recency_weight(le.finished_at, le.updated_at)   AS w
  FROM list_entry le
 WHERE le.score > 0;

-- ------------------------------------------------ recency-aware rewrites --

-- The user's "personal mean" should also lean recent, otherwise a drifting
-- baseline makes every delta wrong.
CREATE OR REPLACE FUNCTION user_mean(p_user integer, p_half_life real DEFAULT 0.5)
RETURNS real AS $$
    SELECT (sum(score * recency_weight(finished_at, updated_at, p_half_life))
            / nullif(sum(recency_weight(finished_at, updated_at, p_half_life)), 0))::real
      FROM list_entry
     WHERE user_id = p_user AND score > 0;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION rec_affinity(
    p_user      integer,
    p_provider  text,
    p_shrink    real DEFAULT 1.5,
    p_half_life real DEFAULT 0.5
)
RETURNS TABLE (mal_id integer, affinity real) AS $$
    WITH m AS (SELECT user_mean(p_user, p_half_life) AS mu)
    SELECT e.src,
           (sum(e.weight * rw.w * (le.score - m.mu))
            / (sum(e.weight * rw.w) + p_shrink))::real
      FROM rec_edge e
      JOIN list_entry le ON le.mal_id = e.dst AND le.user_id = p_user AND le.score > 0
      CROSS JOIN LATERAL (
          SELECT recency_weight(le.finished_at, le.updated_at, p_half_life) AS w
      ) rw
      CROSS JOIN m
     WHERE e.provider = p_provider
     GROUP BY e.src;
$$ LANGUAGE sql STABLE;

-- For a franchise, "how much did they like it" is dominated by the most recent
-- strong opinion rather than an old one.
CREATE OR REPLACE FUNCTION franchise_affinity(p_user integer, p_half_life real DEFAULT 0.5)
RETURNS TABLE (mal_id integer, best_delta real, has_history boolean) AS $$
    WITH m AS (SELECT user_mean(p_user, p_half_life) AS mu),
    rated AS (
        SELECT f.franchise_id,
               max((le.score - m.mu) * recency_weight(le.finished_at, le.updated_at, p_half_life))::real AS best
          FROM list_entry le
          JOIN franchise f ON f.mal_id = le.mal_id
          CROSS JOIN m
         WHERE le.user_id = p_user AND le.score > 0
         GROUP BY f.franchise_id
    )
    SELECT f.mal_id, coalesce(r.best, 0)::real, r.franchise_id IS NOT NULL
      FROM franchise f LEFT JOIN rated r ON r.franchise_id = f.franchise_id;
$$ LANGUAGE sql STABLE;

CREATE OR REPLACE FUNCTION user_taste_vector(p_user integer, p_half_life real DEFAULT 0.5)
RETURNS vector AS $$
    WITH m AS (SELECT user_mean(p_user, p_half_life) AS mu),
    w AS (
        SELECT a.tag_vec,
               ((le.score - m.mu)
                * recency_weight(le.finished_at, le.updated_at, p_half_life))::float8 AS wt
          FROM list_entry le
          JOIN anime a ON a.mal_id = le.mal_id
          CROSS JOIN m
         WHERE le.user_id = p_user AND le.score > 0 AND a.tag_vec IS NOT NULL
    ),
    scaled AS (
        SELECT tag_vec * array_fill(wt, ARRAY[vector_dims(tag_vec)])::vector AS v,
               abs(wt) AS aw
          FROM w
    )
    SELECT CASE WHEN count(*) = 0 OR sum(aw) = 0 THEN NULL ELSE l2_normalize(sum(v)) END
      FROM scaled;
$$ LANGUAGE sql STABLE;

-- Drop the pre-recency single-argument forms so nothing calls them by accident.
DROP FUNCTION IF EXISTS user_taste_vector(integer);
DROP FUNCTION IF EXISTS franchise_affinity(integer);
DROP FUNCTION IF EXISTS rec_affinity(integer, text, real);
