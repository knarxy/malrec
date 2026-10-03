-- Marks an anime whose detail payload (recommendations + relations) has been
-- pulled. Needed as its own column because an anime with zero recommendations
-- writes no rec_edge rows, and would otherwise look unfetched forever.
ALTER TABLE anime ADD COLUMN IF NOT EXISTS graph_fetched_at timestamptz;
CREATE INDEX IF NOT EXISTS anime_graph_pending_idx
    ON anime (mal_popularity) WHERE graph_fetched_at IS NULL;
