-- A learned order per tab (recsys/ltr.py SURFACES); existing rankers are Safe Bets'.
ALTER TABLE order_ranker ADD COLUMN IF NOT EXISTS surface text NOT NULL DEFAULT 'safe_bets';
CREATE INDEX IF NOT EXISTS order_ranker_model_surface ON order_ranker (global_model_id, surface);
