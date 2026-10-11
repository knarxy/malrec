-- A learned Safe Bets order (src/malrec/recsys/ltr.py), one per population
-- model: its features are that model's predictions, so it is dropped with it.
CREATE TABLE IF NOT EXISTS order_ranker (
    id              serial PRIMARY KEY,
    created_at      timestamptz NOT NULL DEFAULT now(),
    global_model_id integer NOT NULL REFERENCES global_model(id) ON DELETE CASCADE,
    meta            jsonb NOT NULL,
    artifact        bytea NOT NULL
);
CREATE INDEX IF NOT EXISTS order_ranker_model ON order_ranker (global_model_id, id);
