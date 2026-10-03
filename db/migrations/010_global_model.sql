-- The fitted population model (item biases, item-item similarities, latent
-- factors) and the stacker that blends them. Fitted from the CF sample, used
-- for every user. Kept as versions so a refit can be compared and rolled back.
CREATE TABLE IF NOT EXISTS global_model (
    id          serial PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now(),
    active      boolean     NOT NULL DEFAULT false,
    meta        jsonb       NOT NULL DEFAULT '{}',
    artifact    bytea       NOT NULL
);
-- at most one active version
CREATE UNIQUE INDEX IF NOT EXISTS global_model_active_idx ON global_model (active) WHERE active;
