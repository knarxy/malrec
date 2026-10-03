-- The disagreement penalty's share of rec_candidate.bonus (stored positive,
-- subtracted from the order key), so "Why this?" can show it on its own line.
ALTER TABLE rec_candidate ADD COLUMN IF NOT EXISTS risk real NOT NULL DEFAULT 0;
