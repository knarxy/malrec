-- The popularity prior's share of rec_candidate.bonus, kept apart so "Why
-- this?" can show both parts of the ordering adjustment: the long-memory mix
-- (bonus - pop_bonus) and the popularity prior for thin lists (pop_bonus).
ALTER TABLE rec_candidate ADD COLUMN IF NOT EXISTS pop_bonus real NOT NULL DEFAULT 0;
