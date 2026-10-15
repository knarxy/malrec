-- Series (one story: seasons, films, other versions, side stories and recaps)
-- next to franchises (a series plus its spin-offs and other settings). Both
-- are written by recsys/series.py refresh(), which replaces the SQL
-- connected components: one crossover short listed as a side story of two
-- shows used to merge them into one franchise.
CREATE TABLE IF NOT EXISTS series (
    mal_id     integer PRIMARY KEY,
    series_id  integer NOT NULL
);
CREATE INDEX IF NOT EXISTS series_sid_idx ON series (series_id);
DROP FUNCTION IF EXISTS refresh_franchises();
