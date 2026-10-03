-- The account's MyAnimeList profile picture (a cdn.myanimelist.net URL), read
-- from /users/@me at sign-in and on every list sync made with the user's token.
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS picture_url text;
