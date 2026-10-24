-- Accounts are MAL users, not MAL names (audit 2026-10-23, SEC-2).
--
-- A MAL name can change, and a freed one can be taken by someone else; the
-- numeric id MAL gives an account cannot. The sign-in finds the account by
-- that id and keeps the name current (auth.account_for). Existing accounts
-- get the id their newest session recorded.
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS mal_user_id integer;
CREATE UNIQUE INDEX IF NOT EXISTS app_user_mal_user_id_key ON app_user (mal_user_id);
WITH newest AS (
    SELECT DISTINCT ON (s.mal_user_id) s.mal_user_id, s.app_user_id
      FROM mal_session s
     WHERE s.mal_user_id IS NOT NULL
     ORDER BY s.mal_user_id, s.created_at DESC
)
UPDATE app_user u SET mal_user_id = n.mal_user_id
  FROM newest n
 WHERE u.id = n.app_user_id AND u.mal_user_id IS NULL
   AND NOT EXISTS (SELECT 1 FROM app_user o WHERE o.mal_user_id = n.mal_user_id);

-- The sign-in is bound to the browser that started it (audit 2026-10-23,
-- SEC-3): the hash of a nonce the browser holds in a short-lived cookie.
ALTER TABLE oauth_state ADD COLUMN IF NOT EXISTS browser text;
