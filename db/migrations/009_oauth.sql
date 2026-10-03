-- MyAnimeList OAuth2 (authorization code + PKCE).
--
-- Tokens never reach the browser. The browser holds an opaque random session
-- id in an HttpOnly cookie; the database stores only its SHA-256, so a leaked
-- table cannot be replayed as a cookie. Access and refresh tokens live here
-- because the server needs them to call MAL on the user's behalf.

-- In-flight authorisation requests. Single use and short-lived: the state
-- value is what stops a forged callback (CSRF) from logging someone in.
CREATE TABLE IF NOT EXISTS oauth_state (
    state          text PRIMARY KEY,
    code_verifier  text NOT NULL,
    return_to      text NOT NULL DEFAULT '/',
    created_at     timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS mal_session (
    token_hash     text PRIMARY KEY,
    app_user_id    integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    mal_user_id    integer,
    access_token   text NOT NULL,
    refresh_token  text NOT NULL,
    expires_at     timestamptz NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT now(),
    last_used_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mal_session_user_idx ON mal_session (app_user_id);
