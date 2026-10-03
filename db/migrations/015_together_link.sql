-- Watch together by link: single-use, expiring. Only a hash of the token is
-- stored, so a database copy cannot be used to accept anyone's invitation.
CREATE TABLE IF NOT EXISTS together_link (
    id          bigserial PRIMARY KEY,
    inviter     integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    token_hash  bytea   NOT NULL UNIQUE,
    created_at  timestamptz NOT NULL DEFAULT now(),
    expires_at  timestamptz NOT NULL,
    used_by     integer REFERENCES app_user (id) ON DELETE SET NULL,
    used_at     timestamptz
);
CREATE INDEX IF NOT EXISTS together_link_inviter_idx ON together_link (inviter);
