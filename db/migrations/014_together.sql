-- Watch together: two approved accounts, both of whom agreed.
CREATE TABLE IF NOT EXISTS together_pair (
    id          bigserial PRIMARY KEY,
    inviter     integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    invitee     integer NOT NULL REFERENCES app_user (id) ON DELETE CASCADE,
    status      text    NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    accepted_at timestamptz,
    CHECK (inviter <> invitee)
);
-- one relationship per pair of people, whoever invited whom
CREATE UNIQUE INDEX IF NOT EXISTS together_pair_uniq
    ON together_pair (least(inviter, invitee), greatest(inviter, invitee));
