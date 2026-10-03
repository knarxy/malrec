-- Access control: every account is approved by the admin before it gets
-- recommendations. Accounts that existed before this migration keep working.
-- (Replay-safe: the UPDATE only touches rows the ADD COLUMN just created.)
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS status text;
UPDATE app_user SET status = 'approved' WHERE status IS NULL;
ALTER TABLE app_user ALTER COLUMN status SET DEFAULT 'pending';
ALTER TABLE app_user ALTER COLUMN status SET NOT NULL;
ALTER TABLE app_user DROP CONSTRAINT IF EXISTS app_user_status_check;
ALTER TABLE app_user ADD CONSTRAINT app_user_status_check
    CHECK (status IN ('pending', 'approved', 'rejected', 'blocked'));
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS requested_at timestamptz;
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS approved_at timestamptz;
ALTER TABLE app_user ADD COLUMN IF NOT EXISTS last_login_at timestamptz;

-- admin actions, for the audit list in the admin panel
CREATE TABLE IF NOT EXISTS admin_log (
    id          bigserial PRIMARY KEY,
    admin       text        NOT NULL,
    action      text        NOT NULL,
    target      text,
    at          timestamptz NOT NULL DEFAULT now()
);
