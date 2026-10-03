#!/bin/sh
# Nightly list sync (scheduled by /etc/cron.d/malrec): re-read every app
# user's MAL list and rebuild those that changed. Mails the admin only when
# something failed (malrec.notify).
set -u
cd /opt/malrec
mkdir -p logs
exec >>logs/users.log 2>&1
out=$(flock -w 14400 logs/refresh.lock \
      docker compose --profile lab run --rm -T lab malrec sync users)
rc=$?
echo "$out"
if [ "$rc" -eq 0 ]; then
    printf '%s' "$out" | docker exec -i malrec-api malrec notify nightly \
        || echo "failure mail failed (non-fatal)"
else
    tail -40 logs/users.log | docker exec -i malrec-api malrec notify job-failed --job list_sync \
        >/dev/null 2>&1 || true
fi
exit "$rc"
