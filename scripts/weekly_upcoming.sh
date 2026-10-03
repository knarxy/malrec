#!/bin/sh
# Weekly "coming soon" refresh (scheduled by /etc/cron.d/malrec): newly
# announced seasons and films, then every user's Coming Soon list. The admin
# gets the result by mail, or the log's tail if it failed (malrec.notify).
set -u
cd /opt/malrec
mkdir -p logs
exec >>logs/upcoming.log 2>&1
echo "=== $(date -Is) weekly upcoming ==="
out=$(flock -w 14400 logs/refresh.lock \
      docker compose --profile lab run --rm -T lab malrec sync upcoming)
rc=$?
echo "$out"
if [ "$rc" -eq 0 ]; then
    printf '%s' "$out" | docker exec -i malrec-api malrec notify weekly \
        || echo "result mail failed (non-fatal)"
else
    tail -40 logs/upcoming.log | docker exec -i malrec-api malrec notify job-failed --job weekly \
        >/dev/null 2>&1 || true
fi
echo "=== $(date -Is) done (exit $rc) ==="
exit "$rc"
