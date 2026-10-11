#!/bin/sh
# Grow the population sample to TARGET public lists, one step at a time
# (malrec sync cf-expand: up to CHUNK new forum posters, then their lists -
# about an hour per step at RPS requests/s, default 1. Tried 2026-10-09:
# 1.5/s with eight lists in flight made MAL shed load (HTTP 307) within
# minutes; the forum/list endpoints are best left at 1/s, three in flight).
#
# Each step holds the shared MAL lock (logs/refresh.lock) only while it runs,
# so the nightly and weekly jobs get their turn between steps. Stops on its
# own at TARGET, when discovery runs dry, or when logs/expand.stop exists.
#
#   nohup scripts/expand_population.sh 30000 [CHUNK] [RPS] &      on the server
#
# Only touches the malrec compose project. Output: /opt/malrec/logs/expand.log
set -u
cd /opt/malrec
mkdir -p logs
TARGET=${1:?usage: expand_population.sh TARGET [CHUNK] [RPS]}
CHUNK=${2:-2000}
RPS=${3:-1}
exec >>logs/expand.log 2>&1
rm -f logs/expand.stop
lists() { docker exec malrec-db psql -U malrec -d malrec -Atc \
          "SELECT count(*) FROM cf_user WHERE state='done'"; }
echo "=== $(date -Is) expand to $TARGET lists (now $(lists)), $CHUNK per step at $RPS req/s ==="
while [ "$(lists)" -lt "$TARGET" ]; do
    [ -f logs/expand.stop ] && { echo "$(date -Is) stop file found"; break; }
    out=$(flock -w 14400 logs/refresh.lock \
          docker compose --profile lab run --rm -T lab malrec sync cf-expand --chunk "$CHUNK" --rps "$RPS")
    rc=$?
    echo "$(date -Is) step: $out"
    if [ "$rc" -ne 0 ]; then
        tail -40 logs/expand.log | docker exec -i malrec-api malrec notify job-failed \
            --job expand >/dev/null 2>&1 || true
        exit "$rc"
    fi
    # nothing new to find and nothing stored: the forum sources are exhausted
    if echo "$out" | grep -q '"discovered": 0' && echo "$out" | grep -q '"stored": 0'; then
        echo "$(date -Is) discovery exhausted"; break
    fi
    sleep 120           # let a waiting nightly job take the lock
done
echo "=== $(date -Is) done: $(lists) lists ==="
