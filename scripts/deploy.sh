#!/bin/sh
# Safe deployment on the server (run by `make dev-up` after syncing code):
#
#   1. migrations are run against a copy of last night's backup in a scratch
#      database - a migration that would fail on real data stops the deploy
#      before anything changes
#   2. a fresh dump of the live database (backups/predeploy-*.dump, the
#      newest three kept): the restore point if a migration that passed on
#      the copy still does harm, without losing the day since the backup
#   3. the running images are kept as a rollback point
#   4. api, app and worker are rebuilt and restarted
#   5. if that fails, or they are not all healthy within WAIT seconds, the
#      previous images are started again and the deploy fails
#   6. a tab whose learned order is on but has no ranker for the current
#      settings is reported (and mailed): it serves its hand-built order
#      until `malrec ranker train --if-missing` has run
#
# The cron jobs run in the lab container on the synced tree (/work), not on
# the images: after a rollback they keep running the new code until the
# previous code is synced again. Migrations are not undone either (they are
# additive; the predeploy dump is the way back).
#
# Only the malrec compose project is touched.
set -eu
cd /opt/malrec
SCRATCH=malrec_predeploy
WAIT=180
LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

# the lab image runs the migration check and the scheduled jobs: same code deps
docker compose --profile lab build -q lab

latest=$(ls -1t backups/malrec-*.dump 2>/dev/null | head -1 || true)
if [ -n "$latest" ]; then
    echo "== migration check against $latest"
    if [ -n "$(find "$latest" -mmin +2880)" ]; then
        echo "   (note: that backup is more than two days old)"
    fi
    docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"
    docker exec malrec-db createdb -U malrec "$SCRATCH"
    # pg_restore reports harmless errors (object owners, the extension's
    # comment) with a non-zero exit, so the restore is judged by its result:
    # a copy without accounts or sample would let any migration pass
    docker exec -i malrec-db pg_restore -U malrec -d "$SCRATCH" --no-owner < "$latest" \
        > "$LOG" 2>&1 || true
    restored=$(docker exec malrec-db psql -U malrec -d "$SCRATCH" -Atc \
        "SELECT (SELECT count(*) FROM app_user) || ' ' || (SELECT count(*) FROM cf_rating)" \
        2>/dev/null || echo "0 0")
    if [ "${restored% *}" -eq 0 ] || [ "${restored#* }" -eq 0 ]; then
        tail -20 "$LOG"
        docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"
        echo "== the backup did not restore (accounts, ratings: $restored) - nothing deployed"
        exit 1
    fi
    if ! docker compose --profile lab run --rm -T lab sh -c \
            'DATABASE_URL="${DATABASE_URL%/*}/'"$SCRATCH"'" malrec init' > "$LOG" 2>&1; then
        tail -20 "$LOG"
        docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"
        echo "== migrations FAIL on a copy of the live data - nothing deployed"
        exit 1
    fi
    docker exec malrec-db dropdb -U malrec "$SCRATCH"
    echo "== migrations OK"
else
    echo "== no backup found - migration check skipped"
fi

dump="backups/predeploy-$(date +%Y%m%d-%H%M%S).dump"
echo "== dumping the live database to $dump"
umask 077
if ! docker exec malrec-db pg_dump -U malrec -Fc -Z 6 malrec > "$dump.part"; then
    rm -f "$dump.part"
    echo "== the predeploy dump failed - nothing deployed"
    exit 1
fi
mv "$dump.part" "$dump"
ls -1t backups/predeploy-*.dump | tail -n +4 | xargs -r rm -f

echo "== keeping the running images as rollback"
for s in api app worker; do
    docker image inspect "malrec-$s" > /dev/null 2>&1 && docker tag "malrec-$s" "malrec-$s:rollback"
done

rollback() {
    echo "== $1 - rolling back"
    docker logs --tail 30 malrec-api 2>&1 | tail -30
    for s in api app worker; do
        docker image inspect "malrec-$s:rollback" > /dev/null 2>&1 &&
            docker tag "malrec-$s:rollback" "malrec-$s"
    done
    docker compose up -d --no-build api app worker || true
    echo "== rolled back to the previous images; the database keeps the new migrations"
    echo "   ($dump is from just before them) and the cron jobs run the synced tree"
    exit 1
}

echo "== building and starting"
# up waits for the API's health (the worker depends on it) and fails when it
# never comes: that is the most common broken deploy, so it rolls back too
docker compose up -d --build api app worker || rollback "the new containers did not start"

healthy() {
    for c in malrec-api malrec-app malrec-worker; do
        [ "$(docker inspect -f '{{.State.Health.Status}}' "$c" 2>/dev/null)" = healthy ] ||
            return 1
    done
}
i=0
while [ $i -lt $WAIT ]; do
    if healthy; then
        echo "== healthy after ${i}s"
        # tidy up: malrec's own superseded images (by project label - never
        # other stacks'), and the build cache back to its recent 2 GB
        old=$(docker images -f dangling=true -f label=com.docker.compose.project=malrec -q)
        [ -n "$old" ] && docker rmi $old > /dev/null 2>&1 || true
        docker builder prune -f --keep-storage 2GB > /dev/null 2>&1 || true
        if ! docker exec malrec-api malrec ranker check > "$LOG" 2>&1; then
            cat "$LOG"
            docker exec -i malrec-api malrec notify job-failed --job deploy < "$LOG" \
                > /dev/null 2>&1 || true
        fi
        exit 0
    fi
    sleep 5; i=$((i + 5))
done
rollback "NOT healthy after ${WAIT}s"
