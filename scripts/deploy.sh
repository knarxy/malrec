#!/bin/sh
# Safe deployment on the server (run by `make dev-up` after syncing code):
#
#   1. migrations are replayed against a copy of last night's backup in a
#      scratch database - a migration that would fail on real data stops the
#      deploy before anything changes
#   2. the running images are kept as a rollback point
#   3. api and app are rebuilt and restarted
#   4. if they are not healthy within two minutes, the previous images are
#      started again and the deploy fails
#
# Only the malrec compose project is touched.
set -eu
cd /opt/malrec
SCRATCH=malrec_predeploy
WAIT=120

# the lab image runs the migration check and the scheduled jobs: same code deps
docker compose --profile lab build -q lab

latest=$(ls -1t backups/malrec-*.dump 2>/dev/null | head -1 || true)
if [ -n "$latest" ]; then
    echo "== migration check against $latest"
    docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"
    docker exec malrec-db createdb -U malrec "$SCRATCH"
    docker exec -i malrec-db pg_restore -U malrec -d "$SCRATCH" --no-owner < "$latest" \
        > /dev/null 2>&1 || true
    if ! docker compose --profile lab run --rm -T lab sh -c \
            'DATABASE_URL="${DATABASE_URL%/*}/'"$SCRATCH"'" malrec init' > /tmp/malrec-migrate.log 2>&1; then
        tail -20 /tmp/malrec-migrate.log
        docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"
        echo "== migrations FAIL on a copy of the live data - nothing deployed"
        exit 1
    fi
    docker exec malrec-db dropdb -U malrec "$SCRATCH"
    echo "== migrations OK"
else
    echo "== no backup found - migration check skipped"
fi

echo "== keeping the running images as rollback"
for s in api app; do
    docker image inspect "malrec-$s" > /dev/null 2>&1 && docker tag "malrec-$s" "malrec-$s:rollback"
done

echo "== building and starting"
docker compose up -d --build api app

healthy() {
    [ "$(docker inspect -f '{{.State.Health.Status}}' malrec-api 2>/dev/null)" = healthy ] &&
    [ "$(docker inspect -f '{{.State.Health.Status}}' malrec-app 2>/dev/null)" = healthy ]
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
        exit 0
    fi
    sleep 5; i=$((i + 5))
done

echo "== NOT healthy after ${WAIT}s - rolling back"
docker logs --tail 30 malrec-api 2>&1 | tail -30
for s in api app; do
    docker image inspect "malrec-$s:rollback" > /dev/null 2>&1 && docker tag "malrec-$s:rollback" "malrec-$s"
done
docker compose up -d --no-build api app
exit 1
