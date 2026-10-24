#!/bin/sh
# The test suite on the server, against a copy of last night's backup
# (make dev-test). The tests create and delete accounts, feedback and sample
# rows, so they must never run against the live database (audit 2026-10-23,
# P1-16); on a copy of it the tests that need a synced user, the catalogue
# and the sample run too, and the new migrations are applied to real data
# on the way. A test that cannot reach the copy fails (MALREC_REQUIRE_DB).
#
# Only the malrec compose project is touched; the copy is dropped at the end.
set -eu
cd /opt/malrec
SCRATCH=malrec_test
latest=$(ls -1t backups/malrec-*.dump 2>/dev/null | head -1 || true)
if [ -z "$latest" ]; then
    echo "no backup to copy - nothing tested"
    exit 1
fi
echo "== tests against a copy of $latest"
docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"
docker exec malrec-db createdb -U malrec "$SCRATCH"
trap 'docker exec malrec-db dropdb -U malrec --if-exists "$SCRATCH"' EXIT
docker exec -i malrec-db pg_restore -U malrec -d "$SCRATCH" --no-owner < "$latest" \
    > /dev/null 2>&1 || true
docker compose --profile lab run --rm -T lab sh -c '
    export DATABASE_URL="${DATABASE_URL%/*}/'"$SCRATCH"'" MALREC_REQUIRE_DB=1
    pip install -q --no-warn-script-location --require-hashes -r requirements-dev.lock &&
    malrec init > /dev/null &&
    python -m pytest -q -p no:cacheprovider tests'
