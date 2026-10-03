#!/bin/sh
# Monthly population refresh (scheduled by /etc/cron.d/malrec).
#
#   1. rotate the CF sample: ~800 new public lists from recent discussion
#      replies, as many lists older than a year retired (~850 MAL requests at
#      the configured 1 req/s)
#   2. refit the population model, activated only if the reference profile's (MALREC_USER) temporal
#      holdout does not get worse (malrec population fit --gate-user)
#   3. if it was activated, retrain and rebuild every app user
#   4. log what in-app feedback says about the ranking weights, the
#      evaluation report (malrec report) and the baseline comparison
#
# DRY_RUN=1 checks every step without changing what users see: no sample rotation,
# the gate decides but activates nothing (the candidate is kept inactive), users are not rebuilt, the
# baseline comparison runs on 10 users. Output goes to refresh-dryrun.log so
# the admin panel's job status still shows the last real run.
#
# Only touches the malrec compose project. Output: /opt/malrec/logs/refresh.log
set -eu
cd /opt/malrec
mkdir -p logs
DRY_RUN=${DRY_RUN:-}
if [ -n "$DRY_RUN" ]; then exec >>logs/refresh-dryrun.log 2>&1; else exec >>logs/refresh.log 2>&1; fi
# the nightly jobs share this lock; wait for them rather than skip the month
exec 9>logs/refresh.lock
flock -w 7200 9 || { echo "$(date -Is) lock held for 2 hours; skipped"; exit 1; }

echo "=== $(date -Is) monthly refresh${DRY_RUN:+ (dry run)} ==="
lab() { docker compose --profile lab run --rm -T lab "$@"; }

if [ -n "$DRY_RUN" ]; then
    lab malrec sync cf-refresh --help >/dev/null && echo "cf-refresh: command ok (not run)"
    fit=$(lab sh -c 'malrec population fit --dry-run --gate-user "${MALREC_USER:?set MALREC_USER in .env}"')
else
    lab malrec sync cf-refresh
    fit=$(lab sh -c 'malrec population fit --gate-user "${MALREC_USER:?set MALREC_USER in .env}"')
fi
echo "$fit"
if [ -n "$DRY_RUN" ]; then
    echo "dry run: users not rebuilt ($(docker exec malrec-db psql -U malrec -d malrec -Atc \
         "SELECT count(*) FROM app_user") accounts would be)"
elif echo "$fit" | grep -q '"activated"'; then
    # the API picks up the newly active model on its next request
    for u in $(docker exec malrec-db psql -U malrec -d malrec -Atc \
               "SELECT mal_username FROM app_user ORDER BY id"); do
        # per user, so one broken account (e.g. no scores yet) cannot stop the rest
        if docker exec malrec-api malrec train --user "$u" >/dev/null \
           && docker exec malrec-api malrec build-all --user "$u" --limit 60 >/dev/null; then
            echo "rebuilt $u"
        else
            echo "rebuild of $u FAILED (continuing)"
        fi
    done
else
    echo "new model not activated; users unchanged"
fi
# what in-app feedback says about the ranking weights (suggestions only)
lab malrec learn-weights || echo "learn-weights failed (non-fatal)"
# evaluation report: gate vs baselines, range coverage, prospective test
lab malrec report || echo "report failed (non-fatal)"
# population-level comparison with baselines on the deployed configuration
lab python -u experiments/exp_audit.py --max-eval "$([ -n "$DRY_RUN" ] && echo 10 || echo 100)" \
    2>&1 | grep -v "^[0-9][0-9]:" || echo "exp_audit failed (non-fatal)"
echo "=== $(date -Is) done ==="
