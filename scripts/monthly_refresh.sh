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
# Only touches the malrec compose project. Output: /opt/malrec/logs/refresh.log
set -eu
cd /opt/malrec
mkdir -p logs
exec >>logs/refresh.log 2>&1
exec 9>logs/refresh.lock
flock -n 9 || { echo "$(date -Is) previous refresh still running; skipped"; exit 0; }

echo "=== $(date -Is) monthly refresh ==="
lab() { docker compose --profile lab run --rm -T lab "$@"; }

lab malrec sync cf-refresh
fit=$(lab sh -c 'malrec population fit --gate-user "${MALREC_USER:?set MALREC_USER in .env}"')
echo "$fit"
if echo "$fit" | grep -q '"activated"'; then
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
lab python -u experiments/exp_audit.py --max-eval 100 2>&1 | grep -v "^[0-9][0-9]:" \
    || echo "exp_audit failed (non-fatal)"
echo "=== $(date -Is) done ==="
