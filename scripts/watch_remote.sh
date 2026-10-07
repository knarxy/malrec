#!/bin/sh
# Watchdog for a long job on the remote host (experiments, dry runs).
#
#   scripts/watch_remote.sh LOG DONE_REGEX KILL_AT_EPOCH WATCH_MIN
#
# Polls LOG on $DEV (ssh target, from .make.local) and prints one line when:
# DONE_REGEX appears (DONE), a failure signature appears (FAILED), no lab
# container runs any more (ENDED), the absolute deadline KILL_AT_EPOCH passes
# (TIMEOUT - the lab container is stopped), or WATCH_MIN minutes of watching
# pass (STILL RUNNING - start it again to keep watching).
LOG=$1; DONE=$2; KILL_AT=$3; WATCH=$4
cd "$(dirname "$0")/.." || exit 1
DEV=${DEV:-$(sed -n 's/^DEV *:*= *//p' .make.local 2>/dev/null)}
[ -n "$DEV" ] || { echo "set DEV (ssh target) or DEV := ... in .make.local"; exit 1; }
stop_watch=$(( $(date +%s) + WATCH * 60 ))
while :; do
  out=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=15 "$DEV" \
        "grep -E -c '$DONE' $LOG 2>/dev/null; grep -E -c 'Traceback|Killed|MemoryError' $LOG 2>/dev/null; docker ps -q --filter name=malrec-lab-run | wc -l" 2>/dev/null | tr '\n' ' ')
  set -- $out
  done_n=${1:-0}; err_n=${2:-0}; running=${3:-1}
  now=$(date +%s)
  if [ "$done_n" -gt 0 ] 2>/dev/null; then echo "DONE: $LOG"; exit 0; fi
  if [ "$err_n" -gt 0 ] 2>/dev/null; then echo "FAILED (error in $LOG)"; exit 1; fi
  if [ "$running" = "0" ]; then echo "ENDED without the done marker: $LOG"; exit 1; fi
  if [ "$now" -ge "$KILL_AT" ]; then
    timeout 60 ssh -o BatchMode=yes "$DEV" 'for c in $(docker ps -q --filter name=malrec-lab-run); do docker stop -t 10 $c; done' >/dev/null 2>&1
    echo "TIMEOUT: lab container stopped ($LOG)"; exit 2
  fi
  if [ "$now" -ge "$stop_watch" ]; then echo "STILL RUNNING after ${WATCH} min: $LOG"; exit 3; fi
  sleep 30
done
