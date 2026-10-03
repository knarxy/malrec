#!/bin/sh
# Nightly pg_dump of the malrec database (scheduled by /etc/cron.d/malrec).
#
# Custom format, compressed: ~130 MB and ~25 s at 2026-09 sizes. Seven days
# are kept (~1 GB). Dumps contain MAL session tokens, so the directory is
# root-only. Skipped when the disk has less than MIN_FREE_GB free.
#
# Restore: docker exec -i malrec-db pg_restore -U malrec -d malrec --clean \
#            --if-exists < /opt/malrec/backups/malrec-YYYY-MM-DD.dump
set -eu
DIR=/opt/malrec/backups
KEEP_DAYS=7
MIN_FREE_GB=20
mkdir -p "$DIR" /opt/malrec/logs
chmod 700 "$DIR"
exec >>/opt/malrec/logs/backup.log 2>&1
# a failed or skipped backup tells the admin by mail (malrec.notify)
on_exit() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        tail -20 /opt/malrec/logs/backup.log | docker exec -i malrec-api malrec notify job-failed \
            --job backup >/dev/null 2>&1 || true
    fi
}
trap on_exit EXIT

free_gb=$(df -BG --output=avail "$DIR" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "$(date -Is) only ${free_gb} GB free; backup skipped"
    exit 1
fi

out="$DIR/malrec-$(date +%F).dump"
umask 077
docker exec malrec-db pg_dump -U malrec -Fc -Z 6 malrec > "$out.part"
mv "$out.part" "$out"
find "$DIR" -name 'malrec-*.dump' -mtime +$((KEEP_DAYS - 1)) -delete
echo "$(date -Is) $(du -h "$out" | cut -f1) $out; $(ls "$DIR"/malrec-*.dump | wc -l) kept, $(du -sh "$DIR" | cut -f1) total"
