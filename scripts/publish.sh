#!/bin/sh
# Publish master to the public repository as a fresh tree: private files left
# out, every file scanned for strings that must never be public, then one
# commit on top of the public history and a push.
#
#   make publish MSG="What changed"                  preview: what would change
#   PUBLISH_YES=1 make publish MSG="What changed"    publish it
#
# A push to a public repository cannot be taken back, so the first run only
# shows every file that would be added, changed or removed, and binary files
# by name (audit 2026-10-23, OPS-6). Besides SCAN, every file - binary ones
# too -, every path and the message are checked for local paths and mail
# addresses.
#
# Everything private lives in the untracked .publish.local (sourced here):
#   PUBLIC_DIR      a clone of the public repository
#   PUBLIC_NAME     author name for the public commit
#   PUBLIC_EMAIL    author address (e.g. a GitHub no-reply address)
#   EXCLUDE         space-separated paths that stay private
#   SCAN            extended regex of strings that must not appear
#   ALLOW           extended regex of matching lines that may stay
#   TRAILER         optional commit-message trailer
set -eu
GENERIC='/home/[^/ ]+/|/tmp/claude|-home-[a-z]|[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[a-z]{2,}'
# addresses at the reserved example domains (the tests use them) are no hit
EXAMPLE_ADDR='[A-Za-z0-9._%+-]+@example\.(com|org|net)([^A-Za-z0-9.-]|$)'
REPO=$(git rev-parse --show-toplevel)
cd "$REPO"
[ -f .publish.local ] || { echo "no .publish.local - see scripts/publish.sh"; exit 1; }
. ./.publish.local
[ -n "${MSG:-}" ] || { echo 'usage: make publish MSG="what changed"'; exit 1; }
[ -d "$PUBLIC_DIR/.git" ] || { echo "PUBLIC_DIR $PUBLIC_DIR is not a clone"; exit 1; }

cd "$PUBLIC_DIR"
git pull -q --ff-only
git ls-files -z | xargs -0 rm -f
find . -mindepth 1 -type d -not -path './.git' -not -path './.git/*' -empty -delete
(cd "$REPO" && git archive master) | tar -x
for f in $EXCLUDE; do rm -rf -- "$f"; done

undo() { git reset -q && git checkout -q -- . && git clean -fdq; }

# grep exits 0 on a match, 1 on none, 2 on an error: an error must stop the
# publish, never read as "nothing found". -a: binary files are read as text
set +e
found=$(grep -rnaiE --exclude-dir=.git -e "$SCAN" -e "$GENERIC" .)
rc=$?
set -e
if [ "$rc" -gt 1 ]; then
    echo "== the private-string scan failed (grep exit $rc) - nothing published"
    undo
    exit 1
fi
# SCAN is checked on the lines as they are; GENERIC without the example
# addresses and without its own definition above
scan=$(printf '%s\n' "$found" | grep -iE -e "$SCAN" || true)
generic=$(printf '%s\n' "$found" | grep -v '^\./scripts/publish\.sh:[0-9]*:GENERIC=' \
          | sed -E "s/$EXAMPLE_ADDR/\\2/g" | grep -iE -e "$GENERIC" || true)
names=$(find . -path ./.git -prune -o -print | grep -iE -e "$SCAN" -e "$GENERIC" || true)
msg=$(printf '%s\n' "$MSG" | grep -iE -e "$SCAN" -e "$GENERIC" || true)
hits=$(printf '%s\n%s\n%s\n%s\n' "$scan" "$generic" "$names" "$msg" \
       | grep -vE -e "${ALLOW:-^\$}" | grep -v '^$' || true)
if [ -n "$hits" ]; then
    echo "$hits" | cut -c1-160
    echo "== private strings found - nothing published"
    undo
    exit 1
fi
git add -A
if git diff --cached --quiet; then echo "== nothing to publish"; exit 0; fi
echo "== what the public repository would get:"
git diff --cached --name-status
bin=$(git diff --cached --numstat | awk '$1 == "-" {print $3}')
[ -n "$bin" ] && printf '== binary files among them (look at them):\n%s\n' "$bin"
if [ "${PUBLISH_YES:-}" != 1 ]; then
    undo
    echo "== preview only; publish with PUBLISH_YES=1 make publish MSG=\"...\""
    exit 0
fi
git -c user.name="$PUBLIC_NAME" -c user.email="$PUBLIC_EMAIL" commit -q \
    -m "$MSG" ${TRAILER:+-m "$TRAILER"}
git push -q origin HEAD
echo "== published: $(git log --oneline -1)"
