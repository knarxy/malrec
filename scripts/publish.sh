#!/bin/sh
# Publish master to the public repository as a fresh tree: private files left
# out, every file scanned for strings that must never be public, then one
# commit on top of the public history and a push.
#
#   make publish MSG="What changed"
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

hits=$(grep -rnIiE -- "$SCAN" . --exclude-dir=.git | grep -vE -- "${ALLOW:-^\$}" || true)
if [ -n "$hits" ]; then
    echo "$hits" | cut -c1-160
    echo "== private strings found - nothing published"
    git checkout -q -- . && git clean -fdq
    exit 1
fi
git add -A
if git diff --cached --quiet; then echo "== nothing to publish"; exit 0; fi
git diff --cached --stat | tail -5
git -c user.name="$PUBLIC_NAME" -c user.email="$PUBLIC_EMAIL" commit -q \
    -m "$MSG" ${TRAILER:+-m "$TRAILER"}
git push -q origin HEAD
echo "== published: $(git log --oneline -1)"
