#!/usr/bin/env bash
# Code-quality and security audit of the repository. Prints findings and
# writes them to audit/ (untracked: reports can quote what they found); it
# never fails, the findings are for reading, not a gate.
#
#   make audit-install     the locked Python tools (requirements-audit.lock)
#   make audit             everything below
#   scripts/audit.sh ruff bandit     only these steps
#
# semgrep and gitleaks run as pinned containers (docker, else podman) that
# are removed when they finish. pip-audit and npm audit send package names
# and versions to PyPI / OSV / npm; semgrep fetches its public rule packs and
# sends no metrics. Nothing else leaves the machine.
set -uo pipefail
cd "$(dirname "$0")/.."

B=.venv/bin
OUT=audit
SEMGREP_IMAGE=docker.io/semgrep/semgrep:1.97.0
GITLEAKS_IMAGE=docker.io/zricethezav/gitleaks:v8.21.2
CTR=$(docker info >/dev/null 2>&1 && echo docker || echo podman)
STEPS=${*:-ruff bandit vulture pip npm semgrep gitleaks}
mkdir -p "$OUT"

step() { printf '\n== %s\n' "$1"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has ruff; then
    step "ruff: likely bugs, security, complexity (beyond make lint)"
    $B/ruff check src --select B,S,BLE,PLE,PLW,PERF,C90,RUF \
        --ignore S101,S311,RUF001,RUF002,RUF003 --output-format concise \
        > "$OUT/ruff.txt" 2>&1
    tail -n 40 "$OUT/ruff.txt"
fi
if has bandit; then
    step "bandit: medium and high severity"
    $B/bandit -q -r src -ll -f txt -o "$OUT/bandit.txt"
    grep -E "^>> Issue|Location" "$OUT/bandit.txt" | paste - - | sed 's/   */ /g' | head -40
fi
if has vulture; then
    step "vulture: unused code (80 % confidence and up)"
    $B/vulture src --min-confidence 80 > "$OUT/vulture.txt" 2>&1
    head -n 40 "$OUT/vulture.txt"
fi
if has pip; then
    step "pip-audit: known vulnerabilities in the locked Python packages"
    $B/pip-audit --disable-pip --no-deps --timeout 60 -r requirements.lock > "$OUT/pip-audit.txt" 2>&1
    tail -n 30 "$OUT/pip-audit.txt"
fi
if has npm; then
    step "npm audit: the web app's packages"
    (cd app && npm audit) > "$OUT/npm-audit.txt" 2>&1
    tail -n 30 "$OUT/npm-audit.txt"
fi
if has semgrep; then
    step "semgrep: python, security-audit and secrets rule packs"
    $CTR run --rm -v "$PWD:/src:ro,z" -w /src "$SEMGREP_IMAGE" semgrep scan \
        --metrics=off --quiet --config p/python --config p/security-audit \
        --config p/secrets --config p/dockerfile --config p/nginx \
        --exclude experiments --exclude app/node_modules --exclude app/dist \
        --exclude .venv src app scripts db docker-compose.yml Dockerfile \
        > "$OUT/semgrep.txt" 2>&1
    tail -n 60 "$OUT/semgrep.txt"
fi
if has gitleaks; then
    step "gitleaks: secrets anywhere in the git history (redacted)"
    $CTR run --rm -v "$PWD:/repo:ro,z" "$GITLEAKS_IMAGE" git /repo \
        --redact --no-banner --report-format json --report-path /dev/stdout \
        > "$OUT/gitleaks.json" 2> "$OUT/gitleaks.log"
    tail -n 3 "$OUT/gitleaks.log"
    python3 - "$OUT/gitleaks.json" <<'EOF'
import json, sys
try:
    found = json.load(open(sys.argv[1]))
except ValueError:
    found = []
for f in found:
    print(f"{f['RuleID']:<28} {f['File']}:{f['StartLine']}  {f['Commit'][:8]}")
print(f"{len(found)} finding(s)")
EOF
fi
printf '\nreports in %s/\n' "$OUT"
