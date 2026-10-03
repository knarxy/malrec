# Uses docker if the daemon is reachable, otherwise rootless podman.
COMPOSE := $(shell docker info >/dev/null 2>&1 && echo "docker compose" || echo "podman-compose")
PY := .venv/bin/python

.PHONY: db-up db-down db-logs psql install init bootstrap train recommend serve worker publish test lint reset up down app-dev build logs

db-up:        ; $(COMPOSE) up -d db
db-down:      ; $(COMPOSE) down
db-logs:      ; $(COMPOSE) logs -f db
psql:         ; podman exec -it malrec-db psql -U malrec -d malrec
install:      ; python3 -m venv .venv && $(PY) -m pip install -q -e ".[dev]"
init:         ; $(PY) -m malrec.cli init
bootstrap:    ; $(PY) -m malrec.cli sync all
train:        ; $(PY) -m malrec.cli train
recommend:    ; $(PY) -m malrec.cli recommend
serve:        ; $(PY) -m malrec.cli serve --reload
worker:       ; $(PY) -m malrec.cli worker        # background tasks (or TASKS_INLINE=true)
publish:      ; MSG="$(MSG)" scripts/publish.sh          # master -> public repo (see the script)
test:         ; .venv/bin/pytest -q
lint:         ; .venv/bin/ruff check src tests

# Full stack in containers: database, API and web app on :3000.
# `down` first because podman-compose will otherwise recreate containers from
# the previously tagged image and silently run stale code.
up:
	$(COMPOSE) down 2>/dev/null || true
	$(COMPOSE) up -d --build
down:         ; $(COMPOSE) down
logs:         ; $(COMPOSE) logs -f api app
build:        ; $(COMPOSE) build

# Frontend against a locally-run API (make serve in another shell).
app-dev:      ; cd app && npm install && npm run dev

# Wipes the database volume. Everything must be refetched afterwards.
reset:
	$(COMPOSE) down -v && $(COMPOSE) up -d db

# ------------------------------------------------------- remote server --
# A remote host for deployment and experiments. Everything there runs in its
# own compose project, bound to loopback (or a private address) only, and never
# touches the host's other containers or volumes. Set DEV to its ssh target,
# here or in an untracked .make.local (DEV := user@host).
-include .make.local
DEV     ?= user@your-server
DEV_DIR ?= /opt/malrec
RSYNC_EXCLUDES := --exclude .venv/ --exclude app/node_modules/ --exclude app/dist/ \
	--exclude __pycache__/ --exclude .pytest_cache/ --exclude .ruff_cache/ \
	--exclude '*.tsbuildinfo' --exclude .env --exclude malrec.dump

.PHONY: dev-sync dev-up dev-lab dev-ps dev-logs

# code only; the dev .env (VPN bind address, secrets) lives on the server
dev-sync:
	rsync -az $(RSYNC_EXCLUDES) ./ $(DEV):$(DEV_DIR)/

# migrations checked on a backup copy first; unhealthy deploys roll back
dev-up: dev-sync
	ssh $(DEV) '$(DEV_DIR)/scripts/deploy.sh'

# e.g. make dev-lab CMD="python -u experiments/exp_final.py"
dev-lab: dev-sync
	ssh -t $(DEV) 'cd $(DEV_DIR) && docker compose --profile lab run --rm lab $(CMD)'

dev-ps:
	ssh $(DEV) 'docker ps --filter name=malrec --format "{{.Names}}\t{{.Status}}\t{{.Ports}}"'

dev-logs:
	ssh $(DEV) 'cd $(DEV_DIR) && docker compose logs -f --tail 50 api app'
