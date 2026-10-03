from __future__ import annotations

import atexit
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

log = logging.getLogger(__name__)

def _find_migrations() -> Path:
    """Locate db/migrations for both an editable checkout and an installed
    package. In a container the code lives in site-packages while the SQL is
    copied next to the working directory, so the source-relative path alone is
    not enough."""
    env = os.environ.get("MALREC_MIGRATIONS")
    candidates = [
        Path(env) if env else None,
        Path(__file__).resolve().parents[2] / "db" / "migrations",   # src layout
        Path.cwd() / "db" / "migrations",                            # container
        Path("/app/db/migrations"),
    ]
    for c in candidates:
        if c and c.is_dir() and any(c.glob("*.sql")):
            return c
    # fall back to the source-relative guess so the error names a real path
    return Path(__file__).resolve().parents[2] / "db" / "migrations"


MIGRATIONS = _find_migrations()

_pool: ConnectionPool | None = None


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(settings().conninfo, min_size=1, max_size=8, open=True,
                               kwargs={"row_factory": dict_row})
        # psycopg's pool spawns worker threads; closing them at interpreter
        # shutdown avoids PythonFinalizationError on 3.13+.
        atexit.register(close_pool)
    return _pool


@contextmanager
def conn() -> Iterator[psycopg.Connection]:
    with pool().connection() as c:
        yield c


def query(sql: str, params: Any = None) -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall() if cur.description else []


def one(sql: str, params: Any = None) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def scalar(sql: str, params: Any = None) -> Any:
    row = one(sql, params)
    return next(iter(row.values())) if row else None


def execute(sql: str, params: Any = None) -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def migrate() -> list[str]:
    """Apply every .sql file in db/migrations in name order. All statements are
    idempotent (CREATE ... IF NOT EXISTS / CREATE OR REPLACE), so re-running is
    safe and doubles as the schema-drift check."""
    applied = []
    files = sorted(MIGRATIONS.glob("*.sql"))
    if not files:
        raise FileNotFoundError(f"no migrations found in {MIGRATIONS}")
    with conn() as c:
        for f in files:
            with c.cursor() as cur:
                cur.execute(f.read_text())
            c.commit()
            applied.append(f.name)
            log.info("applied migration %s", f.name)
    return applied


def refresh_franchises() -> int:
    """Recompute connected components over `relation`. Returns passes needed."""
    with conn() as c, c.cursor() as cur:
        cur.execute("SELECT refresh_franchises() AS passes")
        n = cur.fetchone()["passes"]
        c.commit()
    return n


def log_ingest(job: str, status: str, detail: dict | None = None) -> None:
    execute(
        "INSERT INTO ingest_log (job, status, detail, finished_at) VALUES (%s, %s, %s, now())",
        (job, status, psycopg.types.json.Jsonb(detail or {})),
    )
