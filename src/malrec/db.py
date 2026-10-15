from __future__ import annotations

import atexit
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, LiteralString, cast

import psycopg
import psycopg.types.json
from psycopg.rows import DictRow, TupleRow, dict_row, tuple_row
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
def conn() -> Iterator[psycopg.Connection[DictRow]]:
    with pool().connection() as c:
        yield cast(psycopg.Connection[DictRow], c)     # the pool sets row_factory=dict_row


def trusted_sql(sql: str) -> LiteralString:
    # SQL here is written in the code, never built from input (values always
    # go in as parameters); psycopg's types only accept literal strings
    return cast(LiteralString, sql)


def query(sql: str, params: Any = None) -> list[dict]:
    with conn() as c, c.cursor() as cur:
        cur.execute(trusted_sql(sql), params)
        return cur.fetchall() if cur.description else []


def one(sql: str, params: Any = None) -> dict | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def stream(sql: str, params: Any = None, batch: int = 50_000) -> Iterator[TupleRow]:
    """Rows as plain tuples from a server-side cursor, `batch` at a time - for
    the millions of sampled list entries, which as dicts in one list would
    need gigabytes (recsys.service.load_cf_columns)."""
    with (conn() as c, c.transaction(),
          c.cursor(name="malrec_stream", row_factory=tuple_row) as cur):
        cur.itersize = batch
        cur.execute(trusted_sql(sql), params)
        yield from cur


def require_one(sql: str, params: Any = None) -> dict:
    """The one row a query must return (an INSERT ... RETURNING, or a row the
    caller knows exists). Raises LookupError instead of a TypeError later."""
    row = one(sql, params)
    if row is None:
        raise LookupError(f"expected a row from: {' '.join(sql.split())[:80]}")
    return row


def returned(cur: psycopg.Cursor[DictRow]) -> DictRow:
    """The row an INSERT/UPDATE ... RETURNING just produced."""
    row = cur.fetchone()
    if row is None:
        raise LookupError("the statement returned no row")
    return row


def scalar(sql: str, params: Any = None) -> Any:
    row = one(sql, params)
    return next(iter(row.values())) if row else None


def execute(sql: str, params: Any = None) -> int:
    with conn() as c, c.cursor() as cur:
        cur.execute(trusted_sql(sql), params)
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
                cur.execute(trusted_sql(f.read_text()))
            c.commit()
            applied.append(f.name)
            log.info("applied migration %s", f.name)
    return applied


def refresh_franchises() -> tuple[int, int]:
    """Recompute series and franchises from `relation` (recsys/series.py).
    Returns how many of each have more than one title."""
    from .recsys.series import refresh
    return refresh()


def log_ingest(job: str, status: str, detail: dict | None = None) -> None:
    execute(
        "INSERT INTO ingest_log (job, status, detail, finished_at) VALUES (%s, %s, %s, now())",
        (job, status, psycopg.types.json.Jsonb(detail or {})),
    )
