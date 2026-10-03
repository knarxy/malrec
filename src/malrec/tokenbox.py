"""MAL tokens at rest.

Access and refresh tokens let the app write to a user's MAL list, so they are
never stored in the clear once a key is configured: Fernet (AES-128-CBC +
HMAC-SHA256) with TOKEN_KEY from the server's .env, which is not in the
database and not in its backups. Values carry a version prefix; anything
without it is a token stored before encryption and is read as is (and
re-sealed by `malrec init`).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from .config import settings

log = logging.getLogger(__name__)
PREFIX = "enc:v1:"


@lru_cache
def _fernet():
    key = settings().token_key
    if not key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(key.encode())


def seal(token: str | None) -> str | None:
    f = _fernet()
    if token is None or f is None or token.startswith(PREFIX):
        return token
    return PREFIX + f.encrypt(token.encode()).decode()


def open_(value: str | None) -> str | None:
    if value is None or not value.startswith(PREFIX):
        return value
    f = _fernet()
    if f is None:
        raise RuntimeError("TOKEN_KEY is not set, but stored MAL tokens are encrypted")
    return f.decrypt(value[len(PREFIX):].encode()).decode()


def reseal_all() -> int:
    """Encrypt any token still stored in the clear. Returns rows changed."""
    from .db import execute, query
    if _fernet() is None:
        log.warning("TOKEN_KEY not set: MAL tokens are stored unencrypted")
        return 0
    n = 0
    for r in query("SELECT token_hash, access_token, refresh_token FROM mal_session"
                   " WHERE access_token NOT LIKE 'enc:%%' OR refresh_token NOT LIKE 'enc:%%'"):
        execute("UPDATE mal_session SET access_token=%s, refresh_token=%s WHERE token_hash=%s",
                (seal(r["access_token"]), seal(r["refresh_token"]), r["token_hash"]))
        n += 1
    return n
