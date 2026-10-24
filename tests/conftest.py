import os

import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """With MALREC_REQUIRE_DB set (make check), a test skipped because the
    database is unreachable fails: a suite that skips half its tests must
    not pass for a green one."""
    outcome = yield
    rep = outcome.get_result()
    if (os.environ.get("MALREC_REQUIRE_DB") and rep.skipped
            and "database not reachable" in str(rep.longrepr)):
        rep.outcome = "failed"


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Rate-limit counters are process-wide; no test may inherit another's."""
    from malrec import access
    access._hits.clear()
    yield
    access._hits.clear()


@pytest.fixture(autouse=True)
def _no_real_mail(monkeypatch):
    """Tests never mail the admin, whatever the environment configures
    (the sign-in tests create waiting accounts). Mail tests patch in their
    own fake server."""
    from malrec.config import settings
    monkeypatch.setattr(settings(), "smtp_host", "")
    # nor do they depend on the server's test-phase auto-approval
    monkeypatch.setattr(settings(), "auto_approve_limit", 0)
