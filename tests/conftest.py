import pytest


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
