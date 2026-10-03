import pytest


@pytest.fixture(autouse=True)
def _fresh_rate_limits():
    """Rate-limit counters are process-wide; no test may inherit another's."""
    from malrec import access
    access._hits.clear()
    yield
    access._hits.clear()
