"""Admin e-mails (malrec.notify), without a database or a mail server."""
from __future__ import annotations

import datetime as dt

import pytest

from malrec import notify
from malrec.config import settings


@pytest.fixture()
def mailer(monkeypatch):
    cfg = settings()
    monkeypatch.setattr(cfg, "smtp_host", "mail.example.org")
    monkeypatch.setattr(cfg, "smtp_user", "malrec@example.org")
    monkeypatch.setattr(cfg, "smtp_password", "x")
    monkeypatch.setattr(cfg, "admin_email", "admin@example.org")
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout):
            assert (host, port) == ("mail.example.org", 587)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context):
            pass

        def login(self, user, password):
            assert user == "malrec@example.org"

        def send_message(self, msg):
            sent.append(msg)
    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(notify, "_record", lambda *a, **k: None)
    monkeypatch.setattr(notify, "admin_lang", lambda: "de")
    return sent


def test_render_escapes_text_and_renders_links_only_from_the_link_type():
    page, text = notify.render("en", "A <b>title</b>", "lead & more", [
        {"h": "Rows", "rows": [("plain", '<a href="https://evil.example">x</a>'),
                               ("link", notify.Link("https://myanimelist.net/profile/x"))]},
        {"pre": "Traceback <script>"}])
    assert "&lt;b&gt;title&lt;/b&gt;" in page and "<b>title</b>" not in page
    assert '<a href="https://evil.example">' not in page          # a value, not markup
    assert '<a href="https://myanimelist.net/profile/x"' in page
    assert "&lt;script&gt;" in page
    assert "https://myanimelist.net/profile/x" in text and "<" not in text.split("\n")[0][:1]


def test_a_mail_is_html_with_a_text_part_and_the_logo_inline(mailer):
    page, text = notify.render("de", "Titel", "Einleitung", [])
    assert notify.send("Betreff", page, text) is None
    msg = mailer[0]
    assert msg["Subject"] == "[malrec] Betreff" and msg["To"] == "admin@example.org"
    from email.utils import parsedate_to_datetime
    sent_at = parsedate_to_datetime(msg["Date"])           # clients show 1970 without it
    assert abs((sent_at - dt.datetime.now(dt.UTC)).total_seconds()) < 60
    kinds = [p.get_content_type() for p in msg.walk()]
    # related around the text/HTML choice and the inline logo - nesting the
    # logo inside the HTML alternative made some clients show plain text only
    assert kinds == ["multipart/related", "multipart/alternative", "text/plain", "text/html",
                     "image/png"]
    assert msg.get_param("type") == "multipart/alternative"
    assert next(p for p in msg.walk() if p.get_content_type() == "image/png"
                )["Content-Disposition"].startswith("inline")
    assert any(p.get("Content-ID") == "<malrec-logo>" for p in msg.walk())


def test_nothing_is_sent_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings(), "smtp_host", "")
    assert notify.send("x", "<p>x</p>", "x") == "notifications are not configured"
    assert notify.pending_signup(1) is False


def test_task_failures_are_bundled(mailer, monkeypatch):
    now = dt.datetime.now(dt.UTC)
    state = {"last": None, "rows": [{"kind": "rebuild", "username": "u", "error": "Boom",
                                     "finished_at": now, "attempts": 1}]}
    monkeypatch.setattr(notify, "scalar", lambda sql, p=None: state["last"])
    monkeypatch.setattr(notify, "query", lambda sql, p=None: state["rows"])
    import malrec.tasks
    monkeypatch.setattr(malrec.tasks, "status", lambda: {"queued": 0, "running": 0,
                                                         "failed_24h": 1})
    assert notify.task_failures() is True                       # the first: at once
    assert "fehlgeschlagen" in mailer[-1]["Subject"]
    state["last"] = now - dt.timedelta(minutes=5)
    assert notify.task_failures() is False                      # within 30 min: waits
    state["last"] = now - dt.timedelta(minutes=31)
    assert notify.task_failures() is True                       # then the bundle
    state["rows"] = []
    state["last"] = now - dt.timedelta(hours=2)
    assert notify.task_failures() is False                      # nothing new: no mail
    assert len(mailer) == 2


def test_nightly_mails_only_on_errors(mailer):
    assert notify.nightly([{"user": "a", "changed": False}]) is False
    assert notify.nightly([{"user": "a", "error": "MalApiError: 404"}]) is True
    assert "Listen-Abgleich" in mailer[-1]["Subject"]
