"""Admin notifications by e-mail.

The admin (ADMIN_EMAIL) hears about what needs a hand or a glance:

  pending       an account is waiting for approval (sent at sign-in)
  failures      background tasks failed - bundled: the first failure is
                mailed at once, later ones at most every 30 minutes
  monthly       the monthly population refresh ran: new model or kept, with
                the gate's numbers
  weekly        the weekly "coming soon" refresh ran
  job_failed    a scheduled job (backup, nightly list sync, the monthly or
                weekly run) failed

Mails are in the admin account's app language (German or English), styled
like the app, with a plain-text alternative. Off unless SMTP_HOST and
ADMIN_EMAIL are set; sending never raises into the caller - a mail problem
must not fail a sign-in or a job. Every mail is recorded in the
`notification` table, which also keeps the same event from being mailed
twice.
"""
from __future__ import annotations

import datetime as dt
import html
import logging
import smtplib
import ssl
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formataddr, formatdate, make_msgid
from importlib import resources

from .config import settings
from .db import execute, one, query, scalar

log = logging.getLogger(__name__)

FAILURE_EVERY = dt.timedelta(minutes=30)


class Link:
    """A row value rendered as a link (everything else is escaped text)."""
    def __init__(self, url: str, label: str | None = None):
        self.url, self.label = url, label or url

    def __str__(self) -> str:
        return self.label

# design tokens of the app (app/src/styles.css), as e-mail-safe literals
BG, PANEL, LINE, TEXT, MUTED = "#0b0d12", "#141821", "#262c38", "#e8eaf0", "#9aa3b5"
BLUE, MINT, WARN, BAD = "#7cb3ff", "#57d9a3", "#f0a868", "#f07878"
FONT = "'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

STRINGS: dict[str, dict[str, str]] = {
    "de": {
        "footer": "Du bekommst diese Mail als Admin von malrec.",
        "open_app": "malrec öffnen",
        "pending.subject": "Wartet auf Freigabe: {user}",
        "pending.title": "{user} wartet auf Freigabe",
        "pending.lead": "Ein neues Konto hat sich angemeldet und kann malrec erst nach deiner Freigabe nutzen. "
                        "Du kannst es über die Glocke oben in der App freischalten.",
        "pending.who": "Konto",
        "pending.profile": "MAL-Profil",
        "pending.when": "Angefragt",
        "pending.queue": "Wartend insgesamt",
        "pending.others": "Außerdem wartend",
        "failures.subject": "{n} Hintergrund-Aufgabe(n) fehlgeschlagen",
        "failures.title": "Hintergrund-Aufgaben fehlgeschlagen",
        "failures.lead": "Diese Aufgaben sind seit der letzten Meldung fehlgeschlagen. Weitere Fehler werden "
                         "höchstens alle 30 Minuten gesammelt gemeldet.",
        "failures.worker": "Worker",
        "failures.queue": "{q} wartend · {r} laufend · {f} fehlgeschlagen in 24 h",
        "monthly.subject.on": "Monatliche Aktualisierung: neues Modell #{id} aktiv",
        "monthly.subject.off": "Monatliche Aktualisierung: Modell #{id} bleibt",
        "monthly.subject.dry": "Monatliche Aktualisierung (Testlauf): {result}",
        "monthly.title.on": "Neues Bevölkerungsmodell ist live",
        "monthly.title.off": "Das bisherige Modell bleibt",
        "monthly.lead.on": "Der monatliche Lauf hat ein neues Modell trainiert, es hat die Prüfung bestanden "
                           "und alle Konten wurden neu berechnet.",
        "monthly.lead.off": "Der monatliche Lauf hat ein neues Modell trainiert, es hat die Prüfung nicht bestanden. "
                            "Nutzer sehen keine Änderung.",
        "monthly.lead.dry": "Testlauf: Es wurde nichts aktiviert und niemand neu berechnet.",
        "monthly.gate": "Prüfung",
        "monthly.result": "Ergebnis",
        "monthly.reason": "Begründung",
        "monthly.candidate": "Kandidat",
        "monthly.lists": "{n} Listen",
        "monthly.exam": "Test auf neuen Listen",
        "monthly.exam_val": "{users} Listen · ρ {before} → {after} (Δ {delta}, ±{se})",
        "monthly.gate_user": "Referenzprofil",
        "monthly.gate_val": "ρ {before} → {after}",
        "monthly.prospective": "Test mit neuen Bewertungen",
        "monthly.users": "Konten",
        "monthly.rebuilt": "{ok} neu berechnet, {bad} fehlgeschlagen",
        "monthly.feedback": "Feedback für die Gewichte",
        "monthly.passed": "bestanden",
        "monthly.failed": "nicht bestanden",
        "weekly.subject": "Wöchentliche Aktualisierung: {related} neue Verknüpfungen",
        "weekly.title": "Wöchentliche „Demnächst“-Aktualisierung",
        "weekly.lead": "Neu angekündigte Staffeln und Filme wurden geladen und die Demnächst-Listen neu berechnet.",
        "weekly.catalog": "Katalogeinträge aktualisiert",
        "weekly.related": "Neue Verknüpfungen (Fortsetzungen, Filme …)",
        "weekly.users": "Demnächst pro Konto",
        "weekly.titles": "{n} Titel",
        "weekly.failed": "Fehlgeschlagen",
        "job.subject": "Fehlgeschlagen: {job}",
        "job.title": "{job} ist fehlgeschlagen",
        "job.lead": "Ein geplanter Job ist nicht sauber durchgelaufen. Die letzten Zeilen seines Protokolls:",
        "job.backup": "Nächtliche Datenbank-Sicherung",
        "job.list_sync": "Nächtlicher Listen-Abgleich",
        "job.monthly": "Monatliche Aktualisierung",
        "job.weekly": "Wöchentliche Demnächst-Aktualisierung",
        "kind.rebuild": "Neuberechnung", "kind.sync": "Abgleich", "kind.login_sync": "Abgleich bei Anmeldung",
        "kind.onboard": "Einrichtung",
        "test.subject": "Testmail von malrec",
        "test.title": "Die Mails funktionieren",
        "test.lead": "Diese Mail wurde über das konfigurierte Postfach verschickt. So sehen malrec-Mails aus.",
        "sample": "Beispiel",
    },
    "en": {
        "footer": "You get this mail as an admin of malrec.",
        "open_app": "Open malrec",
        "pending.subject": "Waiting for approval: {user}",
        "pending.title": "{user} is waiting for approval",
        "pending.lead": "A new account signed in and can use malrec once you approve it. "
                        "You can approve it from the bell at the top of the app.",
        "pending.who": "Account",
        "pending.profile": "MAL profile",
        "pending.when": "Requested",
        "pending.queue": "Waiting in total",
        "pending.others": "Also waiting",
        "failures.subject": "{n} background task(s) failed",
        "failures.title": "Background tasks failed",
        "failures.lead": "These tasks failed since the last report. Further failures are reported together, "
                         "at most every 30 minutes.",
        "failures.worker": "Worker",
        "failures.queue": "{q} waiting · {r} running · {f} failed in 24 h",
        "monthly.subject.on": "Monthly refresh: new model #{id} is live",
        "monthly.subject.off": "Monthly refresh: model #{id} stays",
        "monthly.subject.dry": "Monthly refresh (dry run): {result}",
        "monthly.title.on": "A new population model is live",
        "monthly.title.off": "The current model stays",
        "monthly.lead.on": "The monthly run trained a new model, it passed the check, and every account was rebuilt.",
        "monthly.lead.off": "The monthly run trained a new model, and it did not pass the check. Users see no change.",
        "monthly.lead.dry": "Dry run: nothing was activated and nobody was rebuilt.",
        "monthly.gate": "Check",
        "monthly.result": "Result",
        "monthly.reason": "Reason",
        "monthly.candidate": "Candidate",
        "monthly.lists": "{n} lists",
        "monthly.exam": "Test on new lists",
        "monthly.exam_val": "{users} lists · ρ {before} → {after} (Δ {delta}, ±{se})",
        "monthly.gate_user": "Reference profile",
        "monthly.gate_val": "ρ {before} → {after}",
        "monthly.prospective": "Test on new ratings",
        "monthly.users": "Accounts",
        "monthly.rebuilt": "{ok} rebuilt, {bad} failed",
        "monthly.feedback": "Feedback for the weights",
        "monthly.passed": "passed",
        "monthly.failed": "not passed",
        "weekly.subject": "Weekly refresh: {related} new links",
        "weekly.title": "Weekly “coming soon” refresh",
        "weekly.lead": "Newly announced seasons and films were loaded and the Coming Soon lists rebuilt.",
        "weekly.catalog": "Catalogue entries refreshed",
        "weekly.related": "New links (sequels, films, …)",
        "weekly.users": "Coming Soon per account",
        "weekly.titles": "{n} titles",
        "weekly.failed": "Failed",
        "job.subject": "Failed: {job}",
        "job.title": "{job} failed",
        "job.lead": "A scheduled job did not finish cleanly. The last lines of its log:",
        "job.backup": "Nightly database backup",
        "job.list_sync": "Nightly list sync",
        "job.monthly": "Monthly refresh",
        "job.weekly": "Weekly Coming Soon refresh",
        "kind.rebuild": "Rebuild", "kind.sync": "Sync", "kind.login_sync": "Sync at sign-in",
        "kind.onboard": "Setup",
        "test.subject": "Test mail from malrec",
        "test.title": "Mails are working",
        "test.lead": "This mail was sent through the configured mailbox. This is what malrec mails look like.",
        "sample": "Sample",
    },
}


# ------------------------------------------------------------- basics --

def enabled() -> bool:
    cfg = settings()
    return bool(cfg.smtp_host and cfg.admin_email)


def admin_lang() -> str:
    """The admin account's app language (its saved preference), else English."""
    names = [a.strip() for a in settings().admin_users.split(",") if a.strip()]
    if names:
        lang = scalar("SELECT prefs->>'lang' FROM app_user WHERE lower(mal_username) = lower(%s)",
                      (names[0],))
        if lang in STRINGS:
            return lang
    return "en"


def _t(lang: str, key: str, **kw) -> str:
    s = STRINGS.get(lang, STRINGS["en"]).get(key) or STRINGS["en"].get(key, key)
    return s.format(**kw) if kw else s


def _when(ts: dt.datetime | str | None, lang: str) -> str:
    if ts is None:
        return "—"
    if isinstance(ts, str):
        ts = dt.datetime.fromisoformat(ts)
    try:
        from zoneinfo import ZoneInfo
        ts = ts.astimezone(ZoneInfo(settings().notify_timezone))
    except Exception:  # noqa: BLE001 - no tz data: show it as stored
        log.debug("timezone %r unavailable", settings().notify_timezone)
    return ts.strftime("%d.%m.%Y, %H:%M" if lang == "de" else "%Y-%m-%d %H:%M")


def _record(kind: str, key: str, ok: bool, error: str | None = None) -> None:
    execute("INSERT INTO notification (kind, key, ok, error) VALUES (%s, %s, %s, %s)"
            " ON CONFLICT (kind, key) DO UPDATE SET ok = EXCLUDED.ok, error = EXCLUDED.error,"
            " sent_at = now()", (kind, key, ok, error))


def _already(kind: str, key: str) -> bool:
    return bool(scalar("SELECT ok FROM notification WHERE kind=%s AND key=%s", (kind, key)))


# ------------------------------------------------------------ rendering --

def _cell(v) -> str:
    if isinstance(v, Link):
        return (f'<a href="{html.escape(v.url)}" style="color:{BLUE};text-decoration:none">'
                f'{html.escape(v.label)}</a>')
    return html.escape(str(v))


def render(lang: str, title: str, lead: str, sections: list[dict], tone: str = "info",
           badge: str | None = None, button: tuple[str, str] | None = None
           ) -> tuple[str, str]:
    """(html, text) for one mail. A section is {"h": heading, "rows": [(label,
    value)], "p": [paragraph], "pre": preformatted text}."""
    e = html.escape
    accent = {"ok": MINT, "warn": WARN, "bad": BAD}.get(tone, BLUE)
    parts: list[str] = []
    text: list[str] = [title, "", lead, ""]
    for s in sections:
        if s.get("h"):
            parts.append(f'<tr><td style="padding:22px 28px 6px;font:600 12px/1.4 {FONT};'
                         f'letter-spacing:.08em;text-transform:uppercase;color:{MUTED}">{e(s["h"])}</td></tr>')
            text.append(s["h"].upper())
        if s.get("rows"):
            rows = "".join(
                f'<tr><td style="padding:7px 14px 7px 0;font:400 14px/1.5 {FONT};color:{MUTED};'
                f'vertical-align:top;white-space:nowrap">{e(str(k))}</td>'
                f'<td style="padding:7px 0;font:600 14px/1.5 {FONT};color:{TEXT};vertical-align:top">'
                f'{_cell(v)}</td></tr>'
                for k, v in s["rows"])
            parts.append(f'<tr><td style="padding:0 28px"><table role="presentation" width="100%" '
                         f'cellpadding="0" cellspacing="0" style="border-top:1px solid {LINE}">'
                         f'{rows}</table></td></tr>')
            for k, v in s["rows"]:
                text.append(f"  {k}: {v.url if isinstance(v, Link) else v}")
        for p in s.get("p", []):
            parts.append(f'<tr><td style="padding:6px 28px;font:400 14px/1.6 {FONT};color:#c9cfdb">'
                         f'{e(p)}</td></tr>')
            text.append(p)
        if s.get("pre"):
            parts.append(f'<tr><td style="padding:6px 28px"><pre style="margin:0;padding:12px;'
                         f'background:{BG};border:1px solid {LINE};border-radius:10px;'
                         f'font:12px/1.5 Consolas,Menlo,monospace;color:{MUTED};white-space:pre-wrap;'
                         f'word-break:break-word">{e(s["pre"])}</pre></td></tr>')
            text.append(s["pre"])
        text.append("")
    if button:
        url, label = button
        parts.append(f'<tr><td style="padding:24px 28px 4px"><table role="presentation" cellpadding="0" '
                     f'cellspacing="0"><tr><td bgcolor="{BLUE}" style="border-radius:999px;'
                     f'background:{BLUE};background-image:linear-gradient(90deg,{BLUE},{MINT})">'
                     f'<a href="{e(url)}" style="display:inline-block;padding:11px 22px;'
                     f'font:700 14px/1 {FONT};color:#0b1220;text-decoration:none">{e(label)}</a>'
                     f'</td></tr></table></td></tr>')
        text += [f"{label}: {url}", ""]
    pill = (f'<span style="display:inline-block;margin-left:8px;padding:3px 10px;border-radius:999px;'
            f'border:1px solid {accent};color:{accent};font:600 12px/1.4 {FONT};vertical-align:middle">'
            f'{e(badge)}</span>') if badge else ""
    footer = _t(lang, "footer")
    base = settings().app_base_url.rstrip("/")
    page = f"""<!doctype html><html lang="{lang}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark">
<title>{e(title)}</title></head>
<body style="margin:0;padding:0;background:{BG}" bgcolor="{BG}">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" bgcolor="{BG}" style="background:{BG}">
<tr><td align="center" style="padding:28px 12px">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:600px">
<tr><td style="padding:0 4px 16px"><table role="presentation" cellpadding="0" cellspacing="0"><tr>
<td style="vertical-align:middle"><img src="cid:malrec-logo" width="34" height="34" alt="" style="display:block;border-radius:9px"></td>
<td style="vertical-align:middle;padding-left:10px;font:700 20px/1 {FONT};letter-spacing:-0.03em;color:{TEXT}">mal<span style="color:{MINT}">rec</span></td>
</tr></table></td></tr>
<tr><td bgcolor="{PANEL}" style="background:{PANEL};border:1px solid {LINE};border-radius:18px;overflow:hidden">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0">
<tr><td height="4" bgcolor="{accent}" style="height:4px;line-height:4px;font-size:0;background:{accent};background-image:linear-gradient(90deg,{BLUE},{MINT})">&nbsp;</td></tr>
<tr><td style="padding:26px 28px 6px;font:700 22px/1.3 {FONT};letter-spacing:-0.02em;color:{TEXT}">{e(title)}{pill}</td></tr>
<tr><td style="padding:4px 28px 4px;font:400 15px/1.6 {FONT};color:#c9cfdb">{e(lead)}</td></tr>
{"".join(parts)}
<tr><td style="padding:22px 28px 26px"></td></tr>
</table></td></tr>
<tr><td style="padding:18px 8px;font:400 12px/1.6 {FONT};color:{MUTED};text-align:center">
{e(footer)} · <a href="{e(base)}" style="color:{BLUE};text-decoration:none">{e(base.split("//")[-1])}</a></td></tr>
</table></td></tr></table></body></html>"""
    text.append(f"-- {footer} {base}")
    return page, "\n".join(text)


def send(subject: str, page: str, text: str) -> str | None:
    """Send to the admin address(es). Returns None on success, else the
    error. Never raises."""
    cfg = settings()
    if not enabled():
        return "notifications are not configured"
    # multipart/related (type multipart/alternative): the text/HTML choice
    # first, then the inline logo it refers to. Nesting the logo inside the
    # HTML alternative instead made some clients (SOGo among them) fall back
    # to the plain text.
    msg = MIMEMultipart("related", type="multipart/alternative")
    msg["Subject"] = f"[malrec] {subject}"
    msg["From"] = formataddr(("malrec", cfg.smtp_from or cfg.smtp_user))
    msg["To"] = ", ".join(a.strip() for a in cfg.admin_email.split(",") if a.strip())
    msg["Date"] = formatdate(localtime=True)     # neither smtplib nor the server adds it
    msg["Message-ID"] = make_msgid(domain=(cfg.smtp_from or cfg.smtp_user).split("@")[-1] or None)
    alt = MIMEMultipart("alternative")
    alt.attach(MIMEText(text, "plain", "utf-8"))
    alt.attach(MIMEText(page, "html", "utf-8"))
    msg.attach(alt)
    try:
        logo = MIMEImage(resources.files("malrec").joinpath("assets/mail-logo.png").read_bytes(),
                         "png")
        logo.add_header("Content-ID", "<malrec-logo>")
        logo.add_header("Content-Disposition", "inline", filename="malrec.png")
        msg.attach(logo)
    except Exception:  # noqa: BLE001 - a mail without its logo still goes out
        log.warning("mail logo missing")
    try:
        with smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=30) as s:
            s.starttls(context=ssl.create_default_context())
            if cfg.smtp_user:
                s.login(cfg.smtp_user, cfg.smtp_password)
            s.send_message(msg)
        log.info("mail sent: %s", subject)
        return None
    except Exception as e:  # noqa: BLE001 - mail trouble must never fail the caller
        log.warning("mail %r not sent: %s", subject, type(e).__name__)
        return f"{type(e).__name__}: {e}"[:300]


def _deliver(kind: str, key: str, subject: str, page: str, text: str) -> bool:
    err = send(subject, page, text)
    try:
        _record(kind, key, err is None, err)
    except Exception:
        log.exception("could not record notification")
    return err is None


# --------------------------------------------------------------- events --

def pending_signup(user_id: int, sample: bool = False) -> bool:
    """An account asked for approval (called at sign-in, once per request)."""
    if not enabled():
        return False
    u = one("SELECT mal_username, requested_at FROM app_user WHERE id=%s", (user_id,))
    if u is None:
        return False
    key = f"{user_id}:{u['requested_at'].isoformat() if u['requested_at'] else ''}"
    if not sample and _already("pending", key):
        return False
    lang = admin_lang()
    others = [r["mal_username"] for r in query(
        "SELECT mal_username FROM app_user WHERE status='pending' AND id <> %s"
        " ORDER BY requested_at NULLS LAST LIMIT 10", (user_id,))]
    name = u["mal_username"]
    profile = f"https://myanimelist.net/profile/{name}"
    rows = [(_t(lang, "pending.who"), name),
            (_t(lang, "pending.profile"), Link(profile)),
            (_t(lang, "pending.when"), _when(u["requested_at"], lang)),
            (_t(lang, "pending.queue"), str(len(others) + 1))]
    if others:
        rows.append((_t(lang, "pending.others"), ", ".join(others)))
    page, text = render(lang, _t(lang, "pending.title", user=name), _t(lang, "pending.lead"),
                        [{"rows": rows}], tone="info",
                        badge=_t(lang, "sample") if sample else None,
                        button=(settings().app_base_url, _t(lang, "open_app")))
    return _deliver("pending", f"sample:{dt.datetime.now(dt.UTC).isoformat()}" if sample else key,
                    _t(lang, "pending.subject", user=name), page, text)


def task_failures(sample_rows: list[dict] | None = None) -> bool:
    """Mail the tasks that failed since the last such mail - at once for the
    first, then at most every FAILURE_EVERY (called by the worker after a
    failure and once a minute)."""
    if not enabled():
        return False
    last = scalar("SELECT max(sent_at) FROM notification WHERE kind='failures' AND ok")
    now = dt.datetime.now(dt.UTC)
    if sample_rows is None:
        if last is not None and now - last < FAILURE_EVERY:
            return False
        since = last or now - dt.timedelta(days=1)
        rows = query("""SELECT t.kind, u.mal_username AS username, t.error, t.finished_at,
                               t.attempts
                          FROM task t JOIN app_user u ON u.id = t.user_id
                         WHERE t.state = 'failed' AND t.finished_at > %s
                         ORDER BY t.finished_at""", (since,))
        if not rows:
            return False
    else:
        rows = sample_rows
    lang = admin_lang()
    from .tasks import status
    w = status()
    sections = [{"h": f"{_when(r['finished_at'], lang)} · {_t(lang, 'kind.' + r['kind'])} · "
                      f"{r['username']}", "pre": r["error"] or "—"} for r in rows[:20]]
    sections.append({"h": _t(lang, "failures.worker"),
                     "p": [_t(lang, "failures.queue", q=w["queued"], r=w["running"],
                              f=w["failed_24h"])]})
    page, text = render(lang, _t(lang, "failures.title"), _t(lang, "failures.lead"), sections,
                        tone="bad", badge=_t(lang, "sample") if sample_rows is not None
                        else str(len(rows)),
                        button=(settings().app_base_url, _t(lang, "open_app")))
    return _deliver("failures" if sample_rows is None else "sample", now.isoformat(),
                    _t(lang, "failures.subject", n=len(rows)), page, text)


def _fmt(x, nd: int = 3, signed: bool = False) -> str:
    if x is None:
        return "—"
    if isinstance(x, int | float):
        return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return str(x)


def monthly(fit: dict, log_tail: str = "", dry_run: bool = False, sample: bool = False) -> bool:
    """The monthly refresh finished: fit is `malrec population fit`'s output."""
    if not enabled():
        return False
    lang = admin_lang()
    on = "activated" in fit and not dry_run
    cid = fit.get("candidate") or fit.get("activated")
    rows = [(_t(lang, "monthly.result"),
             _t(lang, "monthly.passed") if fit.get("passed") else _t(lang, "monthly.failed")),
            (_t(lang, "monthly.candidate"),
             f"#{cid} · " + _t(lang, "monthly.lists", n=fit.get("users", "—")))]
    if fit.get("exam"):
        x = fit["exam"]
        rows.append((_t(lang, "monthly.exam"), _t(lang, "monthly.exam_val", users=x.get("users"),
                     before=_fmt(x.get("rho_before")), after=_fmt(x.get("rho_after")),
                     delta=_fmt(x.get("delta"), 4, signed=True), se=_fmt(x.get("se"), 4))))
    if fit.get("gate"):
        g = fit["gate"]
        rows.append((_t(lang, "monthly.gate_user"), _t(lang, "monthly.gate_val",
                     before=_fmt(g.get("rho_before")), after=_fmt(g.get("rho_after")))))
    if fit.get("prospective"):
        p = fit["prospective"]
        rows.append((_t(lang, "monthly.prospective"), p.get("note") or p.get("status", "—")))
    if fit.get("reason"):
        rows.append((_t(lang, "monthly.reason"), fit["reason"]))
    sections = [{"h": _t(lang, "monthly.gate"), "rows": rows}]
    rebuilt = [ln for ln in log_tail.splitlines() if ln.startswith("rebuilt ")]
    failed = [ln for ln in log_tail.splitlines() if "FAILED" in ln and ln.startswith("rebuild of")]
    if on:
        sections.append({"h": _t(lang, "monthly.users"),
                         "p": [_t(lang, "monthly.rebuilt", ok=len(rebuilt), bad=len(failed))]
                         + failed})
    fb = next((ln for ln in log_tail.splitlines() if '"status": "' in ln and "feedback" in ln), None)
    if fb:
        sections.append({"h": _t(lang, "monthly.feedback"),
                         "p": [fb.split('"status": "', 1)[1].rstrip('",')]})
    old = scalar("SELECT id FROM global_model WHERE active")
    title = _t(lang, "monthly.title.on" if on else "monthly.title.off")
    lead = _t(lang, "monthly.lead.dry" if dry_run else "monthly.lead.on" if on else "monthly.lead.off")
    subject = (_t(lang, "monthly.subject.dry", result=_t(lang, "monthly.passed") if fit.get("passed")
                  else _t(lang, "monthly.failed")) if dry_run
               else _t(lang, "monthly.subject.on" if on else "monthly.subject.off",
                       id=cid if on else old))
    page, text = render(lang, title, lead, sections, tone="ok" if on else "info",
                        badge=_t(lang, "sample") if sample else None,
                        button=(settings().app_base_url, _t(lang, "open_app")))
    return _deliver("sample" if sample else "monthly",
                    dt.datetime.now(dt.UTC).isoformat(), subject, page, text)


def weekly(result: dict, sample: bool = False) -> bool:
    """The weekly coming-soon refresh finished: result is `malrec sync
    upcoming`'s output."""
    if not enabled():
        return False
    lang = admin_lang()
    per_user = query("""SELECT u.mal_username, count(r.mal_id) AS n,
                               (array_agg(coalesce(a.title_en, a.title) ORDER BY r.rank))[1:3] AS top
                          FROM app_user u
                          LEFT JOIN recommendation r ON r.user_id = u.id AND r.surface = 'coming_soon'
                          LEFT JOIN anime a ON a.mal_id = r.mal_id
                         WHERE u.status = 'approved'
                         GROUP BY u.mal_username ORDER BY u.mal_username""")
    rows = [(_t(lang, "weekly.catalog"), str(result.get("catalog", "—"))),
            (_t(lang, "weekly.related"), str(result.get("related", "—")))]
    users = [(u["mal_username"], _t(lang, "weekly.titles", n=u["n"])
              + (f" · {', '.join(t for t in (u['top'] or []) if t)}" if u["n"] else ""))
             for u in per_user]
    sections = [{"rows": rows}, {"h": _t(lang, "weekly.users"), "rows": users}]
    if result.get("failed"):
        sections.append({"h": _t(lang, "weekly.failed"),
                         "rows": list(result["failed"].items())})
    page, text = render(lang, _t(lang, "weekly.title"), _t(lang, "weekly.lead"), sections,
                        tone="warn" if result.get("failed") else "ok",
                        badge=_t(lang, "sample") if sample else None,
                        button=(settings().app_base_url, _t(lang, "open_app")))
    return _deliver("sample" if sample else "weekly", dt.datetime.now(dt.UTC).isoformat(),
                    _t(lang, "weekly.subject", related=result.get("related", 0)), page, text)


def job_failed(job: str, detail: str, sample: bool = False) -> bool:
    """A scheduled job failed; detail is the tail of its log."""
    if not enabled():
        return False
    lang = admin_lang()
    name = _t(lang, f"job.{job}") if f"job.{job}" in STRINGS["en"] else job
    tail = "\n".join(detail.strip().splitlines()[-30:]) or "—"
    page, text = render(lang, _t(lang, "job.title", job=name), _t(lang, "job.lead"),
                        [{"pre": tail}], tone="bad",
                        badge=_t(lang, "sample") if sample else None,
                        button=(settings().app_base_url, _t(lang, "open_app")))
    return _deliver("sample" if sample else "job_failed",
                    f"{job}:{dt.datetime.now(dt.UTC).isoformat()}",
                    _t(lang, "job.subject", job=name), page, text)


def nightly(result: list[dict]) -> bool:
    """The nightly list sync's per-user results: mail only if any failed."""
    errors = [r for r in result if r.get("error")]
    if not errors:
        return False
    return job_failed("list_sync", "\n".join(f"{r['user']}: {r['error']}" for r in errors))


def test(all_kinds: bool = False) -> dict[str, bool]:
    """A plain test mail, and with all_kinds one sample of every kind."""
    lang = admin_lang()
    page, text = render(lang, _t(lang, "test.title"), _t(lang, "test.lead"), [], tone="ok",
                        button=(settings().app_base_url, _t(lang, "open_app")))
    out = {"test": _deliver("sample", f"test:{dt.datetime.now(dt.UTC).isoformat()}",
                            _t(lang, "test.subject"), page, text)}
    if all_kinds:
        uid = scalar("SELECT id FROM app_user ORDER BY (status='pending') DESC, id LIMIT 1")
        out["pending"] = pending_signup(uid, sample=True) if uid else False
        now = dt.datetime.now(dt.UTC)
        out["failures"] = task_failures(sample_rows=[
            {"kind": "rebuild", "username": "example_user", "attempts": 1, "finished_at": now,
             "error": "MalApiError: 503 Service Unavailable"},
            {"kind": "sync", "username": "example_user", "attempts": 1, "finished_at": now,
             "error": "ReadTimeout: api.myanimelist.net"}])
        out["monthly"] = monthly({
            "candidate": 12, "users": 5019, "passed": True, "activated": 12,
            "exam": {"users": 812, "rho_before": 0.512, "rho_after": 0.518, "delta": 0.0061,
                     "se": 0.0031},
            "gate": {"rho_before": 0.797, "rho_after": 0.794},
            "prospective": {"note": "12 of 20 recommended titles rated since shown"},
            "reason": "exam: no worse on new lists"},
            log_tail="rebuilt user_a\nrebuilt user_b\nrebuilt user_c\nrebuilt user_d", sample=True)
        out["weekly"] = weekly({"catalog": 6913, "related": 2,
                                "rebuilt": "coming_soon for every user"}, sample=True)
        out["job_failed"] = job_failed(
            "backup", "2026-10-04T04:15:02+02:00 only 14 GB free; backup skipped", sample=True)
    return out
