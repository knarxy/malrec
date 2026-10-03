// Verwaltung: das Admin-Panel. Nur für Admin-Konten sichtbar; die API prüft
// jede Anfrage selbst (malrec.admin), die Oberfläche ist nur die Bedienung.
import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, type AdminSystem, type AdminUser } from './api'

const STATUS: Record<string, string> = {
  pending: 'wartet', approved: 'freigeschaltet', rejected: 'abgelehnt', blocked: 'gesperrt',
}

function when(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  return d.toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' })
}

type Tab = 'users' | 'system' | 'report' | 'log'

export function Admin({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<Tab>('users')
  return (
    <>
      <header className="topbar">
        <div className="wrap topbar-inner">
          <span className="brand">mal<span>rec</span> · Verwaltung</span>
          <span className="spacer" />
          <button onClick={onClose}>Zurück zur App</button>
        </div>
      </header>
      <main className="wrap admin">
        <nav className="tabs" role="tablist">
          {([['users', 'Nutzer'], ['system', 'System'], ['report', 'Bericht'], ['log', 'Protokoll']] as
            [Tab, string][]).map(([k, label]) => (
            <button key={k} role="tab" className="tab" aria-selected={tab === k}
                    onClick={() => setTab(k)}>{label}</button>
          ))}
        </nav>
        {tab === 'users' && <Users />}
        {tab === 'system' && <System />}
        {tab === 'report' && <Report />}
        {tab === 'log' && <Log />}
      </main>
    </>
  )
}

function Users() {
  const [users, setUsers] = useState<AdminUser[] | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState<number | null>(null)

  const load = useCallback(() => {
    api.adminUsers().then(setUsers).catch((e) => setMsg(e instanceof ApiError ? e.message : 'Fehler'))
  }, [])
  useEffect(() => { load() }, [load])

  async function act(u: AdminUser, action: Parameters<typeof api.adminAction>[1] | 'delete',
                     confirmText?: string) {
    if (confirmText && !window.confirm(confirmText)) return
    setBusy(u.id); setMsg(null)
    try {
      if (action === 'delete') await api.adminDelete(u.id)
      else await api.adminAction(u.id, action)
      setMsg(`${u.mal_username}: ${{
        approve: 'freigeschaltet – die Einrichtung läuft', reject: 'abgelehnt', block: 'gesperrt',
        unblock: 'entsperrt', logout: 'alle Sitzungen beendet', sync: 'Abgleich gestartet',
        rebuild: 'Neuberechnung gestartet', delete: 'gelöscht',
      }[action]}`)
      load()
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : 'Aktion fehlgeschlagen')
    } finally {
      setBusy(null)
    }
  }

  if (!users) return <p className="surface-note">{msg ?? 'Lädt …'}</p>
  const pending = users.filter((u) => u.status === 'pending')
  const rest = users.filter((u) => u.status !== 'pending')
  return (
    <>
      {msg && <div className="notice">{msg}</div>}
      <h3>Offene Anfragen ({pending.length})</h3>
      {pending.length === 0 && <p className="surface-note">Keine offenen Anfragen.</p>}
      {pending.map((u) => (
        <div className="admin-row" key={u.id}>
          <div>
            <b>{u.mal_username}</b>{' '}
            <a href={`https://myanimelist.net/profile/${encodeURIComponent(u.mal_username)}`}
               target="_blank" rel="noreferrer">MAL-Profil ↗</a>
            <div className="surface-note">angefragt {when(u.requested_at)}</div>
          </div>
          <div className="admin-actions">
            <button className="primary" disabled={busy === u.id} onClick={() => act(u, 'approve')}>
              Freischalten
            </button>
            <button disabled={busy === u.id}
                    onClick={() => act(u, 'reject', `${u.mal_username} ablehnen?`)}>Ablehnen</button>
          </div>
        </div>
      ))}

      <h3>Alle Konten ({rest.length})</h3>
      <div className="admin-table">
        <table>
          <thead>
            <tr><th>Konto</th><th>Status</th><th>Bewertet</th><th>Empf.</th>
              <th>Letzte Anmeldung</th><th>Letzter Abgleich</th><th>Aktionen</th></tr>
          </thead>
          <tbody>
            {rest.map((u) => (
              <tr key={u.id}>
                <td>
                  <b>{u.mal_username}</b>{u.is_admin && <span className="chip">Admin</span>}
                  {u.onboarding === 'running' && <div className="surface-note">Einrichtung läuft …</div>}
                  {u.onboarding === 'failed' && <div className="surface-note">Einrichtung fehlgeschlagen</div>}
                </td>
                <td><span className={`badge ${u.status}`}>{STATUS[u.status] ?? u.status}</span></td>
                <td>{u.scored} / {u.entries}</td>
                <td>{u.recs}</td>
                <td>{when(u.last_login_at)}</td>
                <td>{when(u.last_sync_at)}</td>
                <td className="admin-actions">
                  {u.status === 'approved' && <>
                    <a className="button-link" href={`/?user=${encodeURIComponent(u.mal_username)}`}>Ansehen</a>
                    <button disabled={busy === u.id} onClick={() => act(u, 'sync')}>Abgleichen</button>
                    <button disabled={busy === u.id} onClick={() => act(u, 'rebuild')}>Neu berechnen</button>
                  </>}
                  {!u.is_admin && <>
                    <button disabled={busy === u.id || u.sessions === 0}
                            onClick={() => act(u, 'logout', `Alle Sitzungen von ${u.mal_username} beenden?`)}>
                      Abmelden ({u.sessions})
                    </button>
                    {u.status === 'approved'
                      ? <button disabled={busy === u.id}
                                onClick={() => act(u, 'block', `${u.mal_username} sperren? Alle Sitzungen enden sofort.`)}>Sperren</button>
                      : <button disabled={busy === u.id} onClick={() => act(u, 'unblock')}>Freischalten</button>}
                    <button disabled={busy === u.id}
                            onClick={() => act(u, 'delete', `${u.mal_username} und alle gespeicherten Daten löschen? Auf MyAnimeList ändert sich nichts.`)}>
                      Löschen
                    </button>
                  </>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}

function System() {
  const [s, setS] = useState<AdminSystem | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.adminSystem().then(setS).catch((e) => setErr(e instanceof ApiError ? e.message : 'Fehler'))
  }, [])
  if (!s) return <p className="surface-note">{err ?? 'Lädt …'}</p>
  const labels: Record<string, string> = {
    anime: 'Titel im Katalog', cf_lists: 'Listen im Bevölkerungsmodell', cf_ratings: 'Bewertungen darin',
    users: 'Konten', pending: 'offene Anfragen', sessions: 'aktive Sitzungen',
  }
  return (
    <>
      <h3>Geplante Aufgaben</h3>
      {Object.entries(s.jobs).map(([k, j]) => (
        <div className="admin-row" key={k}>
          <div>
            <b>{j.label}</b>{' '}
            <span className={`badge ${j.ok === true ? 'approved' : j.ok === false ? 'blocked' : 'pending'}`}>
              {j.state}
            </span>
            <div className="surface-note">zuletzt: {when(j.last_run)}</div>
            {j.tail && j.tail.length > 0 && <pre className="log-tail">{j.tail.join('\n')}</pre>}
          </div>
        </div>
      ))}
      <h3>Daten</h3>
      <div className="stat-grid">
        {Object.entries(s.counts).map(([k, v]) => (
          <div className="stat" key={k}><div className="k">{labels[k] ?? k}</div>
            <div className="v">{v.toLocaleString('de-DE')}</div></div>
        ))}
      </div>
      <h3>Modell & Sicherung</h3>
      <p className="surface-note">
        Bevölkerungsmodell #{s.population_model?.id} vom {when(s.population_model?.created_at)}
        {' '}({s.population_model?.users} Listen)<br />
        Letzte Sicherung: {s.last_backup || '—'}
      </p>
    </>
  )
}

type ProspectiveCheck = {
  pairs: number; status: 'waiting' | 'passed' | 'failed'; note?: string
  users: Record<string, { pairs: number; rho_before: number; rho_as_shown: number }>
}

type ReportValue = {
  population_model?: { id: number; range_coverage_80_by_size: Record<string, number | null> }
  prospective_check?: ProspectiveCheck
  users?: Record<string, {
    ratings: number; half_life: number | null; range_coverage_80?: number
    prospective: { pairs: number; spearman?: number; bias?: number }
    gate: string | Record<string, { mean: number }>
  }>
}

function Report() {
  const [r, setR] = useState<{ at: string | null; running: boolean; value: unknown } | null>(null)
  const load = useCallback(() => { api.adminReport().then(setR).catch(() => {}) }, [])
  useEffect(() => {
    load()
    const id = window.setInterval(load, 5000)
    return () => window.clearInterval(id)
  }, [load])
  const v = (r?.value ?? null) as ReportValue | null
  return (
    <>
      <p className="surface-note">
        Prüft jedes Konto: Rangkorrelation auf den letzten Bewertungen im Vergleich zu einfachen
        Vergleichswerten, ob die angezeigten Bereiche stimmen, und den Test mit neuen Bewertungen.
      </p>
      <button className="primary" disabled={r?.running}
              onClick={() => api.adminRunReport().then(load)}>
        {r?.running ? 'Läuft …' : 'Bericht erstellen'}
      </button>
      {r?.at && <p className="surface-note">Stand: {when(r.at)}</p>}
      {v?.prospective_check && <ProspectiveNote c={v.prospective_check} />}
      {v?.users && Object.entries(v.users).map(([name, u]) => (
        <div className="admin-row" key={name}>
          <div>
            <b>{name}</b> · {u.ratings} Bewertungen
            {u.half_life ? ` · Halbwertszeit ${u.half_life} J.` : ''}
            {typeof u.gate === 'object' ? (
              <div className="surface-note">
                Modell {u.gate.model.mean} · Community-Wertung {u.gate['community score'].mean}
                {' '}· Durchschnitt + Titel-Bias {u.gate['mean + item bias'].mean}
              </div>
            ) : <div className="surface-note">zu wenig Verlauf für den Test</div>}
            {u.range_coverage_80 !== undefined && (
              <div className="surface-note">Bereich trifft: {Math.round(u.range_coverage_80 * 100)} % (Ziel ~80 %)</div>
            )}
            <div className="surface-note">
              Neue Bewertungen seit Anzeige: {u.prospective.pairs}
              {u.prospective.spearman !== undefined &&
                ` · Rangkorrelation ${u.prospective.spearman} · Abweichung ${u.prospective.bias}`}
            </div>
          </div>
        </div>
      ))}
    </>
  )
}

function ProspectiveNote({ c }: { c: ProspectiveCheck }) {
  return (
    <div className="admin-row">
      <div>
        <b>Test mit neuen Bewertungen</b>{' '}
        <span className={`badge ${c.status === 'failed' ? 'blocked' : c.status === 'passed' ? 'approved' : 'pending'}`}>
          {c.status === 'waiting' ? `${c.pairs} von 20` : `${c.pairs} Bewertungen`}
        </span>
        <div className="surface-note">
          Jede Modelländerung muss auf den Titeln bestehen, die die App empfohlen hat und die
          danach bewertet wurden – Bewertungen, die beim Empfehlen noch nicht existierten.
          {c.status === 'waiting' && ' Bis 20 zusammenkommen, entscheidet nur der Verlaufstest.'}
        </div>
        {Object.entries(c.users).map(([name, u]) => (
          <div className="surface-note" key={name}>
            {name}: {u.pairs} Titel · Rangkorrelation {u.rho_before} (wie angezeigt {u.rho_as_shown})
          </div>
        ))}
      </div>
    </div>
  )
}

function Log() {
  const [rows, setRows] = useState<{ admin: string; action: string; target: string | null; at: string }[] | null>(null)
  useEffect(() => { api.adminLog().then(setRows).catch(() => setRows([])) }, [])
  if (!rows) return <p className="surface-note">Lädt …</p>
  if (!rows.length) return <p className="surface-note">Noch keine Einträge.</p>
  return (
    <div className="admin-table">
      <table>
        <thead><tr><th>Zeit</th><th>Admin</th><th>Aktion</th><th>Konto</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}><td>{when(r.at)}</td><td>{r.admin}</td><td>{r.action}</td><td>{r.target ?? '—'}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
