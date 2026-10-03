// The admin panel. Shown only to admin accounts; the API checks every request
// itself (malrec.admin), this is just the controls. Strings are in i18n.tsx
// under "adm.*".
import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, type AdminSystem, type AdminUser } from './api'
import { BrandMark } from './components'
import { useI18n } from './i18n'

type Tab = 'users' | 'system' | 'report' | 'log'

function useWhen() {
  const { lang } = useI18n()
  return (iso: string | null | undefined): string => {
    if (!iso) return '—'
    return new Date(iso).toLocaleString(lang === 'de' ? 'de-DE' : 'en-GB',
      { dateStyle: 'short', timeStyle: 'short' })
  }
}

export function Admin({ onClose }: { onClose: () => void }) {
  const { t } = useI18n()
  const [tab, setTab] = useState<Tab>('users')
  return (
    <>
      <header className="topbar">
        <div className="wrap topbar-inner">
          <span className="brand"><BrandMark />mal<span>rec</span>&nbsp;· {t('adm.title')}</span>
          <span className="spacer" />
          <button onClick={onClose}>{t('adm.back')}</button>
        </div>
      </header>
      <main className="wrap admin">
        <nav className="tabs" role="tablist">
          {(['users', 'system', 'report', 'log'] as Tab[]).map((k) => (
            <button key={k} role="tab" className="tab" aria-selected={tab === k}
                    onClick={() => setTab(k)}>{t(`adm.tab.${k}`)}</button>
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
  const { t } = useI18n()
  const when = useWhen()
  const [users, setUsers] = useState<AdminUser[] | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState<number | null>(null)

  const load = useCallback(() => {
    api.adminUsers().then(setUsers)
      .catch((e) => setMsg(e instanceof ApiError ? e.message : t('adm.error')))
  }, [t])
  useEffect(() => { load() }, [load])

  async function act(u: AdminUser, action: Parameters<typeof api.adminAction>[1] | 'delete',
                     confirmKey?: string) {
    if (confirmKey && !window.confirm(t(confirmKey, { user: u.mal_username }))) return
    setBusy(u.id); setMsg(null)
    try {
      if (action === 'delete') await api.adminDelete(u.id)
      else await api.adminAction(u.id, action)
      setMsg(`${u.mal_username}: ${t(`adm.done.${action}`)}`)
      load()
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : t('adm.failed'))
    } finally {
      setBusy(null)
    }
  }

  if (!users) return <p className="surface-note">{msg ?? t('Loading…')}</p>
  const pending = users.filter((u) => u.status === 'pending')
  const rest = users.filter((u) => u.status !== 'pending')
  return (
    <>
      {msg && <div className="notice">{msg}</div>}
      <h3>{t('adm.pending', { n: pending.length })}</h3>
      {pending.length === 0 && <p className="surface-note">{t('adm.pending.none')}</p>}
      {pending.map((u) => (
        <div className="admin-row" key={u.id}>
          <div>
            <b>{u.mal_username}</b>{' '}
            <a href={`https://myanimelist.net/profile/${encodeURIComponent(u.mal_username)}`}
               target="_blank" rel="noreferrer">{t('adm.mal_profile')}</a>
            <div className="surface-note">{t('adm.requested', { when: when(u.requested_at) })}</div>
          </div>
          <div className="admin-actions">
            <button className="primary" disabled={busy === u.id} onClick={() => act(u, 'approve')}>
              {t('adm.approve')}
            </button>
            <button disabled={busy === u.id}
                    onClick={() => act(u, 'reject', 'adm.confirm.reject')}>{t('adm.reject')}</button>
          </div>
        </div>
      ))}

      <h3>{t('adm.accounts', { n: rest.length })}</h3>
      <div className="admin-table">
        <table>
          <thead>
            <tr><th>{t('adm.col.account')}</th><th>{t('adm.col.status')}</th>
              <th>{t('adm.col.rated')}</th><th>{t('adm.col.recs')}</th>
              <th>{t('adm.col.login')}</th><th>{t('adm.col.sync')}</th>
              <th>{t('adm.col.actions')}</th></tr>
          </thead>
          <tbody>
            {rest.map((u) => (
              <tr key={u.id}>
                <td>
                  <b>{u.mal_username}</b>{u.is_admin && <span className="chip" style={{ marginLeft: 6 }}>Admin</span>}
                  {u.onboarding === 'running' && <div className="surface-note">{t('adm.onboarding.running')}</div>}
                  {u.onboarding === 'failed' && <div className="surface-note">{t('adm.onboarding.failed')}</div>}
                </td>
                <td><span className={`badge ${u.status}`}>{t(`adm.status.${u.status}`)}</span></td>
                <td>{u.scored} / {u.entries}</td>
                <td>{u.recs}</td>
                <td>{when(u.last_login_at)}</td>
                <td>{when(u.last_sync_at)}</td>
                <td className="admin-actions">
                  {u.status === 'approved' && <>
                    <a className="button-link" href={`/?user=${encodeURIComponent(u.mal_username)}`}>{t('adm.view')}</a>
                    <button disabled={busy === u.id} onClick={() => act(u, 'sync')}>{t('adm.sync')}</button>
                    <button disabled={busy === u.id} onClick={() => act(u, 'rebuild')}>{t('adm.rebuild')}</button>
                  </>}
                  {!u.is_admin && <>
                    <button disabled={busy === u.id || u.sessions === 0}
                            onClick={() => act(u, 'logout', 'adm.confirm.logout')}>
                      {t('adm.logout', { n: u.sessions })}
                    </button>
                    {u.status === 'approved'
                      ? <button disabled={busy === u.id}
                                onClick={() => act(u, 'block', 'adm.confirm.block')}>{t('adm.block')}</button>
                      : <button disabled={busy === u.id} onClick={() => act(u, 'unblock')}>{t('adm.unblock')}</button>}
                    <button disabled={busy === u.id}
                            onClick={() => act(u, 'delete', 'adm.confirm.delete')}>
                      {t('adm.delete')}
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
  const { t, lang } = useI18n()
  const when = useWhen()
  const [s, setS] = useState<AdminSystem | null>(null)
  const [err, setErr] = useState<string | null>(null)
  useEffect(() => {
    api.adminSystem().then(setS).catch((e) => setErr(e instanceof ApiError ? e.message : t('adm.error')))
  }, [t])
  if (!s) return <p className="surface-note">{err ?? t('Loading…')}</p>
  return (
    <>
      <h3>{t('adm.jobs')}</h3>
      {Object.entries(s.jobs).map(([k, j]) => (
        <div className="admin-row" key={k}>
          <div>
            <b>{t(`adm.job.${j.label}`)}</b>{' '}
            <span className={`badge ${j.ok === true ? 'approved' : j.ok === false ? 'blocked' : 'pending'}`}>
              {t(`adm.jobstate.${j.state}`)}
            </span>
            <div className="surface-note">{t('adm.last_run', { when: when(j.last_run) })}</div>
            {j.tail && j.tail.length > 0 && <pre className="log-tail">{j.tail.join('\n')}</pre>}
          </div>
        </div>
      ))}
      <h3>{t('adm.data')}</h3>
      <div className="stat-grid">
        {Object.entries(s.counts).map(([k, v]) => (
          <div className="stat" key={k}><div className="k">{t(`adm.count.${k}`)}</div>
            <div className="v">{v.toLocaleString(lang === 'de' ? 'de-DE' : 'en-GB')}</div></div>
        ))}
      </div>
      <h3>{t('adm.model_backup')}</h3>
      <p className="surface-note">
        {t('adm.population', {
          id: s.population_model?.id ?? '—', when: when(s.population_model?.created_at),
          n: s.population_model?.users ?? '—',
        })}<br />
        {t('adm.last_backup', { line: s.last_backup || '—' })}
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
  const { t } = useI18n()
  const when = useWhen()
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
      <p className="surface-note">{t('adm.report.intro')}</p>
      <button className="primary" disabled={r?.running}
              onClick={() => api.adminRunReport().then(load)}>
        {r?.running ? t('adm.report.running') : t('adm.report.run')}
      </button>
      {r?.at && <p className="surface-note">{t('adm.report.at', { when: when(r.at) })}</p>}
      {v?.prospective_check && <ProspectiveNote c={v.prospective_check} />}
      {v?.users && Object.entries(v.users).map(([name, u]) => (
        <div className="admin-row" key={name}>
          <div>
            <b>{name}</b> · {t('adm.report.ratings', { n: u.ratings })}
            {u.half_life ? ` · ${t('adm.report.half_life', { y: u.half_life })}` : ''}
            {typeof u.gate === 'object' ? (
              <div className="surface-note">
                {t('adm.report.gate', {
                  model: u.gate.model.mean, community: u.gate['community score'].mean,
                  bias: u.gate['mean + item bias'].mean,
                })}
              </div>
            ) : <div className="surface-note">{t('adm.report.no_history')}</div>}
            {u.range_coverage_80 !== undefined && (
              <div className="surface-note">
                {t('adm.report.coverage', { p: Math.round(u.range_coverage_80 * 100) })}
              </div>
            )}
            <div className="surface-note">
              {t('adm.report.new', { n: u.prospective.pairs })}
              {u.prospective.spearman !== undefined &&
                ` · ${t('adm.report.new_fit', { rho: u.prospective.spearman, bias: u.prospective.bias ?? '—' })}`}
            </div>
          </div>
        </div>
      ))}
    </>
  )
}

function ProspectiveNote({ c }: { c: ProspectiveCheck }) {
  const { t } = useI18n()
  return (
    <div className="admin-row">
      <div>
        <b>{t('adm.pro.title')}</b>{' '}
        <span className={`badge ${c.status === 'failed' ? 'blocked' : c.status === 'passed' ? 'approved' : 'pending'}`}>
          {c.status === 'waiting' ? t('adm.pro.waiting', { n: c.pairs }) : t('adm.pro.count', { n: c.pairs })}
        </span>
        <div className="surface-note">
          {t('adm.pro.text')}
          {c.status === 'waiting' && ` ${t('adm.pro.until')}`}
        </div>
        {Object.entries(c.users).map(([name, u]) => (
          <div className="surface-note" key={name}>
            {t('adm.pro.user', { name, n: u.pairs, rho: u.rho_before, shown: u.rho_as_shown })}
          </div>
        ))}
      </div>
    </div>
  )
}

function Log() {
  const { t } = useI18n()
  const when = useWhen()
  const [rows, setRows] = useState<{ admin: string; action: string; target: string | null; at: string }[] | null>(null)
  useEffect(() => { api.adminLog().then(setRows).catch(() => setRows([])) }, [])
  if (!rows) return <p className="surface-note">{t('Loading…')}</p>
  if (!rows.length) return <p className="surface-note">{t('adm.log.empty')}</p>
  const action = (a: string) => {
    const key = `adm.act.${a.replace(/ /g, '_')}`
    const s = t(key)
    return s === key ? a : s
  }
  return (
    <div className="admin-table">
      <table>
        <thead><tr><th>{t('adm.log.time')}</th><th>Admin</th><th>{t('adm.log.action')}</th>
          <th>{t('adm.col.account')}</th></tr></thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}><td>{when(r.at)}</td><td>{r.admin}</td><td>{action(r.action)}</td>
              <td>{r.target ?? '—'}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
