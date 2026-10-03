import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import {
  api, ApiError, type AuthState, type Filters, type ModelInfo, type OnboardStatus,
  type Prefs, type Profile, type Recommendation, type Surface,
} from './api'
import { BrandMark, Card, Controls, Details, Insights, SearchBox, Skeletons } from './components'
import { Admin, AdminBell } from './admin'
import { PrivacyPage } from './privacy'
import { Quiz } from './quiz'
import { captureTogetherToken, Together } from './together'
import {
  I18nContext, initialLang, LANGS, makeT, translateLabel, useI18n, type Lang,
} from './i18n'

const STORED_PREFS = 'malrec.prefs'

function storedPrefs(): Prefs {
  try { return JSON.parse(localStorage.getItem(STORED_PREFS) ?? '{}') as Prefs } catch { return {} }
}
function storePrefs(p: Prefs) {
  try { localStorage.setItem(STORED_PREFS, JSON.stringify(p)) } catch { /* storage blocked */ }
}

/** ?user=name opens another profile - honoured for the admin only (the API
 *  refuses it for everyone else anyway). */
function requestedUser(): string | null {
  return new URLSearchParams(window.location.search).get('user')?.trim() || null
}

const AUTH_ERRORS = ['access_denied', 'cancelled', 'expired', 'profile', 'blocked']

/** Reads ?auth_error= left by the OAuth callback and removes it from the URL. */
function takeAuthError(): string | null {
  const params = new URLSearchParams(window.location.search)
  const code = params.get('auth_error')
  if (!code) return null
  params.delete('auth_error')
  const qs = params.toString()
  window.history.replaceState(null, '', window.location.pathname + (qs ? `?${qs}` : ''))
  return code
}

/** A sign-in failure, explained. The redirect mismatch gets specific
 *  instructions, because MAL's own response to it is a password prompt that
 *  can never succeed. */
function AuthErrorNotice({ code }: { code: string }) {
  const { t } = useI18n()
  const [cfg, setCfg] = useState<{ redirect_uri: string | null; register_at: string } | null>(null)
  useEffect(() => {
    if (code === 'redirect_mismatch') api.authConfig().then(setCfg).catch(() => {})
  }, [code])
  if (code !== 'redirect_mismatch') {
    return <div className="error">{t(AUTH_ERRORS.includes(code) ? `auth.${code}` : 'auth.generic')}</div>
  }
  return (
    <div className="error" style={{ lineHeight: 1.5 }}>
      {t('auth.mismatch1')}{' '}
      <a href={cfg?.register_at ?? 'https://myanimelist.net/apiconfig'} target="_blank"
         rel="noreferrer">myanimelist.net/apiconfig</a>{t('auth.mismatch2')}
      <b> App Redirect URL</b> {t('auth.mismatch3')}
      <code style={{ display: 'block', margin: '8px 0', userSelect: 'all', wordBreak: 'break-all' }}>
        {cfg?.redirect_uri ?? '…'}
      </code>
      {t('auth.mismatch4')}
    </div>
  )
}

/* ------------------------------------------------------------- sign in -- */

function LangPicker({ onChange }: { onChange: (l: Lang) => void }) {
  const { lang, t } = useI18n()
  return (
    <select className="lang" value={lang} aria-label={t('Language')} title={t('Language')}
            onChange={(e) => onChange(e.target.value as Lang)}>
      {LANGS.map((l) => <option key={l.code} value={l.code}>{l.label}</option>)}
    </select>
  )
}

function AccountMenu({ username, picture, viewing, children }:
  { username: string; picture?: string | null; viewing: string | null; children: ReactNode }) {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [picFailed, setPicFailed] = useState(false)
  useEffect(() => { setPicFailed(false) }, [picture])
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])
  return (
    <div className="account" ref={ref}>
      <button className="bar-btn account-btn" aria-haspopup="menu" aria-expanded={open}
              onClick={() => setOpen((o) => !o)}>
        {picture && !picFailed
          ? <img className="avatar" src={picture} alt="" referrerPolicy="no-referrer"
                 onError={() => setPicFailed(true)} />
          : <span className="avatar" aria-hidden="true">{username.slice(0, 1).toUpperCase()}</span>}
        <span className="bar-label">{viewing ?? username}</span>
        <span aria-hidden="true" className="caret">▾</span>
      </button>
      {open && (
        <div className="menu" role="menu" onClick={(e) => {
          if ((e.target as HTMLElement).closest('button, a')) setOpen(false)
        }}>
          <div className="menu-head">
            {t('Signed in as')} <b>{username}</b>
            {viewing && <div>{t('viewing')} <b>{viewing}</b></div>}
          </div>
          {children}
        </div>
      )}
    </div>
  )
}

function Footer() {
  const { t } = useI18n()
  return (
    <footer className="credits" aria-label="credits">
      <span className="credits-line" aria-hidden="true" />
      <p>
        {t('credits.pre')}{' '}
        <span className="credits-heart" aria-label="love">💙</span>{' '}
        {t('credits.by')}{' '}
        <a className="credits-name" href="https://github.com/knarxy" target="_blank" rel="noopener noreferrer">dennis_jar</a>
        {' '}{t('credits.and')}{' '}
        <span className="credits-claude">Claude Opus 5.5</span>
      </p>
      <div className="credits-links">
      <a className="credits-source" href="https://github.com/knarxy/malrec" target="_blank"
         rel="noopener noreferrer">
        <svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true">
          <path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
        </svg>
        {t('credits.source')}
      </a>
      <a className="credits-source" href="/privacy">{t('privacy.link')}</a>
      </div>
    </footer>
  )
}

function SignIn({ authError, onLang }: { authError: string | null; onLang: (l: Lang) => void }) {
  const { t } = useI18n()
  return (
    <div className="center">
      <div className="panel">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <h1 className="brand" style={{ flex: 1, fontSize: 24 }}><BrandMark size={36} />mal<span>rec</span></h1>
          <LangPicker onChange={onLang} />
        </div>
        <p className="sub">{t('tagline')}</p>
        <a className="mal-btn button-link" href={api.loginUrl('/')}>
          {t('Sign in with MyAnimeList')}
        </a>
        <p className="sub" style={{ margin: '10px 0 0', fontSize: 12.5 }}>{t('signin.benefit')}</p>
        {authError && <AuthErrorNotice code={authError} />}
        <p className="sub" style={{ margin: '18px 0 0', fontSize: 13 }}>{t('signin.approval')}</p>
        <p className="sub" style={{ margin: '12px 0 0', fontSize: 12.5 }}>
          <a href="/privacy">{t('privacy.link')}</a>
        </p>
      </div>
    </div>
  )
}

/** Signed in, but the admin has not approved the account yet. */
function Pending({ username, onLang, onLogout, hasInvite }: {
  username: string; onLang: (l: Lang) => void; onLogout: () => void; hasInvite: boolean
}) {
  const { t } = useI18n()
  return (
    <div className="center">
      <div className="panel">
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <h1 style={{ flex: 1 }}>{t('pending.title')}</h1>
          <LangPicker onChange={onLang} />
        </div>
        <p className="sub">{t('pending.text', { user: username })}</p>
        {hasInvite && <p className="sub">{t('together.link_wait')}</p>}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginTop: 8 }}>
          <button onClick={onLogout}>{t('Sign out')}</button>
          <a href="/privacy" style={{ fontSize: 13 }}>{t('privacy.menu')}</a>
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------ onboarding -- */

function Onboarding({
  user, status, onReady, onReset,
}: {
  user: string
  status: OnboardStatus | null
  onReady: () => void
  onReset: () => void
}) {
  const { t } = useI18n()
  const total = status?.step_total ?? 6
  const index = status?.step_index ?? 0
  const pct = status?.state === 'done' ? 100 : Math.round((index / total) * 100)
  const steps = [
    'Reading your MyAnimeList profile',
    'Finding your first recommendations',
    'Catching up on the catalogue',
    'Learning what you have watched',
    'Working out your taste',
    'Building your recommendations',
  ]

  useEffect(() => {
    if (status?.state === 'done' || status?.ready) onReady()
  }, [status, onReady])

  if (status?.state === 'failed') {
    return (
      <div className="center">
        <div className="panel">
          <h1>{t('That did not work')}</h1>
          <p className="sub">{t('Setting up {user} failed.', { user })}</p>
          <div className="error">{status.error ?? t('Unknown error')}</div>
          <p className="sub" style={{ marginTop: 16, fontSize: 13 }}>{t('onboard.fail_hint')}</p>
          <button style={{ marginTop: 8 }} onClick={onReset}>{t('Sign out')}</button>
        </div>
      </div>
    )
  }

  return (
    <div className="center">
      <div className="panel">
        <h1>{t('Setting up {user}', { user })}</h1>
        <p className="sub">{t('onboard.sub')}</p>
        <div className="progress"><div style={{ width: `${pct}%` }} /></div>
        <ul className="steps">
          {steps.map((s, i) => (
            <li key={s} className={i < index ? 'done' : i === index ? 'active' : ''}>
              <span className="dot" />{t(s)}
            </li>
          ))}
        </ul>

      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ app -- */

export default function App() {
  const [lang, setLangState] = useState<Lang>(initialLang)
  const i18n = useMemo(() => ({
    lang, t: makeT(lang), label: (x: string) => translateLabel(lang, x),
  }), [lang])
  useEffect(() => { document.documentElement.lang = lang }, [lang])
  if (window.location.pathname.replace(/\/+$/, '') === '/privacy') {
    const pick = (l: Lang) => {
      setLangState(l)
      try { localStorage.setItem('malrec.lang', l) } catch { /* storage blocked */ }
    }
    return (
      <I18nContext.Provider value={i18n}>
        <PrivacyPage onLang={pick} />
        <Footer />
      </I18nContext.Provider>
    )
  }
  return (
    <I18nContext.Provider value={i18n}>
      <Main lang={lang} setLangState={setLangState} />
    </I18nContext.Provider>
  )
}

function Main({ lang, setLangState }: { lang: Lang; setLangState: (l: Lang) => void }) {
  const { t } = useI18n()
  const [user, setUser] = useState<string | null>(null)
  const [adminView, setAdminView] = useState(false)
  const [status, setStatus] = useState<OnboardStatus | null>(null)
  const [ready, setReady] = useState(false)

  const [surfaces, setSurfaces] = useState<Surface[]>([])
  const [active, setActive] = useState('safe_bets')
  const [items, setItems] = useState<Recommendation[] | null>(null)
  const [note, setNote] = useState('')
  const [profile, setProfile] = useState<Profile | null>(null)
  const [model, setModel] = useState<ModelInfo | null>(null)
  const [detail, setDetail] = useState<Recommendation | null>(null)
  const [insights, setInsights] = useState(false)
  const [quiz, setQuiz] = useState(false)
  const [linkToken, setLinkToken] = useState<string | null>(captureTogetherToken)
  const [together, setTogether] = useState(linkToken !== null)
  const [error, setError] = useState<string | null>(null)
  const [auth, setAuth] = useState<AuthState | null>(null)
  const [authError] = useState<string | null>(takeAuthError)
  const [prefs, setPrefs] = useState<Prefs>(storedPrefs)
  const [reloadKey, setReloadKey] = useState(0)
  const [refresh, setRefresh] = useState<string | null>(null)   // sync / rebuild notice
  const [syncWait, setSyncWait] = useState(0)
  const pollRef = useRef<number | undefined>(undefined)

  const filters: Filters = prefs.filters ?? {}

  // Who is signed in (if anyone). A signed-in user with nothing chosen yet
  // lands on their own recommendations, with the preferences saved to their
  // account (language, filters) taking over from this browser's.
  useEffect(() => {
    api.authMe()
      .then(async (a) => {
        setAuth(a)
        if (a.signed_in && a.username) {
          if (a.status === 'approved') setUser((a.is_admin && requestedUser()) || a.username)
          try {
            const p = await api.getPrefs()
            if (Object.keys(p).length) {
              setPrefs((cur) => { const m = { ...cur, ...p }; storePrefs(m); return m })
              if (p.lang) { setLangState(p.lang); localStorage.setItem('malrec.lang', p.lang) }
            }
            if (a.status === 'approved') {
              const st = await api.syncStatus()
              setSyncWait(st.next_allowed_in)
            }
          } catch { /* prefs are a convenience */ }
        }
      })
      .catch(() => setAuth({ signed_in: false }))
  }, [setLangState])

  const savePrefs = useCallback((patch: Prefs) => {
    setPrefs((cur) => { const m = { ...cur, ...patch }; storePrefs(m); return m })
    if (auth?.signed_in) api.putPrefs(patch).catch(() => {})
  }, [auth])

  const changeLang = useCallback((l: Lang) => {
    setLangState(l)
    try { localStorage.setItem('malrec.lang', l) } catch { /* storage blocked */ }
    savePrefs({ lang: l })
  }, [savePrefs, setLangState])

  // countdown for the sync button
  useEffect(() => {
    if (syncWait <= 0) return
    const id = window.setTimeout(() => setSyncWait((w) => Math.max(0, w - 1)), 1000)
    return () => window.clearTimeout(id)
  }, [syncWait])

  /** Poll the background sync/rebuild; reload the lists when it is done. */
  const watchRefresh = useCallback((label: string) => {
    setRefresh(label)
    window.clearInterval(pollRef.current)
    const started = Date.now()
    pollRef.current = window.setInterval(async () => {
      try {
        const st = await api.syncStatus()
        setSyncWait(st.next_allowed_in)
        if (st.refresh === 'idle' && Date.now() - started > 1500) {
          window.clearInterval(pollRef.current)
          setRefresh(null)
          setReloadKey((k) => k + 1)
          // the sync also re-reads the MAL profile picture
          api.authMe().then((a) => { if (a.signed_in) setAuth(a) }).catch(() => {})
        } else if (st.refresh === 'failed' || st.refresh === 'timeout') {
          window.clearInterval(pollRef.current)
          setRefresh(st.refresh === 'failed'
            ? t('sync.failed', { e: st.refresh_error ?? '' }) : t('sync.timeout'))
          window.setTimeout(() => setRefresh(null), 8000)
        }
      } catch { /* keep polling */ }
    }, 2000)
  }, [t])
  useEffect(() => () => window.clearInterval(pollRef.current), [])

  const startSync = useCallback(async () => {
    try {
      const r = await api.sync()
      setSyncWait(r.next_allowed_in)
      watchRefresh(t('Syncing…'))
    } catch (e) {
      if (e instanceof ApiError && e.status === 429) {
        setSyncWait(Number(e.detail?.retry_after ?? 60))
      }
    }
  }, [watchRefresh, t])

  const onRated = useCallback(() => {
    watchRefresh(t('Updating your recommendations…'))
  }, [watchRefresh, t])

  const logout = useCallback(async () => {
    try { await api.logout() } finally {
      window.history.replaceState(null, '', window.location.pathname)
      setAuth({ signed_in: false }); setUser(null); setReady(false); setStatus(null)
      setItems(null); setProfile(null)
    }
  }, [])

  // Poll while a profile is being built. The first recommendations appear
  // as soon as they exist; polling continues until the full build is done,
  // then the lists reload with the final ones.
  const building = status?.state === 'running'
  useEffect(() => {
    if (!user || (ready && !building)) return
    let alive = true
    const tick = async () => {
      try {
        const s = await api.onboardStatus(user)
        if (!alive) return
        setStatus((prev) => {
          if (prev?.state === 'running' && s.state === 'done') setReloadKey((k) => k + 1)
          return s
        })
        if (s.ready) setReady(true)
      } catch {
        /* keep polling; the API may still be starting */
      }
    }
    tick()
    const id = window.setInterval(tick, 2500)
    return () => { alive = false; window.clearInterval(id) }
  }, [user, ready, building])

  // Once ready, load the shell.
  useEffect(() => {
    if (!user || !ready) return
    api.surfaces().then(setSurfaces).catch(() => {})
    api.profile(user).then(setProfile).catch(() => {})
    api.model(user).then(setModel).catch(() => setModel(null))
  }, [user, ready])

  // Load whichever surface is selected, narrowed by the filters.
  const filterKey = JSON.stringify(filters)
  useEffect(() => {
    if (!user || !ready) return
    let alive = true
    setItems(null)
    setError(null)
    api.recommendations(user, active, 40, JSON.parse(filterKey))
      .then((r) => { if (alive) { setItems(r.items); setNote(r.description) } })
      .catch((e) => alive && setError(e instanceof ApiError ? e.message : t('Failed to load')))
    return () => { alive = false }
  }, [user, ready, active, filterKey, reloadKey, t])

  if (auth === null) {
    return (
      <div className="center">
        <div className="panel" style={{ textAlign: 'center' }}>
          <h1 className="brand" style={{ fontSize: 24, justifyContent: 'center' }}><BrandMark size={36} />mal<span>rec</span></h1>
        </div>
      </div>
    )
  }
  if (!auth.signed_in) return <SignIn authError={authError} onLang={changeLang} />
  if (auth.status !== 'approved' || !user) {
    return <Pending username={auth.username ?? ''} onLang={changeLang} onLogout={logout}
                    hasInvite={linkToken !== null} />
  }
  if (adminView && auth.is_admin) return <Admin onClose={() => setAdminView(false)} />
  if (!ready) {
    // Until the first status response lands we do not know whether this is a
    // new profile or a returning one, so show a neutral wait rather than
    // flashing "Setting up…" at someone whose data is already built.
    if (status === null) {
      return (
        <div className="center">
          <div className="panel" style={{ textAlign: 'center' }}>
            <h1 className="brand" style={{ fontSize: 24, justifyContent: 'center' }}><BrandMark size={36} />mal<span>rec</span></h1>
            <p className="sub" style={{ margin: 0 }}>{t('Checking {user}…', { user })}</p>
          </div>
        </div>
      )
    }
    return (
      <Onboarding
        user={user}
        status={status}
        onReady={() => setReady(true)}
        onReset={logout}
      />
    )
  }

  // Queueing writes to a MyAnimeList account, so it only appears on your own
  // recommendations, and only once you are signed in.
  const isOwn = !!auth?.signed_in && auth.username?.toLowerCase() === user.toLowerCase()

  return (
    <>
      <header className="topbar">
        <div className="wrap topbar-inner">
          <span className="brand"><BrandMark />mal<span>rec</span></span>
          <SearchBox onOpen={(id) => {
            api.animeFor(id, user).then(setDetail).catch(() => {})
          }} />
          <span className="spacer" />
          {isOwn && (
            <button className="bar-btn" onClick={startSync} disabled={syncWait > 0 || refresh !== null}
                    title={syncWait > 0 ? t('sync.wait', { s: syncWait }) : t('sync.title')}>
              <span aria-hidden="true">↻</span>
              <span className="bar-label">
                {refresh !== null && refresh === t('Syncing…') ? t('Syncing…')
                  : syncWait > 0 ? `${t('Sync with MAL')} · ${syncWait}s` : t('Sync with MAL')}
              </span>
            </button>
          )}
          {isOwn && <button className="bar-btn bar-extra" onClick={() => setQuiz(true)}>{t('Rate titles')}</button>}
          {isOwn && <button className="bar-btn bar-extra" onClick={() => setTogether(true)}>{t('Watch together')}</button>}
          <button className="bar-btn bar-extra" onClick={() => setInsights(true)} disabled={!profile}>{t('Your taste')}</button>
          {auth.is_admin && <AdminBell onOpenPanel={() => setAdminView(true)} />}
          <AccountMenu username={auth.username ?? ''} picture={auth.picture} viewing={isOwn ? null : user}>
            {isOwn && <button className="menu-extra" onClick={() => setQuiz(true)}>{t('Rate titles')}</button>}
            {isOwn && <button className="menu-extra" onClick={() => setTogether(true)}>{t('Watch together')}</button>}
            <button className="menu-extra" onClick={() => setInsights(true)} disabled={!profile}>{t('Your taste')}</button>
            {auth.is_admin && <button onClick={() => setAdminView(true)}>{t('adm.menu')}</button>}
            {auth.is_admin && !isOwn && <a className="menu-link" href="/">{t('back.own')}</a>}
            <label className="menu-lang">{t('Language')} <LangPicker onChange={changeLang} /></label>
            <a className="menu-link" href="/privacy">{t('privacy.menu')}</a>
            <button onClick={logout}>{t('Sign out')}</button>
          </AccountMenu>
        </div>
      </header>

      <main className="wrap">
        <nav className="tabs" role="tablist">
          {surfaces.map((s) => (
            <button
              key={s.name}
              role="tab"
              className="tab"
              aria-selected={s.name === active}
              onClick={() => setActive(s.name)}
            >
              {t(`tab.${s.name}`)}
            </button>
          ))}
        </nav>

        {authError && <AuthErrorNotice code={authError} />}
        {building && (
          <div className="notice">{t('refining', { step: t(status?.step ?? '') })}</div>
        )}
        {refresh && <div className="notice">{refresh}</div>}
        {isOwn && profile && profile.scored < 60 && !refresh && (
          <div className="notice">
            {t('quiz.banner')}{' '}
            <button className="button-link" onClick={() => setQuiz(true)}>{t('Rate titles')}</button>
          </div>
        )}
        <div className="surface-head">
          <h1>{t(`tab.${active}`)}</h1>
          <p className="surface-note">{lang === 'en' ? note : t(`desc.${active}`)}</p>
        </div>
        <Controls filters={filters} onFilters={(f) => savePrefs({ filters: f })} />

        {error && <div className="error">{error}</div>}
        {items === null && !error && <Skeletons />}
        {items?.length === 0 && (
          <p className="empty">
            {Object.keys(filters).some((k) => {
              const v = filters[k as keyof Filters]
              return Array.isArray(v) ? v.length > 0 : v !== undefined
            }) ? t('empty.filtered')
              : active === 'next_up' ? t('empty.next_up')
              : active === 'plan_to_watch' ? t('empty.plan_to_watch')
              : active === 'coming_soon' ? t('empty.coming_soon')
              : `${t('Nothing here yet.')} ${t('Try another tab.')}`}
          </p>
        )}
        {items && items.length > 0 && (
          <div className="grid">
            {items.map((it) => (
              <Card
                key={it.mal_id}
                item={it}
                surface={active}
                user={user}
                onOpen={setDetail}
                canQueue={isOwn}
                onRated={onRated}
              />
            ))}
          </div>
        )}
      </main>

      {detail && (
        <Details item={detail} user={user} surface={active} canRate={isOwn} onClose={() => setDetail(null)} onRated={onRated} />
      )}
      {quiz && <Quiz onClose={() => setQuiz(false)} onRated={onRated} />}
      {together && <Together onClose={() => setTogether(false)} linkToken={linkToken}
                             onLinkDone={() => setLinkToken(null)} />}
      {insights && profile && (
        <Insights profile={profile} model={model} onClose={() => setInsights(false)} />
      )}
      <Footer />
    </>
  )
}
