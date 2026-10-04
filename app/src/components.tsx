import { type CSSProperties, type ReactNode, useEffect, useId, useRef, useState } from 'react'
import {
  api, ApiError, type Breakdown, type Filters, type Learning, type ModelInfo, type Profile,
  type Recommendation, type SearchHit, type SimilarHit,
} from './api'
import { Likely, Why } from './explain'
import { useI18n } from './i18n'

const SEASONS: Record<string, [string, string]> = {
  winter: ['Winter', 'Winter'], spring: ['Spring', 'Frühling'],
  summer: ['Summer', 'Sommer'], fall: ['Fall', 'Herbst'],
}

function useSeason() {
  const { lang } = useI18n()
  return (r: Recommendation): string | null => {
    if (!r.season_year) return null
    const s = r.season ? SEASONS[r.season]?.[lang === 'de' ? 1 : 0] ?? r.season : null
    return s ? `${s} ${r.season_year}` : `${r.season_year}`
  }
}

function unaired(r: Recommendation): boolean {
  return (r.reasons ?? []).some((x) => x.kind === 'coming')
    || (r as unknown as { status?: string }).status === 'not_yet_aired'
}

/** 1-10 score picker that writes to MyAnimeList. */
export function RateControl({
  item, onRated,
}: {
  item: Recommendation
  onRated: (score: number, status: string | null) => void
}) {
  const { t } = useI18n()
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const score = item.list_score ?? 0

  async function set(v: number) {
    setBusy(true)
    setMsg(null)
    try {
      const r = await api.rate(item.mal_id, v)
      onRated(r.list_score, r.list_status)
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : t('Could not reach MyAnimeList.'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="rate" title={t('rate.title')}>
      <select
        value={score || ''}
        disabled={busy}
        aria-label={t('Rate')}
        onChange={(e) => set(Number(e.target.value))}
      >
        <option value="" disabled>{score ? t('Rated {n}', { n: score }) : t('Seen it? Rate it')}</option>
        {[10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map((v) => <option key={v} value={v}>{v}</option>)}
        {score > 0 && <option value={0}>{t('Clear score')}</option>}
      </select>
      {msg && <div className="card-msg">{msg}</div>}
    </div>
  )
}

export function Card({
  item, surface, user, onOpen, canQueue, onRated,
}: {
  item: Recommendation
  surface: string
  user: string
  onOpen: (r: Recommendation) => void
  /** true only when signed in with MyAnimeList and looking at your own list */
  canQueue: boolean
  onRated: (mal_id: number) => void
}) {
  const { t } = useI18n()
  const season = useSeason()
  const [dismissed, setDismissed] = useState(false)
  const [listStatus, setListStatus] = useState<string | null>(item.list_status ?? null)
  const [score, setScore] = useState<number>(item.list_score ?? 0)
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<string | null>(null)

  async function dismiss() {
    setBusy(true)
    try {
      if (dismissed) {
        await api.undoFeedback(user, item.mal_id)
        setDismissed(false)
      } else {
        await api.feedback(user, item.mal_id, 'not_interested', surface)
        setDismissed(true)
      }
    } finally {
      setBusy(false)
    }
  }

  async function toggleQueue() {
    setBusy(true)
    setMessage(null)
    try {
      const r = listStatus === 'plan_to_watch'
        ? await api.unqueue(item.mal_id, surface)
        : await api.queue(item.mal_id, surface)
      setListStatus(r.list_status)
    } catch (e) {
      setMessage(e instanceof ApiError ? e.message : t('Could not reach MyAnimeList.'))
    } finally {
      setBusy(false)
    }
  }

  const planned = listStatus === 'plan_to_watch'
  // Anything else on the list (watching, completed, ...) is not ours to touch.
  const otherStatus = listStatus && !planned ? t(`status.${listStatus}`) : null
  const eps = item.media_type === 'movie' ? t('Film')
    : item.num_episodes ? `${item.num_episodes} ${t('ep')}` : null
  const bits = [season(item), eps, item.mal_studios?.[0]].filter(Boolean)

  return (
    <article className={`card${dismissed ? ' dismissed' : ''}${score ? ' rated' : ''}`}>
      <div className="poster" onClick={() => onOpen(item)} style={{ cursor: 'pointer' }}>
        {item.picture_large || item.picture_medium ? (
          <img src={item.picture_large ?? item.picture_medium!} alt={item.title} loading="lazy" />
        ) : null}
        <span className="rank">#{item.rank}</span>
        {/* an unaired title has no community score or ratings yet, so the
            model's number for it means little; coming_soon is ordered by date */}
        {!unaired(item) && (
          <span className="fit" title={t('predicted score for you')}>
            <Sparkle />{item.predicted_score.toFixed(1)}
          </span>
        )}
      </div>
      <div className="card-body">
        <h3 onClick={() => onOpen(item)} style={{ cursor: 'pointer' }}>
          {item.title_en || item.title}
        </h3>
        <div className="meta">
          {bits.map((b, i) => <span key={i}>{b}</span>)}
          {item.mal_mean && <span>★ {item.mal_mean.toFixed(2)}</span>}
        </div>
        {item.also_in && (
          <span className="also-in"><Sparkle />{t('card.also_in', { tab: t(`tab.${item.also_in}`) })}</span>
        )}
        {!unaired(item) && <Likely item={item} />}
        <Why reasons={item.reasons ?? []} />
        {message && <div className="card-msg">{message}</div>}
        <div className="card-actions">
          {canQueue && !otherStatus && (
            <button
              className={planned ? 'planned' : 'plan'}
              onClick={toggleQueue}
              disabled={busy}
              title={planned ? t('queue.remove') : t('queue.add')}
            >
              {busy ? '…' : planned ? t('✓ Planned') : t('+ Plan to watch')}
            </button>
          )}
          {canQueue && otherStatus && (
            <button disabled title={t('Already on your MyAnimeList')}>
              {t('On list: {s}', { s: otherStatus })}
            </button>
          )}
          <button className="quiet" onClick={dismiss} disabled={busy}>
            {dismissed ? t('Undo') : t('Not for me')}
          </button>
        </div>
        {canQueue && !unaired(item) && (
          <RateControl
            item={{ ...item, list_score: score }}
            onRated={(s, st) => { setScore(s); setListStatus(st); onRated(item.mal_id) }}
          />
        )}
      </div>
    </article>
  )
}

export function Skeletons({ n = 8 }: { n?: number }) {
  return (
    <div className="grid">
      {Array.from({ length: n }, (_, i) => <div className="skeleton" key={i} />)}
    </div>
  )
}

function Signed({ v }: { v: number }) {
  return <span className={v >= 0 ? 'pos' : 'neg'}>{v >= 0 ? '+' : '−'}{Math.abs(v).toFixed(2)}</span>
}

const SPARKLE = 'M30 11c1.6 11.7 6.3 16.4 18 18-11.7 1.6-16.4 6.3-18 18-1.6-11.7-6.3-16.4-18-18 11.7-1.6 16.4-6.3 18-18z'

/** The app icon: the sparkle on its dark tile. */
export function BrandMark({ size = 30 }: { size?: number }) {
  const id = `bm${useId().replace(/:/g, '')}`
  return (
    <svg className="brand-mark" width={size} height={size} viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0.15" y1="0.1" x2="0.85" y2="0.95">
          <stop offset="0" stopColor="#7cb3ff" /><stop offset="1" stopColor="#57d9a3" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill="#11141a" />
      <rect x="0.75" y="0.75" width="62.5" height="62.5" rx="14.25" fill="none" stroke="#2a2f3a" strokeWidth="1.5" />
      <path d={SPARKLE} fill={`url(#${id})`} />
      <path d="M47 39c.6 4.4 2.4 6.2 6.8 6.8-4.4.6-6.2 2.4-6.8 6.8-.6-4.4-2.4-6.2-6.8-6.8 4.4-.6 6.2-2.4 6.8-6.8z" fill="#e8eaf0" />
    </svg>
  )
}

/** The bare sparkle in the current text colour, for badges. */
export function Sparkle() {
  return <svg viewBox="10 9 40 40" aria-hidden="true"><path d={SPARKLE} fill="currentColor" /></svg>
}

function Arrow({ up }: { up: boolean }) {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke={up ? 'var(--good)' : 'var(--warn)'}
         strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={up ? 'M12 19V5M6 11l6-6 6 6' : 'M12 5v14M6 13l6 6 6-6'} />
    </svg>
  )
}

type Side = { up: { label: string }[]; down: { label: string }[] }

function VoiceReasons({ side, extra }: { side: Side | null | undefined; extra?: ReactNode }) {
  const { label } = useI18n()
  return (
    <ul>
      {side?.up.slice(0, 3).map((d) => (
        <li key={`u${d.label}`}><span className="sgn-up">+</span><span>{label(d.label)}</span></li>
      ))}
      {extra}
      {side?.down.slice(0, 2).map((d) => (
        <li key={`d${d.label}`}><span className="sgn-down">−</span><span>{label(d.label)}</span></li>
      ))}
    </ul>
  )
}

/** "Why this?" as two voices: what your own ratings say, what viewers with
 *  your taste say, how the two are weighed into the shown score, and what
 *  moves the title within its list. */
function WhyPanel({ user, mal_id, surface, rank }: {
  user: string; mal_id: number; surface: string; rank: number
}) {
  const { t } = useI18n()
  const [b, setB] = useState<Breakdown | null | undefined>(undefined)
  const onList = rank > 0

  useEffect(() => {
    let alive = true
    setB(undefined)
    // rank 0 = opened from search, where the general calibration applies
    api.explain(user, mal_id, surface, onList)
      .then((r) => alive && setB(r)).catch(() => alive && setB(null))
    return () => { alive = false }
  }, [user, mal_id, surface, onList])

  if (b === undefined) return <p className="surface-note">{t('Loading…')}</p>
  if (b === null) return <p className="surface-note">{t('why.none')}</p>
  const p = b.personal !== null ? Math.round(b.personal_share * 100) : 0
  const liked = b.because_you_liked.filter((l) => l.your_score > b.your_mean).slice(0, 2)
  const trim = b.shown - b.raw
  const r = b.ranking
  const adjustments = r ? [
    { key: 'relevance', v: r.relevance_bonus },
    { key: 'novelty', v: r.novelty_bonus },
    { key: 'memory', v: r.memory_bonus ?? 0 },
    { key: 'popularity', v: r.popularity_bonus ?? 0 },
    ...(r.risk_parts
      ? Object.entries(r.risk_parts).map(([k, v]) => ({ key: `risk_${k}`, v }))
      : [{ key: 'risk', v: r.risk_penalty ?? 0 }]),
  ].filter((a) => Math.abs(a.v) >= 0.005).sort((x, y) => Math.abs(y.v) - Math.abs(x.v)) : []

  const fans = liked.length > 0 && (
    <li key="fans"><span className="sgn-up">+</span>
      <span>{t('why2.fans')}{' '}
        {liked.map((l, k) => (
          <span key={l.title}><b>{l.title}</b> ({l.your_score}){k < liked.length - 1 ? ` ${t('and')} ` : ''}</span>
        ))}
      </span>
    </li>
  )

  return (
    <div className="whypanel">
      <div className={`voices${b.personal === null ? ' single' : ''}`}>
        {b.personal !== null && (
          <div className="voice you">
            <div className="who-says">{t('why2.you')}</div>
            <div className="num">{b.personal.toFixed(2)}</div>
            <VoiceReasons side={b.sides?.personal} />
          </div>
        )}
        <div className="voice them">
          <div className="who-says">{t('why2.them')}</div>
          <div className="num">{b.population.toFixed(2)}</div>
          <VoiceReasons side={b.sides?.population} extra={fans} />
        </div>
      </div>

      <div className="weigh">
        {b.personal !== null ? (
          <>
            <div className="weigh-bar">
              <div className="you" style={{ width: `${p}%` }} />
              <div className="them" style={{ width: `${100 - p}%` }} />
            </div>
            <div className="weigh-legend">
              <span>{t('why2.w_you', { p })}</span><span>{t('why2.w_them', { q: 100 - p })}</span>
            </div>
            <p>{t('why2.mixed')} <b>{b.raw.toFixed(2)}</b>. {t('why2.more')}</p>
          </>
        ) : (
          <p>{t('why2.only_them')}</p>
        )}
        <p>
          {Math.abs(trim) < 0.05 ? t('why2.kept')
            : trim < 0 ? t('why2.trim_down') : t('why2.trim_up')}{' '}
          <b>{b.shown.toFixed(2)}</b>{' '}
          <span style={{ color: 'var(--muted)' }}>({t('why2.mean', { m: b.your_mean.toFixed(2) })})</span>
        </p>
      </div>

      {r && adjustments.length > 0 && (
        <>
          <h3 className="section-h">
            {t('why2.place', { n: rank, s: t(`tab.${surface}`) })}
          </h3>
          <p className="why-text" style={{ margin: '0 0 8px', color: 'var(--muted)', fontSize: 13 }}>
            {t('why2.place_sub')}
          </p>
          {adjustments.map((a) => (
            <div className="adj" key={a.key}>
              <Arrow up={a.v >= 0} />
              <span>{a.key.startsWith('risk_') ? t(`why2.${a.key}`)
                : t(`why2.${a.key}_${a.v >= 0 ? 'up' : 'down'}`)}</span>
              <b className={a.v >= 0 ? 'pos' : 'neg'}>{a.v >= 0 ? '+' : '−'}{Math.abs(a.v).toFixed(2)}</b>
            </div>
          ))}
          {r.order_score !== undefined && (
            <div className="adj total">
              <span />
              <span>{t('why2.order', { p: (r.predicted ?? b.shown).toFixed(2) })}</span>
              <b>{r.order_score.toFixed(2)}</b>
            </div>
          )}
        </>
      )}

      <details className="tech">
        <summary>{t('why2.tech')}</summary>
        <p>
          {t('why.population')}: {b.population.toFixed(2)}
          {b.personal !== null && <> · {t('why.personal')}: {b.personal.toFixed(2)} · {t('why.weights', { p, q: 100 - p })}</>}
          <br />
          {t('why.calibration')}: {b.calibration.slope.toFixed(2)} × {b.raw.toFixed(2)}
          {b.calibration.intercept >= 0 ? ' + ' : ' − '}{Math.abs(b.calibration.intercept).toFixed(2)} = {b.shown.toFixed(2)}
          {r && <><br />{t('why.relevance')}: {r.relevance_z >= 0 ? '+' : ''}{r.relevance_z.toFixed(1)} σ → <Signed v={r.relevance_bonus} /></>}
        </p>
      </details>
    </div>
  )
}

/** Slide-over with the full synopsis, the "why", and tag-space neighbours. */
export function Details({
  item, user, surface, canRate, onClose, onRated,
}: {
  item: Recommendation
  user: string
  surface: string
  canRate: boolean
  onClose: () => void
  onRated: (mal_id: number) => void
}) {
  const { t, label } = useI18n()
  const [similar, setSimilar] = useState<SimilarHit[] | null>(null)
  const [score, setScore] = useState(item.list_score ?? 0)

  useEffect(() => {
    let alive = true
    api.similar(item.mal_id).then((s) => alive && setSimilar(s)).catch(() => {})
    return () => { alive = false }
  }, [item.mal_id])

  const predicted = Number.isFinite(item.predicted_score) && !unaired(item) ? item.predicted_score : null
  const deg = predicted === null ? 0 : Math.max(0, Math.min(360, ((predicted - 1) / 9) * 360))

  return (
    <div className="drawer-bg" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>{t('Close')}</button>
        <h2>{item.title_en || item.title}</h2>
        {item.title_en && item.title !== item.title_en && (
          <p className="surface-note" style={{ margin: '2px 0 0' }}>{item.title}</p>
        )}

        <div className="drawer-hero">
          <div className="ring" style={{ '--deg': `${deg}deg` } as CSSProperties}>
            <span>{predicted === null ? '—' : predicted.toFixed(1)}</span>
          </div>
          <div>
            <div className="k">{t('Predicted for you')}</div>
            {!unaired(item) && <Likely item={item} />}
          </div>
        </div>

        <div className="stat-grid">
          <div className="stat">
            <div className="k">{t('MAL community')}</div>
            <div className="v">{item.mal_mean?.toFixed(2) ?? '—'}</div>
          </div>
          <div className="stat">
            <div className="k">{t('Popularity rank')}</div>
            <div className="v">#{item.mal_popularity ?? '—'}</div>
          </div>
          <div className="stat">
            <div className="k">{t('Format')}</div>
            <div className="v">
              {item.media_type === 'movie' ? t('Film')
                : `${item.media_type?.toUpperCase() ?? '—'}${item.num_episodes
                  ? ` · ${item.num_episodes} ${t('ep')}` : ''}`}
            </div>
          </div>
        </div>

        {canRate && !unaired(item) && (
          <RateControl item={{ ...item, list_score: score }}
                       onRated={(s) => { setScore(s); onRated(item.mal_id) }} />
        )}

        <Why reasons={item.reasons ?? []} />

        <h3 className="section-h">{t('Why this?')}</h3>
        <WhyPanel user={user} mal_id={item.mal_id} surface={surface} rank={item.rank} />

        {item.synopsis && (
          <>
            <h3 className="section-h">{t('Story')}</h3>
            <p style={{ fontSize: 13.5, color: 'var(--muted)', margin: 0, lineHeight: 1.65 }}>
              {item.synopsis.slice(0, 700)}{item.synopsis.length > 700 ? '…' : ''}
            </p>
          </>
        )}

        <div className="chips" style={{ marginTop: 14 }}>
          {item.mal_genres?.map((g) => <span className="chip" key={g}>{label(g)}</span>)}
        </div>

        <h3 className="section-h">{t('Similar in tag space')}</h3>
        {similar === null ? (
          <p className="surface-note">{t('Loading…')}</p>
        ) : (
          <div className="results" style={{ position: 'static', marginTop: 8 }}>
            {similar.map((s) => (
              <button key={s.mal_id} style={{ cursor: 'default' }}>
                {s.picture_medium && <img src={s.picture_medium} alt="" />}
                <span>
                  <span className="t">{s.title}</span><br />
                  <span className="s">
                    {t('{p}% match', { p: (s.similarity * 100).toFixed(0) })}
                    {s.mal_mean ? ` · ★ ${s.mal_mean.toFixed(2)}` : ''}
                  </span>
                </span>
              </button>
            ))}
          </div>
        )}

        <p style={{ marginTop: 20 }}>
          <a href={`https://myanimelist.net/anime/${item.mal_id}`} target="_blank" rel="noreferrer">
            {t('Open on MyAnimeList ↗')}
          </a>
        </p>
      </div>
    </div>
  )
}

const MEDIA_TYPES = ['tv', 'movie', 'ona', 'ova', 'special']

/** Filters. Controlled; the parent persists them. */
export function Controls({ filters, onFilters }: {
  filters: Filters
  onFilters: (f: Filters) => void
}) {
  const { t, label } = useI18n()
  const [open, setOpen] = useState(false)
  const [genres, setGenres] = useState<string[]>([])
  useEffect(() => {
    if (open && genres.length === 0) api.genres().then(setGenres).catch(() => {})
  }, [open, genres.length])

  const n = (filters.media_types?.length ? 1 : 0) + (filters.exclude_genres?.length ? 1 : 0)
    + (filters.eps_min !== undefined || filters.eps_max !== undefined ? 1 : 0)
    + (filters.year_min !== undefined || filters.year_max !== undefined ? 1 : 0)

  const num = (v: string) => (v.trim() === '' ? undefined : Math.max(0, Math.floor(Number(v))))
  const toggle = (list: string[] | undefined, v: string) =>
    list?.includes(v) ? list.filter((x) => x !== v) : [...(list ?? []), v]

  return (
    <div className="controls">
      <div className="controls-row">
        <span className="spacer" />
        <button onClick={() => setOpen((o) => !o)} aria-expanded={open}>
          {t('Filters')}{n > 0 ? ` · ${t('filters.active', { n })}` : ''}
        </button>
        {n > 0 && <button onClick={() => onFilters({})}>{t('Reset')}</button>}
      </div>
      {open && (
        <div className="filters">
          <div className="frow">
            <span className="fk">{t('Format')}</span>
            {MEDIA_TYPES.map((m) => (
              <button key={m} className={`tab small${filters.media_types?.includes(m) ? ' on' : ''}`}
                      aria-pressed={!!filters.media_types?.includes(m)}
                      onClick={() => onFilters({ ...filters, media_types: toggle(filters.media_types, m) })}>
                {m.toUpperCase()}
              </button>
            ))}
          </div>
          <div className="frow">
            <span className="fk">{t('Episodes')}</span>
            <input type="number" min={0} placeholder={t('from')} value={filters.eps_min ?? ''}
                   onChange={(e) => onFilters({ ...filters, eps_min: num(e.target.value) })} />
            <input type="number" min={0} placeholder={t('to')} value={filters.eps_max ?? ''}
                   onChange={(e) => onFilters({ ...filters, eps_max: num(e.target.value) })} />
            <span className="fk">{t('Year')}</span>
            <input type="number" min={1950} placeholder={t('from')} value={filters.year_min ?? ''}
                   onChange={(e) => onFilters({ ...filters, year_min: num(e.target.value) })} />
            <input type="number" min={1950} placeholder={t('to')} value={filters.year_max ?? ''}
                   onChange={(e) => onFilters({ ...filters, year_max: num(e.target.value) })} />
          </div>
          <div className="frow wrap-row">
            <span className="fk">{t('Hide genres')}</span>
            {genres.map((g) => (
              <button key={g} className={`tab small${filters.exclude_genres?.includes(g) ? ' on' : ''}`}
                      aria-pressed={!!filters.exclude_genres?.includes(g)}
                      onClick={() => onFilters({ ...filters, exclude_genres: toggle(filters.exclude_genres, g) })}>
                {label(g)}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

/** Taste summary: what the model learned, and how well it scores. */
export function Insights({
  profile, model, onClose,
}: {
  profile: Profile
  model: ModelInfo | null
  onClose: () => void
}) {
  const { t, label } = useI18n()
  const [learn, setLearn] = useState<Learning | null>(null)
  useEffect(() => { api.learning().then(setLearn).catch(() => {}) }, [])
  const maxLift = Math.max(0.01, ...profile.top_genres.map((g) => Math.abs(g.lift)))
  return (
    <div className="drawer-bg" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>{t('Close')}</button>
        <h2>{t('Your taste')}</h2>
        <div className="stat-grid">
          <div className="stat"><div className="k">{t('Entries')}</div><div className="v">{profile.entries}</div></div>
          <div className="stat"><div className="k">{t('Rated')}</div><div className="v">{profile.scored}</div></div>
          <div className="stat"><div className="k">{t('Completed')}</div><div className="v">{profile.completed}</div></div>
          <div className="stat"><div className="k">{t('Mean score')}</div><div className="v">{profile.mean_score ?? '—'}</div></div>
        </div>

        <h3 style={{ fontSize: 14, marginTop: 20 }}>{t('Genres you rate above your own average')}</h3>
        {profile.top_genres.map((g) => (
          <div className="bar-row" key={g.genre}>
            <span className="name">{label(g.genre)}</span>
            <span className="bar">
              <div style={{ width: `${(Math.max(g.lift, 0) / maxLift) * 100}%` }} />
            </span>
            <span style={{ width: 46, textAlign: 'right' }}>
              {g.lift > 0 ? '+' : ''}{g.lift}
            </span>
          </div>
        ))}

        {model && (
          <>
            <h3 style={{ fontSize: 14, marginTop: 22 }}>{t('Model')}</h3>
            <div className="stat-grid">
              <div className="stat">
                <div className="k">{t('Rank correlation')}</div>
                <div className="v">{model.metrics.spearman?.toFixed(3) ?? '—'}</div>
              </div>
              <div className="stat">
                <div className="k">{t('RMSE')}</div>
                <div className="v">{model.metrics.rmse?.toFixed(2) ?? '—'}</div>
              </div>
            </div>
            <p className="surface-note">{t('insights.note')}</p>
          </>
        )}

        {learn && (
          <>
            <h3 style={{ fontSize: 14, marginTop: 22 }}>{t('learn.title')}</h3>
            {learn.status === 'ok' && learn.current ? (
              <div className="whypanel">
                <div className="drv"><span>{t('learn.relevance')}</span>
                  <span>{learn.relevance_points_per_sd?.toFixed(2)} ({learn.relevance_ci90?.map((x) => x.toFixed(2)).join('–')}) · {t('learn.now')} {learn.current.relevance_weight}</span></div>
                <div className="drv"><span>{t('learn.novelty')}</span>
                  <span>{learn.novelty_points?.toFixed(2)} ({learn.novelty_ci90?.map((x) => x.toFixed(2)).join('–')}) · {t('learn.now')} {learn.current.novelty_weight}</span></div>
              </div>
            ) : (
              <>
                <div className="bar-row">
                  <span className="bar"><div style={{ width: `${Math.min(100, learn.events / 2)}%` }} /></span>
                  <span style={{ width: 70, textAlign: 'right' }}>{learn.events} / 200</span>
                </div>
                <p className="surface-note">{t('learn.pending', { p: learn.positive, n: learn.negative })}</p>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export function SearchBox({ onOpen }: { onOpen: (mal_id: number) => void }) {
  const { t } = useI18n()
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<SearchHit[] | null>(null)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => {
    window.clearTimeout(timer.current)
    if (q.trim().length < 2) { setHits(null); return }
    timer.current = window.setTimeout(() => {
      api.search(q.trim()).then(setHits).catch(() => setHits([]))
    }, 220)
    return () => window.clearTimeout(timer.current)
  }, [q])

  return (
    <div className="searchbox">
      <input
        value={q}
        placeholder={t('Search the catalogue…')}
        onChange={(e) => setQ(e.target.value)}
        onBlur={() => window.setTimeout(() => setHits(null), 150)}
      />
      {hits && hits.length > 0 && (
        <div className="results">
          {hits.map((h) => (
            <button key={h.mal_id} onMouseDown={(e) => e.preventDefault()}
                    onClick={() => { setHits(null); setQ(''); onOpen(h.mal_id) }}>
              {h.picture_medium && <img src={h.picture_medium} alt="" />}
              <span>
                <span className="t">{h.title_en || h.title}</span><br />
                <span className="s">
                  {h.season_year ?? '—'} · {h.media_type ?? '—'}
                  {h.mal_mean ? ` · ★ ${h.mal_mean.toFixed(2)}` : ''}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
