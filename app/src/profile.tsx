// The profile page: what your list says about your taste, one headline
// sentence per section over the numbers behind it. The API decides what may
// be said (malrec.taste: each series counts once, only clear differences);
// this file only words and draws it.
import { useEffect, useState, type ReactNode } from 'react'
import { api, type Headline, type Learning, type TasteGroup, type TasteProfile } from './api'
import { useI18n } from './i18n'

const PERCENT = new Set(['new', 'old', 'share'])
const AGE_KEYS = ['profile.age0', 'profile.age1', 'profile.age2', 'profile.age3', 'profile.age4']
const AGE_SHADES = [1, 0.72, 0.5, 0.32, 0.16]

function useHeadline() {
  const { t, label } = useI18n()
  return (h?: Headline) => {
    if (!h) return ''
    const vars: Record<string, string | number> = {}
    for (const [k, v] of Object.entries(h)) {
      if (k === 'key') continue
      if (typeof v === 'number') {
        vars[k] = PERCENT.has(k) ? Math.round(v * 100)
          : k === 'diff' || k === 'rmse' ? Math.abs(v).toFixed(2) : v
      } else {
        vars[k] = label(v)
      }
    }
    return t(`taste.${h.key}`, vars)
  }
}

function Section({ kicker, headline, children }:
  { kicker: string; headline: string; children?: ReactNode }) {
  return (
    <section className="pcard">
      <div className="kicker">{kicker}</div>
      <h2>{headline}</h2>
      {children}
    </section>
  )
}

function sign(x: number) {
  return `${x > 0 ? '+' : '−'}${Math.abs(x).toFixed(2)}`
}

/** Clear groups as bars around the user's average: right is higher, left lower. */
function Diverging({ groups }: { groups: TasteGroup[] }) {
  const { t, label } = useI18n()
  const max = Math.max(0.5, ...groups.map((g) => Math.abs(g.lift)))
  return (
    <>
      {groups.map((g) => (
        <div className="prow" key={g.name}>
          <span className="name" title={label(g.name)}>{label(g.name)}</span>
          <div className="diverge" aria-hidden="true">
            <div>{g.lift < 0 && <div className="neg" style={{ width: `${(-g.lift / max) * 100}%` }} />}</div>
            <div>{g.lift > 0 && <div className="pos" style={{ width: `${(g.lift / max) * 100}%` }} />}</div>
          </div>
          <span className={`val ${g.lift > 0 ? 'up' : 'down'}`}>{sign(g.lift)}</span>
          <span className="cnt">{t('profile.series_n', { n: g.series })}</span>
        </div>
      ))}
    </>
  )
}

/** A mean score on the 5-10 scale; `tone` marks it above or below the average. */
function MeanRow({ name, mean, count, tone }:
  { name: string; mean: number; count: string; tone?: 'hi' | 'lo' }) {
  const w = Math.max(0, Math.min(1, (mean - 5) / 5)) * 100
  return (
    <div className="prow">
      <span className="name" title={name}>{name}</span>
      <div className="track" aria-hidden="true"><div className={tone ?? ''} style={{ width: `${w}%` }} /></div>
      <span className="val">{mean.toFixed(2)}</span>
      <span className="cnt">{count}</span>
    </div>
  )
}

function Learn({ learn }: { learn: Learning }) {
  const { t } = useI18n()
  return (
    <section className="pcard">
      <div className="kicker">{t('learn.title')}</div>
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
          <p className="note">{t('learn.pending', { p: learn.positive, n: learn.negative })}</p>
        </>
      )}
    </section>
  )
}

export function ProfilePage({ user, own, onBack }: { user: string; own: boolean; onBack: () => void }) {
  const { t, label } = useI18n()
  const head = useHeadline()
  const [data, setData] = useState<TasteProfile | null>(null)
  const [failed, setFailed] = useState(false)
  const [learn, setLearn] = useState<Learning | null>(null)

  useEffect(() => {
    let alive = true
    setData(null)
    setFailed(false)
    api.taste(user).then((d) => alive && setData(d)).catch(() => alive && setFailed(true))
    api.learning().then((l) => alive && setLearn(l)).catch(() => {})
    return () => { alive = false }
  }, [user])

  const top = (
    <div className="profile-head">
      <div>
        <button className="button-link" onClick={onBack}>← {t('profile.back')}</button>
        <h1>{own ? t('Your profile') : t('profile.title_of', { u: user })}</h1>
        {data && (
          <div className="sub">{t('profile.sub', { e: data.stats.entries, s: data.stats.series })}</div>
        )}
      </div>
    </div>
  )
  if (failed) return <div className="profile">{top}<div className="error">{t('profile.failed')}</div></div>
  if (!data) return <div className="profile">{top}<p className="surface-note">{t('profile.loading')}</p></div>

  const { stats, scale, genres, themes, disagree, eras, studios, predictions } = data
  const maxHist = Math.max(1, ...scale.hist)
  const listedMax = Math.max(1, ...genres.listed.map((g) => g.n))
  const clearUp = new Set(genres.clear.filter((g) => g.lift > 0).map((g) => g.name))
  const clearDown = new Set(genres.clear.filter((g) => g.lift < 0).map((g) => g.name))
  const eraTone = eras.headline?.key === 'eras.favourite' ? 'hi' : eras.headline?.key === 'eras.lower' ? 'lo' : null
  const eraMark = eraTone ? Number(eras.headline?.from) : null
  const hidden = !themes.headline || !studios.headline || !eras.headline

  return (
    <div className="profile">
      {top}

      <div className="tiles">
        <div className="tile"><div className="k">{t('Entries')}</div><div className="v">{stats.entries}</div>
          <div className="s">{t('profile.series_n', { n: stats.series })}</div></div>
        <div className="tile"><div className="k">{t('Rated')}</div><div className="v">{stats.rated}</div>
          <div className="s">{stats.mean != null ? t('profile.mean', { m: stats.mean.toFixed(2) }) : '—'}</div></div>
        <div className="tile"><div className="k">{t('Completed')}</div><div className="v">{stats.completed}</div>
          <div className="s">{t('profile.dropped', { n: stats.dropped })}</div></div>
        <div className="tile"><div className="k">{t('profile.time')}</div><div className="v">{stats.hours} h</div>
          <div className="s">{t('profile.days', { n: Math.round(stats.hours / 24) })}</div></div>
      </div>

      <div className="pcards">
        <Section kicker={t('profile.k.scale')} headline={head(scale.headline)}>
          {stats.rated > 0 && (
            <>
              <div className="hist" aria-hidden="true">
                {scale.hist.map((n, i) => (
                  <div key={i}>
                    <span className="n">{n || ''}</span>
                    <div className={`b${stats.mean != null && i + 1 > stats.mean ? ' hi' : ''}`}
                         style={{ height: `${n ? Math.max(3, (n / maxHist) * 100) : 0}px` }} />
                  </div>
                ))}
              </div>
              <div className="hist-axis" aria-hidden="true">
                {scale.hist.map((_, i) => <span key={i}>{i + 1}</span>)}
              </div>
            </>
          )}
          {scale.diff != null && scale.compared >= 10 && (
            <p>{t(scale.diff >= 0.05 ? 'taste.scale.body_more' : scale.diff <= -0.05 ? 'taste.scale.body_less'
              : 'taste.scale.body_same', {
              n: scale.compared, d: Math.abs(scale.diff).toFixed(2),
              a: Math.round((scale.above ?? 0) * 100), b: Math.round((scale.below ?? 0) * 100),
            })}</p>
          )}
        </Section>

        <Section kicker={t('profile.k.genres')} headline={head(genres.headline)}>
          {genres.clear.length > 0 && <Diverging groups={genres.clear} />}
          {genres.listed.length > 0 && (
            <>
              <div className="sub-h">{t('profile.listed')}</div>
              {genres.listed.map((g) => (
                <div className="prow" key={g.name}>
                  <span className="name">{label(g.name)}</span>
                  <div className="track" aria-hidden="true">
                    <div className={clearUp.has(g.name) ? 'hi' : clearDown.has(g.name) ? 'lo' : ''}
                         style={{ width: `${(g.n / listedMax) * 100}%` }} />
                  </div>
                  <span className="val">{g.n}</span>
                  <span className="cnt" />
                </div>
              ))}
            </>
          )}
          {data.taste.enough && <p className="note">{t('profile.genres_rest')}</p>}
        </Section>

        {themes.headline && (
          <Section kicker={t('profile.k.themes')} headline={head(themes.headline)}>
            <div className="chips">
              {themes.clear.filter((g) => g.lift > 0).map((g) => (
                <span className="chip" key={g.name}>{g.name} <b>{sign(g.lift)}</b> <span className="c">· {g.series}</span></span>
              ))}
            </div>
            {themes.clear.some((g) => g.lift < 0) && (
              <div className="chips">
                {themes.clear.filter((g) => g.lift < 0).map((g) => (
                  <span className="chip down" key={g.name}>{g.name} <b>{sign(g.lift)}</b> <span className="c">· {g.series}</span></span>
                ))}
              </div>
            )}
            <p className="note">{t('profile.themes_note')}</p>
          </Section>
        )}

        {disagree.headline && (
          <Section kicker={t('profile.k.disagree')} headline={head(disagree.headline)}>
            {(disagree.above.length > 0 || disagree.below.length > 0) && (
              <div className="dis-grid">
                {[['above', disagree.above], ['below', disagree.below]].map(([side, list]) =>
                  (list as typeof disagree.above).length > 0 && (
                    <div key={side as string}>
                      <div className={`side${side === 'below' ? ' down' : ''}`}>
                        {t(side === 'above' ? 'profile.you_more' : 'profile.you_less')}
                      </div>
                      {(list as typeof disagree.above).map((x) => (
                        <div className="dis-item" key={x.mal_id}>
                          {x.picture ? <img src={x.picture} alt="" loading="lazy" /> : <span className="ph" />}
                          <div>
                            <div className="t">{x.title}</div>
                            <div className="s">{t('profile.you_vs', { you: x.you, mal: x.mal.toFixed(2) })}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
              </div>
            )}
            <p className="note">{t('profile.disagree_note')}</p>
          </Section>
        )}

        {eras.headline && (
          <Section kicker={t('profile.k.eras')} headline={head(eras.headline)}>
            {eras.rows.map((r) => (
              <MeanRow key={r.from} name={`${r.from}–${r.to}`} mean={r.mean}
                       count={t('profile.series_n', { n: r.series })}
                       tone={eraMark === r.from && eraTone ? eraTone : undefined} />
            ))}
            {eras.ages && (
              <>
                <div className="sub-h">{t('profile.ages')}</div>
                <div className="ages" aria-hidden="true">
                  {eras.ages.map((p, i) => p > 0 && (
                    <div key={i} style={{ width: `${p * 100}%`, opacity: AGE_SHADES[i] }} />
                  ))}
                </div>
                <div className="ages-key">
                  {eras.ages.map((p, i) => (
                    <span key={i}><i style={{ opacity: AGE_SHADES[i] }} />{t(AGE_KEYS[i])} {Math.round(p * 100)} %</span>
                  ))}
                </div>
                {eras.headline.key !== 'eras.ages_only' && (
                  <p>{t('taste.eras.age_body', {
                    new: Math.round((eras.new ?? 0) * 100), old: Math.round((eras.old ?? 0) * 100),
                  })}</p>
                )}
              </>
            )}
          </Section>
        )}

        {studios.headline && (
          <Section kicker={t('profile.k.studios')} headline={head(studios.headline)}>
            {studios.clear.map((g) => (
              <MeanRow key={g.name} name={g.name} mean={g.mean}
                       count={t('profile.series_n', { n: g.series })} tone={g.lift > 0 ? 'hi' : 'lo'} />
            ))}
            <p>{t('taste.studios.body', { mu: data.taste.series_mean.toFixed(2) })}</p>
          </Section>
        )}

        {predictions && (
          <Section kicker={t('profile.k.pred')} headline={head(predictions.headline)}>
            <div className="mini">
              <div><div className="k">{t('profile.rmse')}</div><div className="v">{predictions.rmse.toFixed(2)}</div>
                <div className="s">{t('profile.rmse_unit')}</div></div>
              {predictions.spearman != null && (
                <div><div className="k">{t('profile.order')}</div><div className="v">{predictions.spearman.toFixed(2)}</div>
                  <div className="s">{t('profile.order_unit')}</div></div>
              )}
            </div>
            {predictions.holdout_min != null && (
              <p className="note">{t('profile.pred_note', { lo: predictions.holdout_min, hi: predictions.holdout_max ?? predictions.holdout_min })}</p>
            )}
          </Section>
        )}

        {learn && <Learn learn={learn} />}
      </div>

      {hidden && <p className="rule">{t('profile.more_later')}</p>}
      <p className="rule">{t('profile.rule')}</p>
    </div>
  )
}
