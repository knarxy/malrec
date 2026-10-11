// Written explanations and the likely-rating range shown on cards and in the
// details drawer. Built only from the reasons the API attached (no free text
// is generated), so every sentence is backed by the model's actual drivers.
import type { ReactNode } from 'react'
import type { Reason, Recommendation } from './api'
import { useI18n } from './i18n'

// Driver labels that describe general acclaim rather than this user's taste,
// and ones that read badly inside a sentence (handled separately or dropped).
const CONSENSUS = new Set([
  "loved by those who've seen it", 'highly rated on MyAnimeList', 'highly rated on AniList',
  'widely watched', 'a favourite for many people', 'broadly liked', 'rarely dropped',
])
const SKIP = new Set([
  'a popular title', 'under the radar', 'often dropped', 'divides opinion',
  'common on lists like yours', 'fits your taste profile', 'similar to shows you rated highly',
  'matches your tag profile', 'recommended alongside shows you rate highly',
])
const PROFILE = new Set([
  'fits your taste profile', 'similar to shows you rated highly', 'matches your tag profile',
])

function joinList(xs: string[], lang: string): string {
  const and = lang === 'de' ? 'und' : 'and'
  if (xs.length <= 1) return xs.join('')
  return `${xs.slice(0, -1).join(', ')} ${and} ${xs[xs.length - 1]}`
}

/** Fill {a} / {b} placeholders with bold titles. */
function withTitles(text: string, a: string, b?: string): ReactNode {
  return text.split(/(\{a\}|\{b\})/).map((part, k) =>
    part === '{a}' ? <b key={k}>{a}</b> : part === '{b}' ? <b key={k}>{b}</b> : part)
}

/** Up to three plain sentences: what on your list the title is tied to,
 *  which of your tastes it matches, and context (acclaim, rarity, or an
 *  honest "no direct link"). */
export function Why({ reasons }: { reasons: Reason[] }) {
  const { t, label, lang } = useI18n()
  const liked = reasons.filter(
    (r): r is Extract<Reason, { kind: 'because_you_liked' }> => r.kind === 'because_you_liked')
  const cont = reasons.find(
    (r): r is Extract<Reason, { kind: 'continues' }> => r.kind === 'continues')
  const coming = reasons.find(
    (r): r is Extract<Reason, { kind: 'coming' }> => r.kind === 'coming')
  const gem = reasons.some((r) => r.kind === 'deep_cut')
  const drivers = reasons.find(
    (r): r is Extract<Reason, { kind: 'drivers' }> => r.kind === 'drivers')
  const labels = drivers?.items.map((d) => d.label) ?? []
  const traits = labels.filter((l) => !CONSENSUS.has(l) && !SKIP.has(l)).slice(0, 3)
  const acclaimed = labels.some((l) => CONSENSUS.has(l))
  const common = labels.includes('common on lists like yours')
  const profile = labels.some((l) => PROFILE.has(l))

  const s: ReactNode[] = []
  if (cont) {
    s.push(withTitles(t(coming ? 'x.new_in' : 'x.continues', { s: cont.your_score }), cont.title))
  } else if (liked.length) {
    const two = liked.slice(0, 2)
    s.push(withTitles(t(two.length > 1 ? 'x.fans2' : 'x.fans1',
      { s1: two[0].your_score, s2: two[1]?.your_score ?? '' }), two[0].title, two[1]?.title))
  } else if (common) {
    s.push(t('x.common'))
  }
  if (traits.length) s.push(t('x.traits', { list: joinList(traits.map(label), lang) }))
  else if (profile) s.push(t('x.profile'))
  if (gem) s.push(t('x.gem'))
  else if (acclaimed && s.length < 3) s.push(t('x.acclaim'))
  if (!cont && !liked.length && !common && drivers && !drivers.has_affinity) s.push(t('x.nolink'))

  return (
    <div className="why">
      {coming && (
        <div className="line coming">
          {coming.date
            ? t('starts', { d: new Date(coming.date).toLocaleDateString(lang === 'de' ? 'de-DE' : 'en-GB',
                { day: 'numeric', month: 'short', year: 'numeric' }) })
            : t('tba')}
        </div>
      )}
      {s.length > 0 && (
        <p className="explain">
          {s.map((x, k) => <span key={k}>{x}{k < s.length - 1 ? ' ' : ''}</span>)}
        </p>
      )}
    </div>
  )
}

/** Drivers are drawn on one fixed scale, so bars compare across cards. */
const DRIVER_SCALE = 1.05

/** The card's reasons, laid out rather than written: the titles on your list
 *  it is tied to (with your score), what speaks for it as bars split into
 *  your list's signals and the community's, and its matching tags. Titles
 *  with neither a link nor drivers fall back to the written sentences. */
export function Reasons({ item }: { item: Recommendation }) {
  const { t, label } = useI18n()
  const reasons = item.reasons ?? []
  const liked = reasons.filter(
    (r): r is Extract<Reason, { kind: 'because_you_liked' }> => r.kind === 'because_you_liked')
  const cont = reasons.find(
    (r): r is Extract<Reason, { kind: 'continues' }> => r.kind === 'continues')
  const coming = reasons.find(
    (r): r is Extract<Reason, { kind: 'coming' }> => r.kind === 'coming')
  const drivers = (reasons.find(
    (r): r is Extract<Reason, { kind: 'drivers' }> => r.kind === 'drivers')?.items ?? [])
    .filter((d) => d.weight > 0).sort((a, b) => b.weight - a.weight).slice(0, 4)
  const tags = reasons.find(
    (r): r is Extract<Reason, { kind: 'tags' }> => r.kind === 'tags')?.tags ?? []
  const gem = reasons.some((r) => r.kind === 'deep_cut')

  const links = cont
    ? [{ title: cont.title, score: cont.your_score, note: t(coming ? 'r.new_in' : 'r.continues') }]
    : liked.slice(0, 3).map((l) => ({ title: l.title, score: l.your_score, note: null }))

  return (
    <div className="reasons">
      <div className="r-links">
        {(coming || item.also_in || gem) && (
          <div className="r-flags">
            {coming && <Why reasons={[coming]} />}
            {item.also_in && <span className="flag">{t('card.also_in', { tab: t(`tab.${item.also_in}`) })}</span>}
            {gem && <span className="flag">{t('r.gem')}</span>}
          </div>
        )}
        {links.length > 0 ? (
          <>
            <div className="r-h">{t(cont ? 'r.continues_h' : 'r.liked_h')}</div>
            <ul className="r-liked">
              {links.map((l) => (
                <li key={l.title}>
                  <span className="r-title">{l.title}{l.note && <span className="r-note"> · {l.note}</span>}</span>
                  <b>{l.score}</b>
                </li>
              ))}
            </ul>
          </>
        ) : (
          <Why reasons={reasons.filter((r) => r.kind !== 'coming')} />
        )}
        {!unairedReasons(reasons) && <Likely item={item} />}
      </div>
      {drivers.length > 0 && (
        <div className="r-drivers">
          <div className="r-h">{t('r.drivers_h')}</div>
          {drivers.map((d) => (
            <div className="r-driver" key={d.label}
                 title={t(d.personal ? 'r.from_you' : 'r.from_all')}>
              <span className="r-label">{label(d.label)}</span>
              <span className="r-track">
                <span className={d.personal ? 'r-bar you' : 'r-bar all'}
                      style={{ width: `${Math.min(100, (d.weight / DRIVER_SCALE) * 100)}%` }} />
              </span>
            </div>
          ))}
        </div>
      )}
      {tags.length > 0 && (
        <div className="r-tags">
          {tags.slice(0, 5).map((g) => <span className="tag" key={g}>{label(g)}</span>)}
        </div>
      )}
    </div>
  )
}

function unairedReasons(reasons: Reason[]): boolean {
  return reasons.some((r) => r.kind === 'coming')
}

/** The two bar colours, explained once above the cards. */
export function ReasonLegend() {
  const { t } = useI18n()
  return (
    <div className="legend" aria-hidden="true">
      <span><i className="r-bar you" />{t('r.from_you')}</span>
      <span><i className="r-bar all" />{t('r.from_all')}</span>
    </div>
  )
}

/** "likely 7-9 · 35 % chance of 9+" under a prediction. */
export function Likely({ item }: { item: Recommendation }) {
  const { t } = useI18n()
  const l = item.likely
  if (!l) return null
  return (
    <div className="likely" title={t('likely.title')}>
      {t(l.low === l.high ? 'likely.one' : 'likely', { lo: l.low, hi: l.high })}
      {' · '}{t('p9', { p: Math.round(l.p9 * 100) })}
    </div>
  )
}
