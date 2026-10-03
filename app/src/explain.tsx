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
