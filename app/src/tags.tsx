// Clickable tags: a tag on a card or in "Why this?" opens the catalogue's
// titles with it - the ones not on the list best predicted first, and the
// list's own with their scores.
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, type TagTitle, type TagTitles } from './api'
import { useI18n } from './i18n'

export const OpenTag = createContext<((tag: string) => void) | null>(null)

/** A tag as quiet text that opens its titles; plain text where nothing can open them. */
export function TagLink({ tag, children, className }: {
  tag: string; children?: ReactNode; className?: string
}) {
  const open = useContext(OpenTag)
  const { t, label } = useI18n()
  const text = children ?? label(tag)
  if (!open) return <span className={className}>{text}</span>
  return (
    <button type="button" className={`tag-link${className ? ` ${className}` : ''}`}
            title={t('tag.title')} onClick={(e) => { e.stopPropagation(); open(tag) }}>
      {text}
    </button>
  )
}

function Row({ r, onOpen, own }: { r: TagTitle; onOpen: (mal_id: number) => void; own: boolean }) {
  const { t } = useI18n()
  const l = r.likely
  const sub = own
    ? (r.list_score ? t('tag.your_score', { s: r.list_score }) : t(`status.${r.list_status}`))
    : l ? t(l.low === l.high ? 'likely.one' : 'likely', { lo: l.low, hi: l.high }) : null
  return (
    <button onClick={() => onOpen(r.mal_id)}>
      {r.picture_medium && <img src={r.picture_medium} alt="" loading="lazy" />}
      <span style={{ flex: 1, minWidth: 0 }}>
        <span className="t">{r.title_en || r.title}</span><br />
        <span className="s">
          {[r.season_year, sub, r.mal_mean ? `MAL ${r.mal_mean.toFixed(2)}` : null]
            .filter(Boolean).join(' · ')}
        </span>
      </span>
      {!own && r.predicted_score != null && <b className="tag-score">{r.predicted_score.toFixed(1)}</b>}
    </button>
  )
}

export function TagPanel({ tag, user, onOpen, onClose }: {
  tag: string; user: string; onOpen: (mal_id: number) => void; onClose: () => void
}) {
  const { t, label } = useI18n()
  const [data, setData] = useState<TagTitles | null | undefined>(undefined)

  useEffect(() => {
    let alive = true
    setData(undefined)
    api.tag(tag, user).then((d) => alive && setData(d)).catch(() => alive && setData(null))
    return () => { alive = false }
  }, [tag, user])

  return (
    <div className="drawer-bg" onClick={onClose}>
      <div className="drawer" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>{t('Close')}</button>
        <h2>{t('tag.h', { tag: label(tag) })}</h2>
        {data === undefined && <p className="surface-note">{t('Loading…')}</p>}
        {data === null && <p className="surface-note">{t('tag.none')}</p>}
        {data && (
          <>
            <h3 className="section-h">{t('tag.unlisted', { n: data.unlisted })}</h3>
            {data.titles.length === 0 ? <p className="surface-note">{t('tag.none')}</p> : (
              <div className="results tag-results">
                {data.titles.map((r) => <Row key={r.mal_id} r={r} onOpen={onOpen} own={false} />)}
              </div>
            )}
            {data.yours_total > 0 && (
              <>
                <h3 className="section-h">{t('tag.yours', { n: data.yours_total })}</h3>
                <div className="results tag-results">
                  {data.yours.map((r) => <Row key={r.mal_id} r={r} onOpen={onOpen} own />)}
                </div>
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
