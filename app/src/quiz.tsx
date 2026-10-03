// The rating round: titles you have probably seen but not logged, one card at
// a time. A score is written to MyAnimeList like any other rating from the
// app; "haven't seen it" and "skip" just move on and are not asked again.
import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, type QuizCard } from './api'
import { useI18n } from './i18n'

const GOAL = 15

export function Quiz({ onClose, onRated }: { onClose: () => void; onRated: () => void }) {
  const { t, label } = useI18n()
  const [cards, setCards] = useState<QuizCard[] | null>(null)
  const [rated, setRated] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = useCallback(async () => {
    try {
      const r = await api.quiz(4)
      setCards(r.cards)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t('Failed to load'))
    }
  }, [t])
  useEffect(() => { load() }, [load])

  const next = useCallback(async () => {
    setCards((c) => (c ? c.slice(1) : c))
    if (!cards || cards.length <= 2) await load()
  }, [cards, load])

  async function answer(score: number | 'unseen' | 'skip') {
    const card = cards?.[0]
    if (!card) return
    setBusy(true); setError(null)
    try {
      if (typeof score === 'number') {
        await api.rate(card.mal_id, score, 'quiz')
        setRated((n) => n + 1)
        onRated()
      } else {
        await api.quizAnswer(card.mal_id, score)
      }
      await next()
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t('Could not reach MyAnimeList.'))
    } finally {
      setBusy(false)
    }
  }

  const card = cards?.[0]
  return (
    <div className="drawer-bg" onClick={onClose}>
      <div className="drawer quiz" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>{rated >= GOAL ? t('Done') : t('Close')}</button>
        <h2>{t('quiz.title')}</h2>
        <p className="surface-note">{t('quiz.sub')}</p>
        <div className="progress"><div style={{ width: `${Math.min(100, (rated / GOAL) * 100)}%` }} /></div>
        <p className="surface-note">{t('quiz.progress', { n: rated, goal: GOAL })}</p>
        {error && <div className="error">{error}</div>}
        {cards === null && <p className="surface-note">{t('Loading…')}</p>}
        {cards !== null && !card && <p className="empty">{t('quiz.empty')}</p>}
        {card && (
          <div className="quiz-card">
            {(card.picture_large || card.picture_medium) && (
              <img src={card.picture_large ?? card.picture_medium!} alt={card.title} />
            )}
            <div>
              <h3>{card.title_en || card.title}</h3>
              <div className="meta">
                {card.season_year && <span>{card.season_year}</span>}
                {card.media_type && <span>{card.media_type.toUpperCase()}</span>}
                {card.num_episodes && <span>{card.num_episodes} {t('ep')}</span>}
              </div>
              <div className="chips">
                {(card.mal_genres ?? []).slice(0, 4).map((g) => <span className="chip" key={g}>{label(g)}</span>)}
              </div>
              <p className="surface-note">{t('quiz.q')}</p>
              <div className="scores">
                {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((v) => (
                  <button key={v} disabled={busy} onClick={() => answer(v)}>{v}</button>
                ))}
              </div>
              <div className="card-actions">
                <button disabled={busy} onClick={() => answer('unseen')}>{t('quiz.unseen')}</button>
                <button disabled={busy} onClick={() => answer('skip')}>{t('quiz.skip')}</button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
