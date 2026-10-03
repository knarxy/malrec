// The privacy page (/privacy): what the app stores and why, reachable signed
// in or out, plus "delete my account" for a signed-in user. Every statement
// here describes what the code does - keep it in step when that changes.
import { type ReactNode, useEffect, useState } from 'react'
import { api, ApiError, type AuthState } from './api'
import { BrandMark } from './components'
import { useI18n } from './i18n'

type Section = { h: string; p?: string[]; li?: string[] }

const TEXT: Record<'en' | 'de', { title: string; intro: string; sections: Section[] }> = {
  en: {
    title: 'Privacy and your data',
    intro: 'malrec recommends anime from your MyAnimeList list. It stores only what it needs for that, never sells or shares it, and has no ads, analytics or tracking.',
    sections: [
      {
        h: 'What is stored when you sign in',
        li: [
          'Your MyAnimeList username, user ID and the address of your profile picture.',
          'A copy of your anime list: titles, status, scores and dates. Private lists are read with your permission.',
          'What you do in the app: ratings, “Not for me”, Plan to Watch, rating-round answers, your language and filters.',
          'The MyAnimeList access tokens that let malrec read your list and save ratings and Plan to Watch entries for you. They are stored encrypted.',
          'A session cookie that keeps you signed in. It holds a random ID; the server keeps only a hash of it. Sessions end after 30 days.',
          'Your personal model and recommendations, computed from your list.',
        ],
      },
      {
        h: 'What it is used for',
        p: [
          'Only to build and explain your recommendations and to write the changes you make in the app back to MyAnimeList. Nothing on your MAL list changes unless you do it in the app.',
          'The operator of this site can see the list of accounts with their counts, and can open an account’s recommendations to help or to check the model.',
          'With Watch together, the person you paired with sees your username and your predicted scores for titles on your shared list, never your list itself.',
        ],
      },
      {
        h: 'Other services',
        li: [
          'MyAnimeList: sign-in, reading your list, saving your changes.',
          'AniList: public catalogue data only. Nothing about you is sent.',
          'Cover images load from MyAnimeList’s and AniList’s image servers, which see your IP address like any website you visit. Fonts and everything else come from this site.',
          'Your browser stores your language and filters so they work before you sign in.',
        ],
      },
      {
        h: 'The public-list sample',
        p: [
          'The “viewers with your taste” part of the model learns from a sample of other people’s public MyAnimeList lists. Accounts of this app are kept out of that sample. Once a list is fetched, its username is no longer stored.',
        ],
      },
      {
        h: 'How long it is kept, and deleting it',
        p: [
          'Everything stays as long as your account exists. Deleting your account (below, when signed in) removes all of it at once: your list copy, ratings, models, recommendations, feedback, sessions, tokens and pairings. Only an anonymous fingerprint of your username stays, so your public list is never added to the sample later.',
          'Nightly backups of the database are kept for 7 days, so deleted data is gone from them within a week. Server logs can contain your username, never your list or tokens.',
          'Deleting your account changes nothing on MyAnimeList. You can also remove malrec’s access in your MyAnimeList account settings.',
        ],
      },
    ],
  },
  de: {
    title: 'Datenschutz und deine Daten',
    intro: 'malrec empfiehlt Anime anhand deiner MyAnimeList-Liste. Es speichert nur, was es dafür braucht, verkauft oder teilt nichts und verwendet keine Werbung, keine Analyse und kein Tracking.',
    sections: [
      {
        h: 'Was bei der Anmeldung gespeichert wird',
        li: [
          'Dein MyAnimeList-Name, deine Nutzer-ID und die Adresse deines Profilbilds.',
          'Eine Kopie deiner Anime-Liste: Titel, Status, Bewertungen und Daten. Private Listen werden mit deiner Erlaubnis gelesen.',
          'Was du in der App tust: Bewertungen, „Nichts für mich“, Plan to Watch, Antworten in der Bewertungsrunde, Sprache und Filter.',
          'Die MyAnimeList-Zugangstoken, mit denen malrec deine Liste liest und Bewertungen sowie Plan-to-Watch-Einträge für dich speichert. Sie werden verschlüsselt gespeichert.',
          'Ein Sitzungs-Cookie, das dich angemeldet hält. Es enthält eine zufällige ID; der Server speichert nur einen Hash davon. Sitzungen enden nach 30 Tagen.',
          'Dein persönliches Modell und deine Empfehlungen, berechnet aus deiner Liste.',
        ],
      },
      {
        h: 'Wofür es verwendet wird',
        p: [
          'Nur um deine Empfehlungen zu erstellen und zu erklären und um Änderungen, die du in der App machst, an MyAnimeList zu übertragen. Auf deiner MAL-Liste ändert sich nur, was du in der App selbst änderst.',
          'Der Betreiber dieser Seite sieht die Liste der Konten mit ihren Zahlen und kann die Empfehlungen eines Kontos öffnen, um zu helfen oder das Modell zu prüfen.',
          'Bei „Zusammen schauen“ sieht die verbundene Person deinen Namen und deine vorhergesagten Bewertungen für die Titel eurer gemeinsamen Liste, nie deine Liste selbst.',
        ],
      },
      {
        h: 'Andere Dienste',
        li: [
          'MyAnimeList: Anmeldung, Lesen deiner Liste, Speichern deiner Änderungen.',
          'AniList: nur öffentliche Katalogdaten. Über dich wird nichts übertragen.',
          'Cover-Bilder werden von den Bildservern von MyAnimeList und AniList geladen, die dabei wie jede Website deine IP-Adresse sehen. Schriften und alles andere kommen von dieser Seite.',
          'Dein Browser speichert Sprache und Filter, damit sie schon vor der Anmeldung funktionieren.',
        ],
      },
      {
        h: 'Die Stichprobe öffentlicher Listen',
        p: [
          'Der Teil „Zuschauer mit deinem Geschmack“ lernt aus einer Stichprobe öffentlicher MyAnimeList-Listen anderer Leute. Konten dieser App sind davon ausgenommen. Sobald eine Liste geladen ist, wird ihr Name nicht mehr gespeichert.',
        ],
      },
      {
        h: 'Wie lange es gespeichert bleibt, und Löschen',
        p: [
          'Alles bleibt gespeichert, solange dein Konto besteht. Wenn du dein Konto löschst (unten, wenn du angemeldet bist), wird alles sofort entfernt: Listenkopie, Bewertungen, Modelle, Empfehlungen, Feedback, Sitzungen, Token und Verbindungen. Nur ein anonymer Fingerabdruck deines Namens bleibt, damit deine öffentliche Liste später nicht in die Stichprobe aufgenommen wird.',
          'Nächtliche Sicherungen der Datenbank werden 7 Tage aufbewahrt; gelöschte Daten sind daraus nach spätestens einer Woche verschwunden. Server-Protokolle können deinen Namen enthalten, nie deine Liste oder Token.',
          'Das Löschen deines Kontos ändert nichts auf MyAnimeList. Den Zugriff von malrec kannst du zusätzlich in deinen MyAnimeList-Kontoeinstellungen entfernen.',
        ],
      },
    ],
  },
}

function contactLink(c: string): ReactNode {
  if (/^https?:\/\//.test(c)) return <a href={c} target="_blank" rel="noopener noreferrer">{c}</a>
  if (/^[^@\s]+@[^@\s]+$/.test(c)) return <a href={`mailto:${c}`}>{c}</a>
  return c
}

export function PrivacyPage({ onLang }: { onLang: (l: 'en' | 'de') => void }) {
  const { t, lang } = useI18n()
  const text = TEXT[lang === 'de' ? 'de' : 'en']
  const [auth, setAuth] = useState<AuthState | null>(null)
  const [contact, setContact] = useState<string | null>(null)
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)
  const [deleted, setDeleted] = useState(false)

  useEffect(() => {
    api.authMe().then(setAuth).catch(() => setAuth({ signed_in: false }))
    api.authConfig().then((c) => setContact(c.operator_contact ?? null)).catch(() => {})
  }, [])
  useEffect(() => { document.title = `${text.title} · malrec` }, [text.title])

  const name = auth?.username ?? ''
  const matches = confirm.trim().toLowerCase() === name.toLowerCase() && name !== ''

  async function remove() {
    setBusy(true); setMsg(null)
    try {
      await api.deleteMe(confirm.trim())
      setDeleted(true)
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : t('privacy.delete_failed'))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <header className="topbar">
        <div className="wrap topbar-inner">
          <a className="brand" href="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            <BrandMark />mal<span>rec</span>
          </a>
          <span className="spacer" />
          <select className="lang" value={lang} aria-label={t('Language')}
                  onChange={(e) => onLang(e.target.value as 'en' | 'de')}>
            <option value="en">English</option><option value="de">Deutsch</option>
          </select>
          <a className="bar-btn button-link" href="/" style={{ color: 'inherit' }}>{t('privacy.back')}</a>
        </div>
      </header>
      <main className="wrap privacy">
        <h1>{text.title}</h1>
        <p className="privacy-intro">{text.intro}</p>
        {text.sections.map((s) => (
          <section key={s.h}>
            <h2>{s.h}</h2>
            {s.p?.map((x) => <p key={x}>{x}</p>)}
            {s.li && <ul>{s.li.map((x) => <li key={x}>{x}</li>)}</ul>}
          </section>
        ))}
        <section>
          <h2>{t('privacy.contact')}</h2>
          <p>{contact ? contactLink(contact) : t('privacy.operator')}</p>
        </section>

        {auth?.signed_in && !deleted && (
          <section className="danger-zone">
            <h2>{t('privacy.delete_title')}</h2>
            <p>{t('privacy.delete_text')}</p>
            <label className="privacy-confirm">
              <span>{t('privacy.delete_confirm', { user: name })}</span>
              <input value={confirm} onChange={(e) => setConfirm(e.target.value)}
                     autoComplete="off" spellCheck={false} placeholder={name} />
            </label>
            {msg && <div className="error">{msg}</div>}
            <button className="danger" disabled={!matches || busy} onClick={remove}>
              {busy ? '…' : t('privacy.delete_button')}
            </button>
          </section>
        )}
        {deleted && (
          <section className="danger-zone done">
            <h2>{t('privacy.deleted_title')}</h2>
            <p>{t('privacy.deleted_text')}</p>
            <a className="button-link primary-link" href="/">{t('privacy.back')}</a>
          </section>
        )}
      </main>
    </>
  )
}
