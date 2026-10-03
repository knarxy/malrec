// Watch together: invite another approved account; once they accept, a shared
// list ranked for both of you. Only predicted scores are shared - never your
// list or ratings.
import { useCallback, useEffect, useState } from 'react'
import { api, ApiError, type Pair, type TogetherItem } from './api'
import { useI18n } from './i18n'

const LINK_KEY = 'malrec.together'

/** An invitation link (?together=token) is taken out of the address bar at
 *  once and kept locally, so it survives signing in and waiting for approval
 *  without lingering in the URL or browser history. */
export function captureTogetherToken(): string | null {
  const params = new URLSearchParams(window.location.search)
  const fromUrl = params.get('together')
  if (fromUrl) {
    params.delete('together')
    const qs = params.toString()
    window.history.replaceState(null, '', window.location.pathname + (qs ? `?${qs}` : ''))
    try { localStorage.setItem(LINK_KEY, fromUrl) } catch { /* storage blocked */ }
    return fromUrl
  }
  try { return localStorage.getItem(LINK_KEY) } catch { return null }
}

export function forgetTogetherToken() {
  try { localStorage.removeItem(LINK_KEY) } catch { /* storage blocked */ }
}

type Overview = Awaited<ReturnType<typeof api.together>>

export function Together({ onClose, linkToken, onLinkDone }:
  { onClose: () => void; linkToken?: string | null; onLinkDone?: () => void }) {
  const { t, label } = useI18n()
  const [state, setState] = useState<Overview | null>(null)
  const [link, setLink] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [incomingLink, setIncomingLink] = useState<{ inviter: string; own: boolean } | 'bad' | null>(null)
  const [name, setName] = useState('')
  const [msg, setMsg] = useState<string | null>(null)
  const [open, setOpen] = useState<Pair | null>(null)
  const [items, setItems] = useState<TogetherItem[] | null>(null)

  const load = useCallback(() => {
    api.together().then(setState).catch((e) => setMsg(e instanceof ApiError ? e.message : t('Failed to load')))
  }, [t])
  useEffect(() => { load() }, [load])

  useEffect(() => {
    if (!linkToken) return
    api.togetherPeek(linkToken).then(setIncomingLink).catch(() => setIncomingLink('bad'))
  }, [linkToken])

  function doneWithLink() {
    forgetTogetherToken()
    setIncomingLink(null)
    onLinkDone?.()
  }

  async function join() {
    if (!linkToken) return
    try {
      const r = await api.togetherJoin(linkToken)
      doneWithLink()
      load()
      setOpen({ id: r.id, other: r.other, status: 'accepted', outgoing: false })
    } catch {
      setIncomingLink('bad')
    }
  }

  async function makeLink() {
    try {
      const r = await api.togetherLink()
      setLink(r.url); setCopied(false)
      load()
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : t('Failed to load'))
    }
  }

  async function copyLink() {
    if (!link) return
    try { await navigator.clipboard.writeText(link); setCopied(true) } catch { /* select manually */ }
  }

  async function withdrawLinks() {
    if (!state) return
    await Promise.all(state.links.map((l) => api.togetherWithdrawLink(l.id)))
    setLink(null)
    load()
  }

  useEffect(() => {
    if (!open) return
    setItems(null)
    api.togetherList(open.id).then((r) => setItems(r.items))
      .catch((e) => { setMsg(e instanceof ApiError ? e.message : t('Failed to load')); setItems([]) })
  }, [open, t])

  async function invite(e: React.FormEvent) {
    e.preventDefault()
    const u = name.trim()
    if (!u) return
    try {
      await api.togetherInvite(u)
      setMsg(t('together.sent', { user: u }))
      setName('')
      load()
    } catch (err) {
      setMsg(err instanceof ApiError ? err.message : t('Failed to load'))
    }
  }

  async function act(p: Pair, what: 'accept' | 'end') {
    if (what === 'end' && !window.confirm(t('together.confirm_end', { user: p.other }))) return
    try {
      if (what === 'accept') await api.togetherAccept(p.id)
      else { await api.togetherEnd(p.id); if (open?.id === p.id) setOpen(null) }
      load()
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : t('Failed to load'))
    }
  }

  return (
    <div className="drawer-bg" onClick={onClose}>
      <div className="drawer together" onClick={(e) => e.stopPropagation()}>
        <button className="close" onClick={onClose}>{t('Close')}</button>
        <h2>{t('Watch together')}</h2>
        <p className="surface-note">{t('together.sub')}</p>
        {msg && <div className="notice">{msg}</div>}
        {incomingLink && incomingLink !== 'bad' && !incomingLink.own && (
          <div className="notice invite-banner">
            <span>{t('together.link_from', { user: incomingLink.inviter })}</span>
            <span className="admin-actions">
              <button className="primary" onClick={join}>{t('together.accept')}</button>
              <button onClick={doneWithLink}>{t('together.decline')}</button>
            </span>
          </div>
        )}
        {incomingLink && incomingLink !== 'bad' && incomingLink.own && (
          <div className="notice">
            {t('together.link_own')}{' '}
            <button className="button-link" onClick={doneWithLink}>OK</button>
          </div>
        )}
        {incomingLink === 'bad' && (
          <div className="notice">
            {t('together.link_bad')}{' '}
            <button className="button-link" onClick={doneWithLink}>OK</button>
          </div>
        )}

        {!open && state && (
          <>
            <form className="row" onSubmit={invite} style={{ marginTop: 12 }}>
              <input value={name} onChange={(e) => setName(e.target.value)} maxLength={24}
                     placeholder={t('together.placeholder')} spellCheck={false} autoCapitalize="none" />
              <button className="primary" type="submit" disabled={!name.trim()}>{t('together.invite')}</button>
            </form>
            <div className="link-invite">
              <span className="surface-note">{t('together.or')}</span>
              <button onClick={makeLink}>{t('together.make_link')}</button>
              {state.links.length > 0 && (
                <span className="surface-note">
                  {t('together.open_links', { n: state.links.length })} ·{' '}
                  <button className="button-link" onClick={withdrawLinks}>{t('together.withdraw_links')}</button>
                </span>
              )}
            </div>
            {link && (
              <div className="link-box">
                <input readOnly value={link} onFocus={(e) => e.target.select()} />
                <button onClick={copyLink}>{copied ? t('together.copied') : t('together.copy')}</button>
                {'share' in navigator && (
                  <button onClick={() => navigator.share({ title: 'malrec', url: link }).catch(() => {})}>
                    {t('together.share')}
                  </button>
                )}
                <p className="surface-note">{t('together.link_note')}</p>
              </div>
            )}
            {state.incoming.length > 0 && <h3>{t('together.incoming')}</h3>}
            {state.incoming.map((p) => (
              <div className="admin-row" key={p.id}>
                <b>{p.other}</b>
                <div className="admin-actions">
                  <button className="primary" onClick={() => act(p, 'accept')}>{t('together.accept')}</button>
                  <button onClick={() => act(p, 'end')}>{t('together.decline')}</button>
                </div>
              </div>
            ))}
            <h3>{t('together.partners')}</h3>
            {state.partners.length === 0 && <p className="surface-note">{t('together.none')}</p>}
            {state.partners.map((p) => (
              <div className="admin-row" key={p.id}>
                <b>{p.other}</b>
                <div className="admin-actions">
                  <button className="primary" onClick={() => setOpen(p)}>{t('together.show')}</button>
                  <button onClick={() => act(p, 'end')}>{t('together.end')}</button>
                </div>
              </div>
            ))}
            {state.outgoing.length > 0 && <h3>{t('together.outgoing')}</h3>}
            {state.outgoing.map((p) => (
              <div className="admin-row" key={p.id}>
                <span>{p.other}</span>
                <button onClick={() => act(p, 'end')}>{t('together.withdraw')}</button>
              </div>
            ))}
          </>
        )}

        {open && (
          <>
            <button onClick={() => setOpen(null)} style={{ marginTop: 10 }}>← {t('together.back')}</button>
            <h3>{t('together.with', { user: open.other })}</h3>
            {items === null && <p className="surface-note">{t('Loading…')}</p>}
            {items?.length === 0 && <p className="empty">{t('together.empty')}</p>}
            {items?.map((it) => (
              <div className="together-item" key={it.mal_id}>
                {(it.picture_medium || it.picture_large) &&
                  <img src={it.picture_medium ?? it.picture_large!} alt="" loading="lazy" />}
                <div>
                  <b>{it.title_en || it.title}</b>
                  <div className="meta">
                    {it.season_year && <span>{it.season_year}</span>}
                    {it.media_type && <span>{it.media_type.toUpperCase()}</span>}
                    {it.num_episodes && <span>{it.num_episodes} {t('ep')}</span>}
                  </div>
                  <div className="together-scores">
                    <span>{t('together.you')} <b>{it.you.toFixed(1)}</b></span>
                    <span>{open.other} <b>{it.partner.toFixed(1)}</b></span>
                  </div>
                  <div className="chips">
                    {(it.mal_genres ?? []).slice(0, 3).map((g) => <span className="chip" key={g}>{label(g)}</span>)}
                  </div>
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  )
}
