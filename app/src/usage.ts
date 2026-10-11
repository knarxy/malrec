// What the user actually sees and opens, reported in small batches
// (malrec.usage). Only the account's own lists report - the caller decides.
// Each thing is sent once per page load; the server counts per day.
import { api, type UsageEvent } from './api'

const FLUSH_MS = 5000
const MAX_BATCH = 60

let queue: UsageEvent[] = []
let timer: number | undefined
const sent = new Set<string>()

function flush() {
  window.clearTimeout(timer)
  timer = undefined
  while (queue.length) {
    const batch = queue.splice(0, MAX_BATCH)
    api.events(batch).catch(() => { /* usage data is best effort */ })
  }
}

if (typeof document !== 'undefined') {
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush()
  })
}

export function track(e: UsageEvent, once = true) {
  const key = `${e.kind}|${e.surface ?? ''}|${e.mal_id ?? 0}`
  if (once && sent.has(key)) return
  sent.add(key)
  queue.push(e)
  if (queue.length >= MAX_BATCH) flush()
  else if (timer === undefined) timer = window.setTimeout(flush, FLUSH_MS)
}

/** Calls `onSeen` once the element has been at least half on screen for a
 *  second. Returns the cleanup. */
export function whenSeen(el: Element, onSeen: () => void): () => void {
  if (typeof IntersectionObserver === 'undefined') return () => {}
  let t: number | undefined
  const obs = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) {
      t = window.setTimeout(() => { obs.disconnect(); onSeen() }, 1000)
    } else {
      window.clearTimeout(t)
    }
  }, { threshold: 0.5 })
  obs.observe(el)
  return () => { window.clearTimeout(t); obs.disconnect() }
}
