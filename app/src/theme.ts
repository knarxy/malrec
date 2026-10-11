// Light or dark. Nothing chosen follows the system setting; a choice is kept
// in this browser (so the first paint is right) and with the account.
export type Theme = 'light' | 'dark'

const KEY = 'malrec.theme'

export function storedTheme(): Theme | null {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch { return null }
}

export function applyTheme(t: Theme | null) {
  const root = document.documentElement
  if (t) root.dataset.theme = t
  else delete root.dataset.theme
}

export function setTheme(t: Theme) {
  try { localStorage.setItem(KEY, t) } catch { /* storage blocked */ }
  applyTheme(t)
}

export function effectiveTheme(): Theme {
  const d = document.documentElement.dataset.theme
  if (d === 'light' || d === 'dark') return d
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}
