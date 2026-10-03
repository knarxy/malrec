// Every call to the backend lives here, so the rest of the app never builds
// a URL by hand. In the container nginx proxies /api to the api service; the
// Vite dev server does the same via its proxy config.
const BASE = import.meta.env.VITE_API_BASE ?? '/api'

export type Reason =
  | { kind: 'because_you_liked'; title: string; your_score: number }
  | { kind: 'continues'; title: string; your_score: number }
  | { kind: 'coming'; date: string | null }
  | { kind: 'tags'; tags: string[] }
  | { kind: 'same_franchise' }
  | { kind: 'deep_cut' }
  | {
      kind: 'drivers'
      items: { label: string; weight: number; personal: boolean }[]
      personal_share: number
      has_affinity: boolean
    }

export interface Recommendation {
  rank: number
  mal_id: number
  title: string
  title_en: string | null
  media_type: string | null
  num_episodes: number | null
  season_year: number | null
  season: string | null
  mal_mean: number | null
  mal_popularity: number | null
  picture_medium: string | null
  picture_large: string | null
  synopsis: string | null
  mal_genres: string[]
  mal_studios: string[]
  predicted_score: number
  final_score: number
  novelty: number
  reasons: Reason[]
  /** the viewer's own MAL list status for this title, if any */
  list_status: string | null
  list_score?: number | null
  /** 80 % range of the user's actual rating, and the chance it is 9+ */
  likely?: { low: number; high: number; p9: number } | null
}

export interface Filters {
  media_types?: string[]
  exclude_genres?: string[]
  eps_min?: number
  eps_max?: number
  year_min?: number
  year_max?: number
}

export interface Prefs {
  lang?: 'en' | 'de'
  filters?: Filters
}

export interface Pair { id: number; status: string; outgoing: boolean; other: string }

export interface TogetherItem {
  rank: number
  mal_id: number
  title: string
  title_en: string | null
  media_type: string | null
  num_episodes: number | null
  season_year: number | null
  picture_large: string | null
  picture_medium: string | null
  mal_genres: string[] | null
  you: number
  partner: number
  together: number
}

export interface QuizCard {
  mal_id: number
  title: string
  title_en: string | null
  picture_large: string | null
  picture_medium: string | null
  season_year: number | null
  media_type: string | null
  num_episodes: number | null
  mal_genres: string[] | null
}

export interface Learning {
  status: string
  events: number
  positive: number
  negative: number
  relevance_points_per_sd?: number
  relevance_ci90?: [number, number]
  novelty_points?: number
  novelty_ci90?: [number, number]
  current?: { relevance_weight: number; relevance_weight_personal: number; novelty_weight: number }
}

export interface SyncState {
  last_sync_at: string | null
  next_allowed_in: number
  refresh: 'idle' | 'syncing' | 'rebuilding' | 'failed' | 'timeout'
  refresh_error: string | null
  refreshed_at: number | null
}

export interface Breakdown {
  your_mean: number
  population: number
  personal: number | null
  personal_share: number
  raw: number
  calibration: { slope: number; intercept: number }
  shown: number
  up: { label: string; value: number }[]
  down: { label: string; value: number }[]
  because_you_liked: { title: string; your_score: number; weight: number }[]
  /** each side's own reasons, in that side's points */
  sides?: {
    population: { up: { label: string; value: number }[]; down: { label: string; value: number }[] }
    personal: { up: { label: string; value: number }[]; down: { label: string; value: number }[] } | null
  }
  ranking?: {
    relevance_z: number; relevance_bonus: number; novelty: number; novelty_bonus: number
    memory_bonus?: number; popularity_bonus?: number; risk_penalty?: number
    predicted?: number; order_score?: number
    /** the risk penalty by reason (negative), when there is one */
    risk_parts?: Partial<Record<'long' | 'disagree' | 'unfamiliar' | 'acclaim', number>>
  }
}

export interface AuthState {
  signed_in: boolean
  username?: string
  mal_user_id?: number
  status?: 'pending' | 'approved' | 'rejected' | 'blocked'
  is_admin?: boolean
  picture?: string | null
}

export interface AdminUser {
  id: number
  mal_username: string
  status: string
  requested_at: string | null
  approved_at: string | null
  last_login_at: string | null
  last_sync_at: string | null
  entries: number
  scored: number
  recs: number
  sessions: number
  onboarding: string | null
  is_admin: boolean
}

export interface AdminSystem {
  population_model: { id: number; created_at: string; users: string } | null
  counts: Record<string, number>
  last_backup: string
  jobs: Record<string, { label: string; ok: boolean | null; state: string;
    last_run?: string; tail?: string[] }>
}

export interface Surface {
  name: string
  description: string
}

export interface OnboardStatus {
  state: 'absent' | 'running' | 'done' | 'failed'
  step?: string
  step_index?: number
  step_total?: number
  error?: string | null
  ready: boolean
  username: string
}

export interface Profile {
  user: string
  entries: number
  scored: number
  completed: number
  mean_score: number | null
  top_genres: { genre: string; n: number; lift: number }[]
}

export interface ModelInfo {
  run_id: number
  algo: string
  params: Record<string, unknown>
  metrics: { spearman?: number; rmse?: number; 'ndcg@10'?: number; n_rated?: number }
  baseline: number
  top_features: [string, number][]
}

export interface SearchHit {
  mal_id: number
  title: string
  title_en: string | null
  media_type: string | null
  season_year: number | null
  mal_mean: number | null
  picture_medium: string | null
}

export interface SimilarHit {
  mal_id: number
  title: string
  mal_mean: number | null
  picture_medium: string | null
  similarity: number
}

class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: Record<string, unknown>) {
    super(message)
  }
}

function recQuery(user: string, limit: number, f?: Filters): string {
  const q = new URLSearchParams({ user, limit: String(limit) })
  if (f?.media_types?.length) q.set('types', f.media_types.join(','))
  if (f?.exclude_genres?.length) q.set('exclude_genres', f.exclude_genres.join(','))
  for (const k of ['eps_min', 'eps_max', 'year_min', 'year_max'] as const) {
    if (f?.[k] !== undefined && f[k] !== null) q.set(k, String(f[k]))
  }
  return q.toString()
}

// Sent on every state-changing call. A cross-site form cannot set it and a
// cross-origin fetch cannot send it without a CORS preflight the API refuses,
// so the server can reject forged requests that ride on the session cookie.
const CSRF = { 'X-Requested-With': 'malrec' }

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    credentials: 'same-origin',
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    let detail: Record<string, unknown> | undefined
    try {
      const body = await res.json()
      const d = body?.detail
      if (typeof d === 'string') msg = d
      else if (d && typeof d.message === 'string') { msg = d.message; detail = d }
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(res.status, msg, detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  authMe: () => req<AuthState>('/auth/me'),

  authConfig: () => req<{ redirect_uri: string | null; register_at: string }>('/auth/config'),

  /** Full-page navigation: the OAuth dance has to leave the app. */
  loginUrl: (returnTo = '/') => `${BASE}/auth/login?return_to=${encodeURIComponent(returnTo)}`,

  logout: () => req<AuthState>('/auth/logout', { method: 'POST', headers: CSRF }),

  queue: (mal_id: number) =>
    req<{ mal_id: number; list_status: string | null }>(`/me/queue/${mal_id}`,
      { method: 'POST', headers: CSRF }),

  unqueue: (mal_id: number) =>
    req<{ mal_id: number; list_status: string | null }>(`/me/queue/${mal_id}`,
      { method: 'DELETE', headers: CSRF }),


  onboardStatus: (username: string) =>
    req<OnboardStatus>(`/onboard/${encodeURIComponent(username)}`),

  surfaces: () => req<Surface[]>('/surfaces'),

  profile: (user: string) => req<Profile>(`/profile?user=${encodeURIComponent(user)}`),

  model: (user: string) => req<ModelInfo>(`/model?user=${encodeURIComponent(user)}`),

  recommendations: (user: string, surface: string, limit = 40, filters?: Filters) =>
    req<{ surface: string; description: string; count: number; items: Recommendation[] }>(
      `/recommendations/${surface}?${recQuery(user, limit, filters)}`,
    ),

  explain: (user: string, mal_id: number, surface: string, onList = true) =>
    req<Breakdown>(`/explain/${mal_id}?user=${encodeURIComponent(user)}&surface=${surface}`
      + (onList ? '' : '&list=false')),

  genres: () => req<string[]>('/genres'),

  learning: () => req<Learning>('/model/learning'),

  getPrefs: () => req<Prefs>('/me/prefs'),

  putPrefs: (p: Prefs) =>
    req<Prefs>('/me/prefs', { method: 'PUT', headers: CSRF, body: JSON.stringify(p) }),

  rate: (mal_id: number, score: number, source?: 'quiz') =>
    req<{ mal_id: number; list_status: string | null; list_score: number }>(
      `/me/rate/${mal_id}${source ? `?source=${source}` : ''}`,
      { method: 'POST', headers: CSRF, body: JSON.stringify({ score }) }),

  quiz: (n = 3) => req<{ cards: QuizCard[]; rated_in_round: number }>(`/me/quiz?n=${n}`),

  quizAnswer: (mal_id: number, answer: 'unseen' | 'skip') =>
    req<unknown>(`/me/quiz/${mal_id}/${answer}`, { method: 'POST', headers: CSRF }),

  sync: () => req<{ status: string; next_allowed_in: number }>('/me/sync',
    { method: 'POST', headers: CSRF }),

  syncStatus: () => req<SyncState>('/me/sync'),

  feedback: (_user: string, mal_id: number, action: string, surface?: string) =>
    req<unknown>('/feedback', {
      method: 'POST', headers: CSRF,
      body: JSON.stringify({ mal_id, action, surface }),
    }),

  undoFeedback: (_user: string, mal_id: number) =>
    req<unknown>(`/feedback/${mal_id}`, { method: 'DELETE', headers: CSRF }),

  // ---- watch together
  together: () => req<{ partners: Pair[]; incoming: Pair[]; outgoing: Pair[];
    links: { id: number; created_at: string; expires_at: string }[] }>('/together'),
  togetherLink: () =>
    req<{ url: string; expires_at: string }>('/together/link', { method: 'POST', headers: CSRF }),
  togetherPeek: (token: string) =>
    req<{ inviter: string; own: boolean }>(`/together/link/${encodeURIComponent(token)}`),
  togetherJoin: (token: string) =>
    req<{ id: number; other: string }>(`/together/link/${encodeURIComponent(token)}/accept`,
      { method: 'POST', headers: CSRF }),
  togetherWithdrawLink: (id: number) =>
    req<unknown>(`/together/link/${id}`, { method: 'DELETE', headers: CSRF }),
  togetherInvite: (username: string) =>
    req<{ status: string }>('/together/invite',
      { method: 'POST', headers: CSRF, body: JSON.stringify({ username }) }),
  togetherAccept: (id: number) =>
    req<unknown>(`/together/${id}/accept`, { method: 'POST', headers: CSRF }),
  togetherEnd: (id: number) => req<unknown>(`/together/${id}`, { method: 'DELETE', headers: CSRF }),
  togetherList: (id: number) =>
    req<{ partner: string | null; items: TogetherItem[] }>(`/together/${id}/list`),

  // ---- admin panel
  adminUsers: () => req<AdminUser[]>('/admin/users'),
  adminAction: (id: number, action: 'approve' | 'reject' | 'block' | 'unblock' | 'logout'
    | 'sync' | 'rebuild') =>
    req<unknown>(`/admin/users/${id}/${action}`, { method: 'POST', headers: CSRF }),
  adminDelete: (id: number) =>
    req<unknown>(`/admin/users/${id}`, { method: 'DELETE', headers: CSRF }),
  adminSystem: () => req<AdminSystem>('/admin/system'),
  adminLog: () => req<{ admin: string; action: string; target: string | null; at: string }[]>(
    '/admin/log'),
  adminReport: () => req<{ at: string | null; running: boolean; value: unknown }>('/admin/report'),
  adminRunReport: () => req<unknown>('/admin/report', { method: 'POST', headers: CSRF }),

  /** Any title as a Recommendation-shaped item, e.g. opened from search. */
  animeFor: async (mal_id: number, user: string): Promise<Recommendation> => {
    const a = await req<Record<string, unknown>>(
      `/anime/${mal_id}?user=${encodeURIComponent(user)}`)
    return {
      rank: 0, final_score: 0, novelty: 0, reasons: [],
      mal_genres: [], mal_studios: [], ...a,
      predicted_score: (a.predicted_score as number | null) ?? NaN,
    } as unknown as Recommendation
  },

  search: (q: string) => req<SearchHit[]>(`/search?q=${encodeURIComponent(q)}&limit=12`),

  similar: (mal_id: number) => req<SimilarHit[]>(`/similar/${mal_id}?limit=10`),
}

export { ApiError }
