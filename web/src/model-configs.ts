/** Stored chat/image endpoint configurations, as the settings page sees them. */

export type ModelKind = 'chat' | 'image'
export type ActivateScope = 'web' | 'forum' | 'image'

export interface ModelSecret {
  configured?: boolean
  last_four?: string | null
  source?: 'ui' | 'environment' | null
}

export interface ModelConfigEntry {
  id: string
  kind: ModelKind
  name: string
  base_url: string
  model: string
  api_format?: 'chat_completions' | 'responses' | null
  source: 'default' | 'custom'
  secret?: ModelSecret
}

export interface ModelConfigLibrary {
  chat: ModelConfigEntry[]
  image: ModelConfigEntry[]
  active: Record<ActivateScope, string>
}

export interface ModelConfigDraft {
  id: string | null
  kind: ModelKind
  name: string
  base_url: string
  model: string
  api_format?: 'chat_completions' | 'responses' | null
  api_key: string
}

export interface ProbeState {
  ok: boolean
  models: string[]
  message: string
}

export const KIND_LABELS: Record<ModelKind, string> = {
  chat: '文字模型',
  image: '生图模型',
}

export const EMPTY_LIBRARY: ModelConfigLibrary = {
  chat: [],
  image: [],
  active: { web: 'default', forum: 'default', image: 'default' },
}

export function defaultEntry(kind: ModelKind, entries: ModelConfigEntry[]): ModelConfigEntry | null {
  return entries.find(entry => entry.id === 'default') || null
}

export function activeEntry(
  kind: ModelKind,
  scope: ActivateScope,
  library: ModelConfigLibrary,
): ModelConfigEntry | null {
  const entries = library[kind] || []
  const selected = library.active?.[scope]
  return entries.find(entry => entry.id === selected) || defaultEntry(kind, entries)
}

/** Host only: enough to tell two endpoints apart inside a narrow row. */
export function endpointHost(baseUrl?: string | null, fallback = '官方端点'): string {
  const value = (baseUrl || '').trim()
  if (!value) return fallback
  try {
    const parsed = new URL(value)
    return `${parsed.host}${parsed.pathname.replace(/\/$/, '')}`
  } catch {
    return value.replace(/^https?:\/\//, '').replace(/\/$/, '')
  }
}

export function keyLabel(
  secret?: ModelSecret | null,
  options: { fallback?: boolean } = {},
): string {
  if (!secret || !secret.configured) {
    return options.fallback ? '未配置密钥（回退环境密钥）' : '未配置密钥'
  }
  const suffix = secret.last_four ? ` ····${secret.last_four}` : ''
  return (secret.source === 'environment' ? '环境密钥' : '已保存密钥') + suffix
}

export function newDraft(kind: ModelKind): ModelConfigDraft {
  return {
    id: null,
    kind,
    name: '',
    base_url: '',
    model: '',
    api_format: kind === 'chat' ? 'chat_completions' : null,
    api_key: '',
  }
}

export function draftFromEntry(entry: ModelConfigEntry): ModelConfigDraft {
  return {
    id: entry.id,
    kind: entry.kind,
    name: entry.name,
    base_url: entry.base_url,
    model: entry.model,
    api_format:
      entry.kind === 'chat' ? entry.api_format ?? 'chat_completions' : null,
    api_key: '',
  }
}

/** Null when the draft can be saved; otherwise the reason to show. */
export function draftError(draft: ModelConfigDraft | null): string | null {
  if (!draft) return null
  if (!draft.name.trim()) return '请填写配置名称'
  if (!draft.model.trim()) return '请填写或选择模型名称'
  const baseUrl = draft.base_url.trim()
  if (baseUrl && !/^https?:\/\//.test(baseUrl)) return 'Base URL 需要以 http:// 或 https:// 开头'
  return null
}

export function configPayload(draft: ModelConfigDraft) {
  return {
    kind: draft.kind,
    name: draft.name.trim(),
    base_url: draft.base_url.trim(),
    model: draft.model.trim(),
    api_format: draft.kind === 'chat' ? draft.api_format || 'chat_completions' : null,
    ...(draft.api_key.trim() ? { api_key: draft.api_key.trim() } : {}),
  }
}

/** The list is long and unordered; keep the configured model reachable. */
export function sortModels(models: string[], current?: string): string[] {
  const unique = Array.from(new Set(models.filter(model => Boolean(model))))
  const wanted = (current || '').trim()
  return unique.sort((left, right) => {
    if (left === wanted) return -1
    if (right === wanted) return 1
    return left.localeCompare(right)
  })
}

/** Which apps currently use this entry, for the row badge. */
export function usedBy(
  entry: ModelConfigEntry | null,
  library: ModelConfigLibrary,
): string[] {
  if (!entry) return []
  const scopes: [ActivateScope, string][] = [
    ['web', '网页'],
    ['forum', '论坛'],
    ['image', '全机器人'],
  ]
  return scopes
    .filter(([scope]) => (library.active?.[scope] || 'default') === entry.id)
    .map(([, label]) => label)
}
