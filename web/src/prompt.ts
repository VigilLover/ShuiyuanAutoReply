/** Prompt profile status derived from /api/settings/profiles payloads. */

export type Tone = 'ok' | 'pending' | 'muted'

export interface PromptMetadata {
  code_version?: string
  code_hash?: string
  prompt_mode?: 'managed' | 'legacy'
  rules_version?: string
  prompt_hash?: string
  migration_required?: boolean
}

export interface RuntimeProfileUsed extends PromptMetadata {
  profile_revision?: number
  scope?: string
  at?: string
}

export interface PromptProfile {
  scope: string
  active_revision?: number
  draft_changed?: boolean
  prompt_metadata?: PromptMetadata | null
  last_runtime_used?: RuntimeProfileUsed | null
  draft?: Record<string, any>
}

export interface StatusCard {
  label: string
  value: string
  hint: string
  tone: Tone
}

export interface ModeState {
  managed: boolean
  label: string
  hint: string
  tone: Tone
}

function joinHint(parts: (string | undefined | null)[]): string {
  return parts.filter(part => Boolean(part)).join(' · ') || '—'
}

export function formatMoment(value?: string | null): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  const pad = (part: number) => String(part).padStart(2, '0')
  const time = `${pad(date.getHours())}:${pad(date.getMinutes())}`
  const sameDay = date.toDateString() === new Date().toDateString()
  return sameDay ? `今天 ${time}` : `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${time}`
}

export function shortHash(value?: string | null, length = 8): string {
  return value ? value.slice(0, length) : ''
}

export function promptModeState(profile: PromptProfile | undefined): ModeState {
  const mode = profile?.draft?.prompt_mode ?? profile?.prompt_metadata?.prompt_mode
  if (mode === 'managed') {
    return {
      managed: true,
      label: '托管规则模式',
      hint: '执行规则随版本升级，人设与补充要求单独维护',
      tone: 'ok',
    }
  }
  return {
    managed: false,
    label: '旧完整提示词模式',
    hint: '程序执行规则不会自动升级；迁移后原文会归档保留',
    tone: 'pending',
  }
}

export function effectiveCard(profile: PromptProfile | undefined): StatusCard {
  const metadata = profile?.prompt_metadata ?? {}
  const revision = profile?.active_revision
  return {
    label: '已生效',
    value: revision ? `r${revision}` : '—',
    hint: joinHint([
      metadata.rules_version ? `规则 v${metadata.rules_version}` : '',
      metadata.code_version ? `代码 ${metadata.code_version}` : '',
    ]),
    tone: 'ok',
  }
}

export function draftCard(profile: PromptProfile | undefined, dirty: boolean): StatusCard {
  if (dirty) {
    return { label: '草稿', value: '未保存', hint: '编辑内容还没有写入草稿', tone: 'pending' }
  }
  if (profile?.draft_changed) {
    return { label: '草稿', value: '待应用', hint: '草稿已保存，应用后生效', tone: 'pending' }
  }
  return { label: '草稿', value: '一致', hint: '草稿与生效配置相同', tone: 'muted' }
}

export function lastUsedCard(profile: PromptProfile | undefined): StatusCard {
  const used = profile?.last_runtime_used
  if (!used) {
    return { label: '最近任务使用', value: '—', hint: '还没有任务使用过该配置', tone: 'muted' }
  }
  return {
    label: '最近任务使用',
    value: used.profile_revision ? `r${used.profile_revision}` : '—',
    hint: joinHint([
      used.rules_version ? `规则 v${used.rules_version}` : '',
      formatMoment(used.at),
    ]),
    tone: used.prompt_hash === profile?.prompt_metadata?.prompt_hash ? 'ok' : 'pending',
  }
}

/** Null when the active prompt has already served a task, or when nothing ran yet. */
export function verificationNotice(profile: PromptProfile | undefined): string | null {
  const used = profile?.last_runtime_used
  const metadata = profile?.prompt_metadata
  if (!used?.prompt_hash || !metadata?.prompt_hash) return null
  if (used.prompt_hash === metadata.prompt_hash) return null
  const revision = used.profile_revision ? `r${used.profile_revision}` : '上一版配置'
  return `当前提示词还没有被任务使用过：最近一次任务运行在 ${revision}，下次回复将使用新版本。`
}

export function hasArchivedPrompt(profile: PromptProfile | undefined): boolean {
  return Boolean(String(profile?.draft?.system_prompt ?? '').trim())
}
