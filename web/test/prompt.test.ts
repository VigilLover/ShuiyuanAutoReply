import assert from 'node:assert/strict'
import test from 'node:test'
import {
  draftCard,
  effectiveCard,
  formatMoment,
  hasArchivedPrompt,
  lastUsedCard,
  promptModeState,
  shortHash,
  verificationNotice,
  type PromptProfile,
} from '../src/prompt.ts'

const base: PromptProfile = {
  scope: 'forum',
  active_revision: 2,
  draft_changed: false,
  draft: { prompt_mode: 'managed', system_prompt: 'legacy body' },
  prompt_metadata: {
    code_version: '1.0.11',
    rules_version: '3',
    prompt_mode: 'managed',
    prompt_hash: 'aaaa',
  },
}

test('effective revision carries the rules and code version', () => {
  const card = effectiveCard(base)
  assert.equal(card.value, 'r2')
  assert.equal(card.hint, '规则 v3 · 代码 1.0.11')
  assert.equal(effectiveCard({ scope: 'web' }).value, '—')
})

test('draft card separates unsaved, saved and identical drafts', () => {
  assert.deepEqual(draftCard(base, true), {
    label: '草稿',
    value: '未保存',
    hint: '编辑内容还没有写入草稿',
    tone: 'pending',
  })
  assert.equal(draftCard({ ...base, draft_changed: true }, false).value, '待应用')
  assert.equal(draftCard(base, false).value, '一致')
})

test('last used card reports the run time and verification tone', () => {
  const at = new Date()
  at.setHours(9, 5, 0, 0)
  const verified: PromptProfile = {
    ...base,
    last_runtime_used: {
      profile_revision: 2,
      rules_version: '3',
      prompt_hash: 'aaaa',
      at: at.toISOString(),
    },
  }
  assert.equal(lastUsedCard(verified).value, 'r2')
  assert.equal(lastUsedCard(verified).tone, 'ok')
  assert.match(lastUsedCard(verified).hint, /规则 v3 · 今天 09:05/)
  assert.equal(lastUsedCard(base).value, '—')

  const stale: PromptProfile = {
    ...verified,
    prompt_metadata: { ...base.prompt_metadata, prompt_hash: 'bbbb' },
  }
  assert.equal(lastUsedCard(stale).tone, 'pending')
  assert.match(verificationNotice(stale) ?? '', /最近一次任务运行在 r2/)
  assert.equal(verificationNotice(verified), null)
  assert.equal(verificationNotice(base), null)
})

test('legacy and managed modes describe their own rules', () => {
  assert.equal(promptModeState(base).managed, true)
  const legacy = promptModeState({ ...base, draft: { prompt_mode: 'legacy' } })
  assert.equal(legacy.managed, false)
  assert.equal(legacy.tone, 'pending')
  assert.match(legacy.hint, /不会自动升级/)
  // A migrated-but-unapplied draft still edits as managed.
  assert.equal(
    promptModeState({
      ...base,
      draft: { prompt_mode: 'managed' },
      prompt_metadata: { ...base.prompt_metadata, prompt_mode: 'legacy' },
    }).managed,
    true,
  )
})

test('archived prompts and hashes render only when present', () => {
  assert.equal(hasArchivedPrompt(base), true)
  assert.equal(hasArchivedPrompt({ scope: 'web', draft: { system_prompt: '  ' } }), false)
  assert.equal(hasArchivedPrompt(undefined), false)
  assert.equal(shortHash('0123456789abcdef'), '01234567')
  assert.equal(shortHash(null), '')
})

test('an unparseable timestamp degrades to an empty hint part', () => {
  assert.equal(formatMoment('not-a-date'), '')
  assert.equal(formatMoment(null), '')
})
