import assert from 'node:assert/strict'
import test from 'node:test'
import {
  activeEntry,
  configPayload,
  defaultEntry,
  draftError,
  draftFromEntry,
  endpointHost,
  keyLabel,
  newDraft,
  sortModels,
  usedBy,
  type ModelConfigLibrary,
} from '../src/model-configs.ts'

const entry = (over: Partial<Parameters<typeof draftFromEntry>[0]> = {}): any => ({
  id: 'cfg-1',
  kind: 'chat',
  name: '聚合站',
  base_url: 'https://4router.example/v1',
  model: 'deepseek-v4-vision',
  api_format: 'chat_completions',
  source: 'custom',
  secret: { configured: true, last_four: '9999', source: 'ui' },
  ...over,
})

const library = (active: Partial<Record<'web' | 'forum' | 'image', string>> = {}): ModelConfigLibrary => ({
  chat: [
    { id: 'default', kind: 'chat', name: '默认', base_url: '', model: 'official', source: 'default' },
    entry(),
  ],
  image: [
    { id: 'default', kind: 'image', name: '默认生图', base_url: '', model: 'gpt-image-2', source: 'default' },
  ],
  active: { web: 'default', forum: 'default', image: 'default', ...active },
})

test('the active entry matches the selection and falls back to the default', () => {
  assert.equal(activeEntry('chat', 'web', library())?.id, 'default')
  assert.equal(activeEntry('chat', 'web', library({ web: 'cfg-1' }))?.name, '聚合站')
  // A removed configuration must not leave the row blank.
  assert.equal(activeEntry('chat', 'forum', library({ forum: 'gone' }))?.id, 'default')
  assert.equal(defaultEntry('image', library().image)?.model, 'gpt-image-2')
})

test('endpoint host stays short and readable', () => {
  assert.equal(endpointHost('https://4router.example/v1/'), '4router.example/v1')
  assert.equal(endpointHost('https://api.deepseek.com'), 'api.deepseek.com')
  assert.equal(endpointHost(''), '官方端点')
  assert.equal(endpointHost('not a url'), 'not a url')
})

test('key labels describe where the key comes from', () => {
  assert.equal(keyLabel({ configured: true, last_four: '9999', source: 'ui' }), '已保存密钥 ····9999')
  assert.equal(
    keyLabel({ configured: true, last_four: '1234', source: 'environment' }),
    '环境密钥 ····1234',
  )
  assert.equal(keyLabel({ configured: false }), '未配置密钥')
  assert.equal(keyLabel(null), '未配置密钥')
  assert.equal(
    keyLabel({ configured: false }, { fallback: true }),
    '未配置密钥（回退环境密钥）',
  )
})

test('draft validation names the missing field', () => {
  const draft = newDraft('chat')
  assert.equal(draftError(draft), '请填写配置名称')
  draft.name = 'A'
  assert.equal(draftError(draft), '请填写或选择模型名称')
  draft.model = 'm'
  assert.equal(draftError(draft), null)
  draft.base_url = '4router.example'
  assert.match(draftError(draft) ?? '', /https/)
  draft.base_url = ''
  assert.equal(draftError(draft), null)
})

test('payload trims fields and omits an untouched key', () => {
  const draft = draftFromEntry(entry())
  assert.deepEqual(configPayload(draft), {
    kind: 'chat',
    name: '聚合站',
    base_url: 'https://4router.example/v1',
    model: 'deepseek-v4-vision',
    api_format: 'chat_completions',
  })
  draft.api_key = ' sk-new '
  assert.equal(configPayload(draft).api_key, 'sk-new')
  const image = configPayload(newDraft('image'))
  assert.equal(image.kind, 'image')
  assert.equal(image.api_format, null)
})

test('image drafts carry no api format', () => {
  const draft = draftFromEntry(entry({ kind: 'image', api_format: 'responses' }))
  assert.equal(draft.api_format, null)
  assert.equal(draftError(draft), null)
})

test('model list keeps the current model first and de-duplicates', () => {
  assert.deepEqual(sortModels(['b', 'a', 'b'], 'b'), ['b', 'a'])
  assert.deepEqual(sortModels([], 'x'), [])
  assert.deepEqual(sortModels(['', 'z'], ''), ['z'])
})

test('usedBy reports every scope an entry serves', () => {
  assert.deepEqual(usedBy(entry(), library({ web: 'cfg-1' })), ['网页'])
  assert.deepEqual(usedBy(entry(), library({ web: 'cfg-1', forum: 'cfg-1' })), ['网页', '论坛'])
  assert.deepEqual(usedBy(entry({ kind: 'image', id: 'img' }), library({ image: 'img' })), ['全机器人'])
})
