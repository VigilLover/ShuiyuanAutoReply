import assert from 'node:assert/strict'
import { registerHooks } from 'node:module'
import test from 'node:test'
import { createPinia, setActivePinia } from 'pinia'
import type { ConversationDetail } from '../src/api.ts'

// The app resolves extensionless imports through Vite; use the same TS source in Node.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (context.parentURL?.endsWith('/src/stores/conversations.ts') && specifier === '../api') {
      return nextResolve('../api.ts', context)
    }
    return nextResolve(specifier, context)
  },
})
const { useConversations } = await import('../src/stores/conversations.ts')

Object.defineProperty(globalThis, 'sessionStorage', { configurable: true, value: { getItem: () => null, setItem: () => {} } })

function detail(): ConversationDetail {
  return {
    conversation: { id: 'c', channel: 'web', title: 'test', updated_at: '', external_id: 'web', bot_id: 'bot', persona_id: 'bot', title_custom: false },
    messages: [], events: [],
  }
}

test('failed web request reloads persisted reasoning and keeps retry text, files and previews', async t => {
  setActivePinia(createPinia())
  const store = useConversations()
  store.selected = detail()
  const file = new File(['image'], 'fixture.png', { type: 'image/png' })
  const thought = 'full reasoning\n'.repeat(10000)
  const saved = detail()
  saved.messages = [{ id: 'failure', role: 'system', content: '请求失败', status: 'failed', run_id: 'r', created_at: '2099-01-01T00:00:00Z', epoch: 0, attachments: [] }]
  saved.events = [{ id: 1, run_id: 'r', type: 'model.reasoning', payload: { text: thought }, created_at: '' }, { id: 2, run_id: 'r', type: 'run.failed', payload: { error: 'empty answer' }, created_at: '' }]
  const revoked: string[] = []
  t.mock.method(URL, 'createObjectURL', () => 'blob:preview')
  t.mock.method(URL, 'revokeObjectURL', (url: string) => revoked.push(url))
  const fetch = t.mock.method(globalThis, 'fetch', async (path: string, init?: RequestInit) => {
    if (path.endsWith('/messages/stream')) {
      assert.ok(init?.body instanceof FormData)
      return new Response('event: stream.error\ndata: {"error":"empty answer"}\n\n')
    }
    assert.equal(path, '/api/conversations/c')
    return Response.json(saved)
  })
  await store.send('请总结', [file])
  assert.equal(fetch.mock.callCount(), 2)
  assert.equal(store.running, false)
  assert.match(store.error, /empty answer/)
  assert.equal(store.lastFailedMessage, '请总结')
  assert.equal(store.lastFailedFiles[0], file)
  assert.equal(store.selected?.events[0].payload.text, thought)
  const pending = store.selected?.messages.find(message => message.id.startsWith('local:'))
  assert.equal(pending?.status, 'failed')
  assert.equal(pending?.content, '请总结')
  assert.equal(pending?.attachments[0].url, 'blob:preview')
  assert.deepEqual(revoked, [])
  assert.equal(store.selected?.messages.at(-1)?.run_id, 'r')

  fetch.mock.mockImplementation(async (path: string) => {
    if (path.endsWith('/messages/stream')) return new Response('event: message.completed\ndata: {"text":"done"}\n\n')
    if (path === '/api/conversations/c') return Response.json(detail())
    return Response.json([])
  })
  await store.send(store.lastFailedMessage, store.lastFailedFiles)
  assert.ok(revoked.includes('blob:preview'))
  assert.equal(store.lastFailedMessage, '')
  assert.equal(store.lastFailedFiles.length, 0)
  assert.equal(store.liveEvents.length, 0)
})

test('a failed history refresh retains received reasoning and the original stream error', async t => {
  setActivePinia(createPinia())
  const store = useConversations()
  store.selected = detail()
  t.mock.method(globalThis, 'fetch', async (path: string) => {
    if (path.endsWith('/messages/stream')) return new Response('event: model.reasoning\ndata: {"text":"saved thought","run_id":"r","event_id":1}\n\nevent: stream.error\ndata: {"error":"empty answer"}\n\n')
    throw new Error('history unavailable')
  })
  await store.send('hello')
  assert.equal(store.running, false)
  assert.match(store.error, /empty answer/)
  assert.equal(store.liveEvents[0].payload.text, 'saved thought')
  assert.equal(store.selected?.messages[0].status, 'failed')
})
