import assert from 'node:assert/strict'
import test from 'node:test'
import { applyForumEvent, forumTimeline, mergeRun, runEvents, type ForumEvent, type ForumRun } from '../src/forum.ts'
import type { Message, RunEvent } from '../src/api.ts'

const event = (id: number, run: string, conversation: string, type: string): ForumEvent => ({
  event_id: id, run_id: run, conversation_id: conversation, type,
  created_at: `2026-09-08T12:00:${String(id).padStart(2, '0')}Z`, payload: { content: run, username: 'alice' },
})
test('concurrent runs, event replay and stale snapshots remain isolated', () => {
  const runs: Record<string, ForumRun> = {}, events: Record<string, RunEvent[]> = {}
  applyForumEvent(runs, events, event(1, 'a', 'topic1', 'run.accepted'))
  applyForumEvent(runs, events, event(2, 'b', 'topic2', 'run.accepted'))
  const queued = { ...runs.a }
  applyForumEvent(runs, events, event(3, 'a', 'topic1', 'run.started'))
  applyForumEvent(runs, events, event(4, 'b', 'topic2', 'run.failed'))
  applyForumEvent(runs, events, event(3, 'a', 'topic1', 'run.started'))
  mergeRun(runs, queued)
  assert.equal(runs.a.status, 'running')
  assert.equal(runs.b.status, 'failed')
  assert.equal(events.a.length, 2)
  assert.deepEqual(events.b.map(e => e.id), [2, 4])
})
test('accepted instruction is replaced by persisted pair without duplicating the task', () => {
  const runs: Record<string, ForumRun> = {}, events: Record<string, RunEvent[]> = {}
  applyForumEvent(runs, events, event(1, 'a', 'topic1', 'run.accepted'))
  const message = (id: string, role: string): Message => ({ id, role, run_id: 'a', content: role, attachments: [], created_at: '2026-09-08T12:01:00Z', epoch: 0, status: 'completed' })
  const initial = forumTimeline(Object.values(runs), [])
  const completed = forumTimeline(Object.values(runs), [message('u', 'user'), message('a', 'assistant')])
  assert.equal(completed.length, 1)
  assert.equal(completed[0].key, initial[0].key)
  assert.equal(completed[0].messages.length, 2)
  // Failed / clear requests do not need an assistant message to retain a trace.
  assert.equal(initial[0].run?.id, 'a')
})
test('legacy messages remain visible; saved and live events are deduplicated', () => {
  const legacy: Message = { id: 'old', role: 'assistant', content: 'old', attachments: [], status: 'completed', created_at: '', epoch: 0 }
  assert.equal(forumTimeline([], [legacy])[0].messages[0].id, 'old')
  const saved: RunEvent = { id: 1, run_id: 'a', type: 'tool.started', created_at: '', payload: {} }
  assert.equal(runEvents([saved], [saved, { ...saved, id: 2, run_id: 'b' }], 'a').length, 1)
})
