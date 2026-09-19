import assert from 'node:assert/strict'
import test from 'node:test'
import { eventCategory, formatSeconds, groupRounds, outputHint, runSummary } from '../src/runs.ts'
import type { RunEvent } from '../src/api.ts'

let counter = 0
const ev = (type: string, payload: Record<string, unknown> = {}): RunEvent => ({
  id: ++counter, run_id: 'r', type, payload, created_at: `2026-09-19T00:00:${String(counter).padStart(2, '0')}Z`,
})

test('events are grouped into rounds with tools, usage and outcomes', () => {
  const events = [
    ev('run.started'),
    ev('model.prompt_prepared', { message_count: 3 }),
    ev('model.started'),
    ev('model.completed', { usage: { input_tokens: 100, output_tokens: 40, input_token_details: { cache_read: 60 }, output_token_details: { reasoning: 30 } }, elapsed_seconds: 4.2 }),
    ev('tool.started', { name: 'forum_read', arguments: { post_id: 1 } }),
    ev('tool.started', { name: 'users', arguments: { username: 'a' } }),
    ev('tool.timing', { name: 'users', elapsed_seconds: 0.2 }),
    ev('tool.failed', { name: 'users', output: '{"status":"error","code":"not_found","message":"no","hint":"检查用户名"}' }),
    ev('tool.timing', { name: 'forum_read', elapsed_seconds: 1.5 }),
    ev('tool.completed', { name: 'forum_read', output: '{"status":"ok","items":[]}' }),
    ev('model.prompt_prepared', { message_count: 6 }),
    ev('model.started'),
    ev('model.completed', { usage: { input_tokens: 200, output_tokens: 10 }, elapsed_seconds: 2 }),
    ev('retrieval.finished', { model_rounds: 2, queries: 2, stop_reason: 'answered', elapsed_seconds: 9.5, forum_http_requests: 3 }),
  ]
  const rounds = groupRounds(events)
  assert.equal(rounds.length, 2)
  assert.equal(rounds[0].tools.length, 2)
  assert.equal(rounds[0].tools[1].status, 'failed')
  assert.equal(rounds[0].tools[1].hint, '检查用户名')
  assert.equal(rounds[0].tools[0].elapsed, 1.5)
  assert.equal(rounds[0].cacheReadTokens, 60)
  assert.equal(rounds[0].reasoningTokens, 30)
  const summary = runSummary(events)
  assert.equal(summary.rounds, 2)
  assert.equal(summary.stopReason, 'answered')
  assert.equal(summary.elapsed, 9.5)
  assert.equal(summary.inputTokens, 300)
  assert.equal(summary.forumRequests, 3)
})

test('generated image artifacts attach to the generate_image step', () => {
  const events = [
    ev('model.prompt_prepared'),
    ev('model.started'),
    ev('model.completed', {}),
    ev('tool.started', { name: 'generate_image', arguments: { prompt: 'x' } }),
    ev('tool.completed', { name: 'generate_image', output: '{"status":"ok","artifact":"artifact://g1"}' }),
    ev('image.generated', { artifact_id: 'g1', byte_count: 10 }),
  ]
  assert.equal(groupRounds(events)[0].tools[0].artifactId, 'g1')
})

test('helpers format categories, hints and durations', () => {
  assert.equal(eventCategory('tool.started'), '工具')
  assert.equal(eventCategory('usage.recorded'), '模型')
  assert.equal(outputHint('plain text'), undefined)
  assert.equal(outputHint('{"status":"error","message":"bad"}'), 'bad')
  assert.equal(formatSeconds(0.25), '250ms')
  assert.equal(formatSeconds(75), '1m15s')
})
