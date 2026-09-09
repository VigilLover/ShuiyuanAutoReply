import type { Message, RunEvent } from './api'

export interface ForumRun {
  id: string; conversation_id: string; request_id: string; status: string
  started_at: string; finished_at?: string; error?: string; last_event_id: number
  request: { content?: string; username?: string; display_name?: string; post_number?: number; topic_id?: number; handler?: string; previous_run_id?: string }
}
export interface ForumEvent {
  event_id: number; conversation_id: string; run_id: string; type: string
  created_at: string; payload: Record<string, unknown>
}
export const activeRun = (run: ForumRun) => ['queued', 'running', 'publishing'].includes(run.status)
export const statusLabels: Record<string, string> = {
  queued: '排队中', running: '执行中', publishing: '发布中', completed: '已完成',
  failed: '执行失败', interrupted: '执行中断', needs_review: '发送结果待确认',
}
const eventStatuses: Record<string, string> = {
  'run.accepted': 'queued', 'run.started': 'running', 'forum.reply_publishing': 'publishing',
  'run.completed': 'completed', 'run.failed': 'failed', 'run.interrupted': 'interrupted', 'run.needs_review': 'needs_review',
}
export function mergeRun(runs: Record<string, ForumRun>, incoming: ForumRun) {
  const old = runs[incoming.id]
  if (!old || incoming.last_event_id >= old.last_event_id) runs[incoming.id] = incoming
}
export function applyForumEvent(runs: Record<string, ForumRun>, events: Record<string, RunEvent[]>, event: ForumEvent) {
  const list = events[event.run_id] ||= []
  if (!list.some(item => item.id === event.event_id)) {
    list.push({ id: event.event_id, run_id: event.run_id, type: event.type, created_at: event.created_at, payload: event.payload })
    list.sort((a, b) => a.id - b.id)
  }
  const old = runs[event.run_id]
  if (old && old.last_event_id >= event.event_id) return
  if (!old && event.type !== 'run.accepted') return
  runs[event.run_id] = {
    ...(old || { id: event.run_id, conversation_id: event.conversation_id, request_id: '', started_at: event.created_at, request: event.payload }),
    status: eventStatuses[event.type] || old?.status || 'queued',
    last_event_id: event.event_id,
    ...(event.payload.error ? { error: String(event.payload.error) } : {}),
  }
}
export function runEvents(saved: RunEvent[], live: RunEvent[], runId: string) {
  return [...new Map([...saved, ...live].filter(e => e.run_id === runId).map(e => [e.id, e])).values()].sort((a, b) => a.id - b.id)
}
export function forumTimeline(runs: ForumRun[], messages: Message[]) {
  // Only new runs have accepted metadata. Historical messages retain their layout.
  const managed = runs.filter(run => run.request.content !== undefined)
  const ids = new Set(managed.map(run => run.id))
  return [
    ...managed.map(run => ({ key: run.id, time: run.started_at, run, messages: messages.filter(m => m.run_id === run.id) })),
    ...messages.filter(m => !m.run_id || !ids.has(m.run_id)).map(message => ({ key: message.id, time: message.created_at, run: undefined, messages: [message] })),
  ].sort((a, b) => a.time.localeCompare(b.time) || a.key.localeCompare(b.key))
}
