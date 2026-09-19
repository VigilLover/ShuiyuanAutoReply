/** Group a run's flat event list into model rounds for the execution trace. */

import type { RunEvent } from './api'

export interface ToolStep {
  key: string
  name: string
  arguments: unknown
  output: string
  status: 'running' | 'completed' | 'failed'
  elapsed?: number
  hint?: string
  artifactId?: string
}

export interface Round {
  index: number
  startedAt?: string
  elapsed?: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  phase?: string
  failed?: string
  prompt?: RunEvent
  tools: ToolStep[]
}

export interface RunSummary {
  rounds: number
  queries: number
  stopReason?: string
  elapsed?: number
  forumRequests?: number
  imageDownloads?: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
}

export const STOP_REASON_LABELS: Record<string, string> = {
  answered: '模型主动作答',
  no_new_evidence: '连续无新证据',
  query_budget: '查询次数用尽',
  model_budget: '模型轮次用尽',
  time_budget: '时间预算用尽',
  tool_time_budget: '工具批次超时',
  model_or_time_failure: '模型调用失败后收尾',
  source_complete: '来源已读完',
}

export const EVENT_CATEGORIES: Record<string, string> = {
  run: '任务',
  context: '上下文',
  memory: '上下文',
  model: '模型',
  usage: '模型',
  tool: '工具',
  retrieval: '检索',
  image: '图片',
  forum: '论坛',
  runtime: '运行时',
}

export function eventCategory(type: string): string {
  return EVENT_CATEGORIES[type.split('.')[0]] || '其他'
}

function num(value: unknown): number {
  const n = Number(value)
  return Number.isFinite(n) ? n : 0
}

function usageOf(payload: Record<string, unknown>) {
  const usage = (payload.usage ?? payload) as Record<string, any>
  return {
    input: num(usage.input_tokens),
    output: num(usage.output_tokens),
    reasoning: num(usage.output_token_details?.reasoning ?? usage.output_tokens_details?.reasoning_tokens),
    cacheRead: num(usage.input_token_details?.cache_read ?? usage.input_tokens_details?.cached_tokens),
  }
}

/** Parse a tool output preview into a hint when it is a structured error. */
export function outputHint(output: string): string | undefined {
  try {
    const parsed = JSON.parse(output)
    if (parsed && typeof parsed === 'object') {
      if (typeof parsed.hint === 'string') return parsed.hint
      if (parsed.status === 'error' && typeof parsed.message === 'string') return parsed.message
    }
  } catch {
    /* plain text output */
  }
  return undefined
}

export function groupRounds(events: RunEvent[]): Round[] {
  const rounds: Round[] = []
  let current: Round | undefined
  const openTools: ToolStep[] = []

  const ensureRound = () => {
    if (!current) {
      current = { index: rounds.length + 1, inputTokens: 0, outputTokens: 0, reasoningTokens: 0, cacheReadTokens: 0, tools: [] }
      rounds.push(current)
    }
    return current
  }

  for (const event of events) {
    const payload = event.payload || {}
    switch (event.type) {
      case 'model.prompt_prepared': {
        // A prompt starts a new round unless the current one has not called the model yet.
        if (current && current.startedAt) current = undefined
        const round = ensureRound()
        round.prompt = event
        break
      }
      case 'model.started': {
        const round = ensureRound()
        round.startedAt = event.created_at
        break
      }
      case 'model.completed': {
        const round = ensureRound()
        const usage = usageOf(payload)
        round.elapsed = num(payload.elapsed_seconds)
        round.inputTokens += usage.input
        round.outputTokens += usage.output
        round.reasoningTokens += usage.reasoning
        round.cacheReadTokens += usage.cacheRead
        break
      }
      case 'model.failed': {
        const round = ensureRound()
        round.failed = String(payload.error || '模型调用失败')
        round.phase = String(payload.phase || round.phase || '')
        break
      }
      case 'retrieval.progress': {
        const round = ensureRound()
        round.phase = String(payload.phase || round.phase || '')
        break
      }
      case 'tool.started': {
        const round = ensureRound()
        const step: ToolStep = {
          key: `${event.id}`,
          name: String(payload.name || 'unknown'),
          arguments: payload.arguments,
          output: '',
          status: 'running',
        }
        round.tools.push(step)
        openTools.push(step)
        break
      }
      case 'tool.timing': {
        const step = openTools.find(item => item.name === payload.name && item.elapsed === undefined)
        if (step) step.elapsed = num(payload.elapsed_seconds)
        break
      }
      case 'tool.completed':
      case 'tool.failed': {
        const index = openTools.findIndex(item => item.name === payload.name && item.status === 'running')
        const step = index >= 0 ? openTools.splice(index, 1)[0] : undefined
        if (step) {
          step.output = String(payload.output ?? '')
          step.status = event.type === 'tool.failed' ? 'failed' : 'completed'
          step.hint = outputHint(step.output)
        }
        break
      }
      case 'image.generated': {
        const round = ensureRound()
        const step = [...round.tools].reverse().find(item => item.name === 'generate_image')
        if (step) step.artifactId = String(payload.artifact_id || '')
        break
      }
      default:
        break
    }
  }
  return rounds
}

export function runSummary(events: RunEvent[]): RunSummary {
  const rounds = groupRounds(events)
  const summary: RunSummary = {
    rounds: rounds.length,
    queries: 0,
    inputTokens: rounds.reduce((total, round) => total + round.inputTokens, 0),
    outputTokens: rounds.reduce((total, round) => total + round.outputTokens, 0),
    reasoningTokens: rounds.reduce((total, round) => total + round.reasoningTokens, 0),
    cacheReadTokens: rounds.reduce((total, round) => total + round.cacheReadTokens, 0),
  }
  const finished = [...events].reverse().find(event => event.type === 'retrieval.finished' || event.type === 'retrieval.progress')
  if (finished) {
    const payload = finished.payload || {}
    summary.rounds = num(payload.model_rounds) || summary.rounds
    summary.queries = num(payload.queries)
    summary.stopReason = payload.stop_reason ? String(payload.stop_reason) : undefined
    summary.forumRequests = num(payload.forum_http_requests)
    summary.imageDownloads = num(payload.image_downloads)
    if (finished.type === 'retrieval.finished') summary.elapsed = num(payload.elapsed_seconds)
  }
  if (summary.elapsed === undefined) {
    const stamps = events.map(event => Date.parse(event.created_at)).filter(Number.isFinite)
    if (stamps.length >= 2) summary.elapsed = (Math.max(...stamps) - Math.min(...stamps)) / 1000
  }
  return summary
}

export function formatSeconds(value?: number): string {
  if (value === undefined || !Number.isFinite(value)) return '—'
  if (value < 1) return `${Math.round(value * 1000)}ms`
  if (value < 60) return `${value.toFixed(value < 10 ? 1 : 0)}s`
  const minutes = Math.floor(value / 60)
  return `${minutes}m${Math.round(value - minutes * 60)}s`
}

export function formatTokens(value: number): string {
  return new Intl.NumberFormat('zh-CN').format(value)
}
