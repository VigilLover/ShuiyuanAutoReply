<script setup lang="ts">
import { computed } from 'vue'
import {
  PhCaretRight,
  PhCheckCircle,
  PhClock,
  PhImage,
  PhWarningCircle,
  PhWrench,
} from '@phosphor-icons/vue'
import type { RunEvent } from '../api'
import { STOP_REASON_LABELS, formatSeconds, formatTokens, groupRounds, runSummary, type Round, type ToolStep } from '../runs'
import BrandLoader from './BrandLoader.vue'
import PromptEvent from './PromptEvent.vue'

const props = defineProps<{
  events: RunEvent[]
  running?: boolean
  statusLabel?: string
}>()

const rounds = computed(() => groupRounds(props.events))
const summary = computed(() => runSummary(props.events))
const completed = computed(() => !props.running && props.events.some(event => event.type === 'run.completed'))
const failed = computed(() => !props.running && props.events.some(event => event.type === 'run.failed'))
const currentRound = computed(() => rounds.value.at(-1))

const stageLabels: Record<string, string> = {
  'context.topic_loaded': '加载话题',
  'context.style_loaded': '检索历史发言',
  'context.forum_loaded': '加载论坛上下文',
  'memory.loaded': '加载长期记忆',
  'model.prompt_prepared': '准备模型输入',
  'model.started': '等待模型',
  'model.completed': '模型响应完成',
  'tool.started': '调用工具',
  'tool.completed': '工具执行完成',
  'tool.failed': '工具执行失败',
  'image.generated': '生成图片',
  'forum.image_uploaded': '上传论坛图片',
  'forum.reply_publishing': '正在发布回复',
  'forum.reply_published': '发布论坛回复',
  'run.completed': '处理完成',
  'run.failed': '处理失败',
  'run.needs_review': '发送结果待确认',
}

const currentStage = computed(() => {
  const last = props.events.at(-1)
  if (!last) return '准备处理'
  if (last.type === 'model.started') {
    const step = currentRound.value
    return step ? `第 ${step.index} 轮 · 等待模型${step.phase === 'final' ? '（收尾）' : ''}` : '等待模型'
  }
  if (last.type === 'tool.started') return `调用 ${String(last.payload?.name || '工具')}`
  return stageLabels[last.type] || last.type
})

function stringify(value: unknown): string {
  if (typeof value === 'string') return value
  if (value == null) return ''
  try { return JSON.stringify(value, null, 2) } catch { return String(value) }
}

function argsPreview(step: ToolStep): string {
  const args = step.arguments
  if (!args || typeof args !== 'object') return stringify(args)
  return Object.entries(args as Record<string, unknown>)
    .map(([key, value]) => `${key}=${typeof value === 'string' ? value : JSON.stringify(value)}`)
    .join('  ')
}

function prettyOutput(step: ToolStep): string {
  try {
    return JSON.stringify(JSON.parse(step.output), null, 2)
  } catch {
    return step.output
  }
}

function roundLabel(round: Round): string {
  if (round.failed) return `第 ${round.index} 轮 · 模型调用失败`
  if (round.phase === 'final') return `第 ${round.index} 轮 · 收尾回答`
  return `第 ${round.index} 轮${round.tools.length ? ` · ${round.tools.length} 次工具调用` : ' · 直接作答'}`
}

const stopLabel = computed(() => summary.value.stopReason ? (STOP_REASON_LABELS[summary.value.stopReason] || summary.value.stopReason) : '')
</script>

<template>
  <section class="run-progress" :class="{ complete: completed, running, failed }">
    <div class="run-progress-head">
      <BrandLoader v-if="running" compact />
      <PhWarningCircle v-else-if="failed" :size="17" class="run-icon-failed" />
      <PhCheckCircle v-else :size="17" class="run-icon-done" />
      <strong>{{ statusLabel || (running ? '正在执行' : completed ? '已执行' : failed ? '执行失败' : '执行已结束') }}</strong>
      <small v-if="running">{{ currentStage }}</small>
      <template v-else>
        <span class="run-stat"><PhClock :size="12" />{{ formatSeconds(summary.elapsed) }}</span>
        <span class="run-stat">{{ summary.rounds }} 轮 · {{ summary.queries }} 次查询</span>
        <span class="run-stat" :title="`输入 ${formatTokens(summary.inputTokens)} · 输出 ${formatTokens(summary.outputTokens)} · 推理 ${formatTokens(summary.reasoningTokens)} · 缓存命中 ${formatTokens(summary.cacheReadTokens)}`">
          {{ formatTokens(summary.inputTokens + summary.outputTokens) }} tokens
          <em v-if="summary.inputTokens">· 缓存 {{ Math.round((summary.cacheReadTokens / summary.inputTokens) * 100) }}%</em>
        </span>
        <span v-if="stopLabel" class="run-stat">{{ stopLabel }}</span>
      </template>
    </div>

    <details v-if="rounds.length" class="run-history" :open="running">
      <summary>
        <PhCaretRight class="history-caret" :size="13" />
        {{ running ? `已执行 ${rounds.length} 轮` : `查看 ${rounds.length} 轮执行过程` }}
      </summary>
      <ol class="run-round-list">
        <li v-for="round in rounds" :key="round.index" class="run-round" :class="{ failed: round.failed, final: round.phase === 'final' }">
          <header class="run-round-head">
            <strong>{{ roundLabel(round) }}</strong>
            <span v-if="round.elapsed !== undefined" class="run-round-stat"><PhClock :size="11" />{{ formatSeconds(round.elapsed) }}</span>
            <span v-if="round.inputTokens" class="run-round-stat" :title="`推理 ${formatTokens(round.reasoningTokens)} tokens`">
              ↓{{ formatTokens(round.inputTokens) }} ↑{{ formatTokens(round.outputTokens) }}
              <em v-if="round.cacheReadTokens">· 缓存 {{ formatTokens(round.cacheReadTokens) }}</em>
            </span>
            <PromptEvent v-if="round.prompt" :payload="round.prompt.payload" class="run-round-prompt" />
          </header>
          <p v-if="round.failed" class="run-round-error">{{ round.failed }}</p>
          <ul v-if="round.tools.length" class="run-tool-list">
            <li v-for="step in round.tools" :key="step.key" class="run-tool" :class="step.status">
              <details>
                <summary>
                  <PhImage v-if="step.name === 'generate_image'" :size="13" />
                  <PhWrench v-else :size="13" />
                  <strong>{{ step.name }}</strong>
                  <code>{{ argsPreview(step) }}</code>
                  <span class="run-tool-meta">
                    <em v-if="step.elapsed !== undefined">{{ formatSeconds(step.elapsed) }}</em>
                    <em v-if="step.status === 'failed'" class="run-tool-failed">失败</em>
                    <em v-else-if="step.status === 'running'" class="run-tool-running">运行中</em>
                  </span>
                  <PhCaretRight class="step-caret" :size="12" />
                </summary>
                <p v-if="step.hint" class="run-tool-hint">{{ step.hint }}</p>
                <img
                  v-if="step.artifactId"
                  class="run-tool-image"
                  :src="`/api/artifacts/${step.artifactId}`"
                  alt="生成图片"
                  loading="lazy"
                />
                <pre v-if="step.arguments">{{ stringify(step.arguments) }}</pre>
                <pre v-if="step.output">{{ prettyOutput(step) }}</pre>
              </details>
            </li>
          </ul>
        </li>
      </ol>
    </details>
  </section>
</template>
