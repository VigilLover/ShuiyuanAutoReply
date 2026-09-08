<script setup lang="ts">
import { computed } from 'vue'
import {
  PhCaretRight,
  PhCircle,
  PhCircleNotch,
  PhClock,
  PhTerminalWindow,
  PhWarningCircle,
} from '@phosphor-icons/vue'
import type { RunEvent } from '../api'
import PromptEvent from './PromptEvent.vue'

const props = defineProps<{
  events: RunEvent[]
  running?: boolean
}>()

const visibleEvents = computed(() => props.events.filter(event => event.type !== 'message.completed'))
const currentEvent = computed(() => visibleEvents.value.at(-1))
const completed = computed(() => !props.running && visibleEvents.value.some(event => event.type === 'run.completed'))

const labels: Record<string, string> = {
  'run.started': '开始处理',
  'context.style_loaded': '检索历史发言',
  'context.style_failed': '历史发言检索失败',
  'context.forum_loaded': '加载论坛上下文',
  'context.forum_skipped': '跳过论坛上下文',
  'memory.loaded': '加载长期记忆',
  'model.prompt_prepared': '准备模型输入',
  'model.started': '调用模型',
  'model.completed': '模型响应完成',
  'usage.recorded': '记录 Token 用量',
  'tool.started': '调用工具',
  'tool.completed': '工具执行完成',
  'tool.failed': '工具执行失败',
  'image.generated': '生成图片',
  'forum.image_uploaded': '上传论坛图片',
  'forum.reply_published': '发布论坛回复',
  'run.completed': '处理完成',
  'run.failed': '处理失败',
}

function label(event?: RunEvent) {
  return event ? labels[event.type] || event.type : '准备处理'
}

function stringify(value: unknown) {
  if (typeof value === 'string') return value
  if (value == null) return ''
  return JSON.stringify(value, null, 2)
}

function detail(event?: RunEvent) {
  if (!event) return ''
  const payload = event.payload || {}
  if (event.type === 'tool.started') {
    const args = stringify(payload.arguments)
    return `${payload.name || 'unknown'}${args ? ` ${args}` : ''}`
  }
  if (event.type === 'tool.completed' || event.type === 'tool.failed') {
    return `${payload.name || 'unknown'}${payload.output ? ` ${stringify(payload.output)}` : ''}`
  }
  if (event.type === 'context.style_loaded') {
    return `${payload.persona || 'persona'} · 命中 ${payload.count ?? 0} 条 · limit ${payload.limit ?? 8}`
  }
  if (event.type === 'context.style_failed') return stringify(payload.message || payload.error)
  if (event.type === 'memory.loaded') return `${payload.chars ?? 0} 字符`
  if (event.type === 'model.prompt_prepared') return `${payload.message_count ?? 0} 条消息 · ${payload.scope || 'unknown'}`
  if (event.type === 'usage.recorded') {
    return `输入 ${payload.input_tokens ?? 0} · 输出 ${payload.output_tokens ?? 0} tokens`
  }
  return Object.keys(payload).length ? stringify(payload) : ''
}

const usage = computed(() => {
  const totals = { input: 0, output: 0 }
  for (const event of visibleEvents.value) {
    if (event.type !== 'usage.recorded') continue
    totals.input += Number(event.payload.input_tokens || 0)
    totals.output += Number(event.payload.output_tokens || 0)
  }
  return totals
})

const duration = computed(() => {
  const timestamps = visibleEvents.value
    .map(event => Date.parse(event.created_at))
    .filter(value => Number.isFinite(value))
  if (timestamps.length < 2) return null
  return Math.max(0, timestamps.at(-1)! - timestamps[0])
})

function durationText(milliseconds: number | null) {
  if (milliseconds == null) return '—'
  if (milliseconds < 1000) return `${milliseconds}ms`
  return `${(milliseconds / 1000).toFixed(milliseconds < 10000 ? 1 : 0)}s`
}

function tokenText(value: number) {
  return new Intl.NumberFormat('zh-CN').format(value)
}

function isFailed(event: RunEvent) {
  return event.type.endsWith('.failed') || event.type === 'run.failed'
}
</script>

<template>
  <section class="run-progress" :class="{ complete: completed, running }">
    <div class="run-progress-head">
      <PhCircleNotch v-if="running" class="run-progress-spinner" :size="17" weight="bold" />
      <PhWarningCircle v-else-if="!completed" :size="18" />
      <strong>{{ running ? '正在执行' : completed ? '已执行' : '执行已结束' }}</strong>
      <template v-if="completed">
        <span class="run-stat"><PhClock :size="13" />{{ durationText(duration) }}</span>
        <span class="run-stat">输入 {{ tokenText(usage.input) }}</span>
        <span class="run-stat">输出 {{ tokenText(usage.output) }} tokens</span>
      </template>
      <small v-else>{{ label(currentEvent) }}</small>
    </div>

    <details v-if="running && currentEvent" class="current-run-step">
      <summary>
        <PhTerminalWindow :size="15" />
        <span>{{ label(currentEvent) }}</span>
        <code>{{ detail(currentEvent) }}</code>
        <PhCaretRight class="step-caret" :size="13" />
      </summary>
      <PromptEvent v-if="currentEvent.type === 'model.prompt_prepared'" :payload="currentEvent.payload" open />
      <pre v-else-if="detail(currentEvent)">{{ detail(currentEvent) }}</pre>
    </details>

    <details v-if="visibleEvents.length" class="run-history">
      <summary>
        <PhCaretRight class="history-caret" :size="13" />
        {{ running ? `查看已执行的 ${visibleEvents.length} 步` : `查看 ${visibleEvents.length} 个执行步骤` }}
      </summary>
      <div class="run-step-list">
        <details v-for="event in visibleEvents" :key="`${event.id}:${event.type}`" class="run-step" :class="{ failed: isFailed(event) }">
          <summary>
            <PhCircle class="run-step-marker" :size="6" weight="fill" />
            <strong>{{ label(event) }}</strong>
            <code>{{ detail(event) }}</code>
            <PhCaretRight class="step-caret" :size="12" />
          </summary>
          <PromptEvent v-if="event.type === 'model.prompt_prepared'" :payload="event.payload" open />
          <pre v-else-if="detail(event)">{{ detail(event) }}</pre>
        </details>
      </div>
    </details>
  </section>
</template>
