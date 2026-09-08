<script setup lang="ts">
import { computed } from 'vue'
import { PhCaretRight } from '@phosphor-icons/vue'

type PromptMessage = {
  role?: string
  content?: unknown
  name?: string
  tool_call_id?: string
  tool_calls?: unknown
}

const props = withDefaults(defineProps<{
  payload: Record<string, unknown>
  open?: boolean
}>(), { open: false })

const messages = computed<PromptMessage[]>(() =>
  Array.isArray(props.payload.messages) ? props.payload.messages as PromptMessage[] : [],
)

function roleLabel(role?: string) {
  return ({ system: 'SYSTEM', human: 'USER', ai: 'ASSISTANT', tool: 'TOOL' } as Record<string, string>)[role || ''] || (role || 'MESSAGE').toUpperCase()
}

function contentText(value: unknown) {
  if (typeof value === 'string') return value
  if (value == null) return ''
  return JSON.stringify(value, null, 2)
}
</script>

<template>
  <details class="prompt-event" :open="open">
    <summary>
      <PhCaretRight class="prompt-event-caret" :size="13" />
      <span>已发送给模型的提示词</span>
      <small>{{ payload.message_count || messages.length }} 条消息 · {{ payload.scope || 'unknown' }}</small>
    </summary>
    <div v-if="messages.length" class="prompt-message-list">
      <section v-for="(message, index) in messages" :key="index" class="prompt-message">
        <div class="prompt-message-role">
          <strong>{{ roleLabel(message.role) }}</strong>
          <small v-if="message.name">{{ message.name }}</small>
        </div>
        <pre>{{ contentText(message.content) }}</pre>
        <pre v-if="message.tool_calls" class="prompt-tool-calls">tool_calls: {{ contentText(message.tool_calls) }}</pre>
      </section>
    </div>
    <pre v-else class="prompt-fallback">{{ contentText(payload.summary || payload) }}</pre>
  </details>
</template>
