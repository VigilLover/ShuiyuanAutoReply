<script setup lang="ts">
import { computed } from 'vue'
import type { Message, RunEvent } from '../api'
import { activeRun, forumTimeline, statusLabels, type ForumRun } from '../forum'
import MarkdownContent from './MarkdownContent.vue'
import RunProgress from './RunProgress.vue'
const props = defineProps<{ runs: ForumRun[]; messages: Message[]; events: RunEvent[] }>()
const emit = defineEmits<{ preview: [url: string] }>()
const timeline = computed(() => forumTimeline(props.runs, props.messages))
const eventsByRun = computed(() => {
  const grouped: Record<string, RunEvent[]> = {}
  for (const event of props.events) (grouped[event.run_id] ||= []).push(event)
  return grouped
})
const eventsFor = (id?: string) => (id ? eventsByRun.value[id] || [] : [])
</script>

<template>
  <template v-for="entry in timeline" :key="entry.key">
    <article v-if="entry.run" class="forum-task" :data-run-id="entry.run.id">
      <header class="forum-task-head">
        <strong>{{ entry.run.request.display_name || entry.run.request.username }}</strong>
        <span>@{{ entry.run.request.username }} · #{{ entry.run.request.post_number }}</span>
        <time>{{ new Date(entry.run.started_at).toLocaleTimeString() }}</time>
        <span class="forum-task-status" :class="entry.run.status">{{ statusLabels[entry.run.status] || entry.run.status }}</span>
      </header>
      <div class="forum-instruction">
        <MarkdownContent :content="entry.run.request.content || ''" :attachments="entry.messages.find(m => m.role === 'user')?.attachments || []" @preview="emit('preview', $event)" />
      </div>
      <div v-for="message in entry.messages.filter(m => m.role !== 'user')" :key="message.id" class="forum-answer">
        <strong>{{ message.role === 'assistant' ? 'Shuiyuan Bot' : '系统' }}</strong>
        <MarkdownContent :content="message.content" :attachments="message.attachments" @preview="emit('preview', $event)" />
      </div>
      <div v-if="entry.run.request.handler === 'clear' && eventsFor(entry.run.id).some(e => e.type === 'run.generated')" class="forum-answer">
        <MarkdownContent :content="String(eventsFor(entry.run.id).find(e => e.type === 'run.generated')?.payload.text || '')" />
      </div>
      <p v-if="entry.run.error" class="forum-task-error">{{ entry.run.error }}</p>
      <p v-if="entry.run.status === 'needs_review'" class="forum-task-error">论坛可能已收到回复，请核实帖子后再处理。</p>
      <RunProgress :events="eventsFor(entry.run.id)" :running="activeRun(entry.run)" :status-label="statusLabels[entry.run.status]" />
    </article>
    <template v-else>
      <article v-for="message in entry.messages" :key="message.id" class="message-entry" :class="message.role">
        <div class="message-meta"><strong>{{ message.role === 'assistant' ? 'Shuiyuan Bot' : message.role === 'system' ? '系统' : '论坛用户' }}</strong></div>
        <div class="message-content" :class="{ 'user-surface': message.role === 'user' }">
          <MarkdownContent :content="message.content" :attachments="message.attachments" @preview="emit('preview', $event)" />
          <RunProgress v-if="message.role === 'assistant' && eventsFor(message.run_id).length" :events="eventsFor(message.run_id)" />
        </div>
      </article>
    </template>
  </template>
</template>
