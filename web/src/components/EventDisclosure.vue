<script setup lang="ts">
import { ref, watch } from 'vue'
import { PhCaretRight } from '@phosphor-icons/vue'

const props = withDefaults(defineProps<{
  title: string
  metadata?: string
  open?: boolean
}>(), { open: false })

const expanded = ref(props.open)

watch(() => props.open, value => { expanded.value = value })

function onToggle(event: Event) {
  expanded.value = (event.currentTarget as HTMLDetailsElement).open
}
</script>

<template>
  <details class="event-disclosure" :open="expanded" @toggle="onToggle">
    <summary>
      <PhCaretRight class="event-disclosure-caret" :size="13" aria-hidden="true" />
      <span class="event-disclosure-title">{{ title }}</span>
      <small v-if="metadata" class="event-disclosure-metadata">{{ metadata }}</small>
    </summary>
    <div v-if="expanded" class="event-disclosure-content">
      <slot />
    </div>
  </details>
</template>
