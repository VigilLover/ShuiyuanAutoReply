<script setup lang="ts">
import { ref } from 'vue'
import { PhCaretRight } from '@phosphor-icons/vue'

defineProps<{ text: string }>()
const expanded = ref(false)

function onToggle(event: Event) {
  expanded.value = (event.currentTarget as HTMLDetailsElement).open
}
</script>

<template>
  <details v-if="text.trim()" class="reasoning-event" @toggle="onToggle">
    <summary>
      <PhCaretRight class="reasoning-event-caret" :size="13" />
      <span>模型思考</span>
      <small>{{ text.length.toLocaleString() }} 字符</small>
    </summary>
    <pre v-if="expanded">{{ text }}</pre>
  </details>
</template>
