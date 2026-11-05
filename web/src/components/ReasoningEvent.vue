<script setup lang="ts">
import EventDisclosure from './EventDisclosure.vue'

defineProps<{ text: string }>()
</script>

<template>
  <EventDisclosure
    v-if="text.trim()"
    class="reasoning-event"
    title="模型思考"
    :metadata="`${text.length.toLocaleString()} 字符`"
  >
    <pre>{{ text }}</pre>
  </EventDisclosure>
</template>
