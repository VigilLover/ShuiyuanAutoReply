<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import ChatView from './views/ChatView.vue'
import SettingsView from './views/SettingsView.vue'

const route = useRoute()
const settingsOpen = computed(() => route.path === '/settings')
</script>

<template>
  <ChatView />
  <SettingsView v-if="settingsOpen" />
</template>
