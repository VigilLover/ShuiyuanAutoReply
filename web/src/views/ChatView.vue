<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'
import { useConversations } from '../stores/conversations'

const store = useConversations(); const input = ref(''); const editing = ref(false); const title = ref(''); let searchTimer: number | undefined
onMounted(() => store.load())
watch(() => store.selected?.conversation.title, value => { title.value = value || '' })
async function switchChannel(channel: 'web' | 'forum') { store.channel = channel; store.selected = null; await store.load() }
async function send() { const value = input.value.trim(); if (!value || store.running) return; input.value = ''; await store.send(value) }
async function removeConversation() {
  if (window.confirm('永久删除本地记录和图片？')) await store.remove()
}
function searchConversations() {
  window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => store.load(), 250)
}
function eventsForMessage(runId?: string) {
  if (!runId) return []
  return (store.selected?.events || []).filter(event => event.run_id === runId)
}
</script>

<template>
  <main class="chat-layout">
    <aside class="sidebar">
      <div class="channel-tabs"><button :class="{active: store.channel==='web'}" @click="switchChannel('web')">网页对话</button><button :class="{active: store.channel==='forum'}" @click="switchChannel('forum')">论坛记录</button></div>
      <input v-model="store.search" class="session-search" placeholder="搜索会话" @input="searchConversations" />
      <button v-if="store.channel==='web'" class="new-chat" @click="store.create">＋ 新建对话</button>
      <div class="session-list">
        <button v-for="item in store.conversations" :key="item.id" class="session" :class="{selected: store.selected?.conversation.id===item.id}" @click="store.select(item.id)">
          <span>{{ item.title }}</span><small>{{ new Date(item.updated_at).toLocaleString() }}</small>
        </button>
        <p v-if="!store.loading && !store.conversations.length" class="empty-small">暂无记录</p>
        <button v-if="store.hasMore" class="load-more" @click="store.load(true)">加载更多</button>
      </div>
    </aside>
    <section v-if="store.selected" class="conversation">
      <div class="conversation-head">
        <div><input v-if="editing" v-model="title" class="title-input" @keyup.enter="store.rename(title); editing=false"/><h2 v-else>{{ store.selected.conversation.title }}</h2><span class="channel-badge">{{ store.selected.conversation.channel === 'web' ? '网页' : '论坛只读' }}</span></div>
        <div class="head-actions"><button v-if="store.selected.conversation.channel==='web'" @click="editing=!editing">重命名</button><button @click="store.clear">清除上下文</button><button class="danger" @click="removeConversation">删除记录</button></div>
      </div>
      <div class="messages">
        <template v-for="message in store.selected.messages" :key="message.id">
          <div v-if="message.role==='system'" class="system-event" :class="{failed: message.status==='failed'}">{{ message.content }}</div>
          <div v-else class="message-row" :class="message.role">
            <div class="avatar">{{ message.role==='user' ? '你' : '狼' }}</div>
            <div class="bubble"><div class="content">{{ message.content }}</div><img v-for="image in message.attachments" :key="image.artifact_id" :src="image.url" class="generated-image" loading="lazy"/><details v-if="message.role==='assistant' && eventsForMessage(message.run_id).length" class="trace"><summary>执行过程 · {{ eventsForMessage(message.run_id).length }} 项</summary><div v-for="event in eventsForMessage(message.run_id)" :key="event.id" class="trace-event"><span>{{ event.type }}</span><code>{{ JSON.stringify(event.payload) }}</code></div></details><small>{{ new Date(message.created_at).toLocaleTimeString() }}</small></div>
          </div>
        </template>
        <details v-if="store.liveEvents.length" class="trace live" open><summary>正在执行</summary><div v-for="(event,index) in store.liveEvents" :key="index" class="trace-event"><span>{{ event.type }}</span><code>{{ JSON.stringify(event.payload) }}</code></div></details>
        <div v-if="store.error" class="system-event failed">{{ store.error }} <button v-if="store.lastFailedMessage" @click="store.send(store.lastFailedMessage)">重试</button></div>
      </div>
      <form v-if="store.selected.conversation.channel==='web'" class="composer" @submit.prevent="send"><textarea v-model="input" rows="1" placeholder="输入消息，Enter 发送，Shift+Enter 换行" @keydown.enter.exact.prevent="send"></textarea><button :disabled="store.running || !input.trim()">{{ store.running ? '处理中…' : '发送' }}</button></form>
      <div v-else class="readonly-note">论坛自动回复记录仅供查看，不能从这里向论坛发送内容。</div>
    </section>
    <section v-else class="empty-state"><div class="empty-orb">水</div><h2>选择一段对话</h2><p>网页会话与论坛自动回复记录相互隔离。</p></section>
  </main>
</template>
