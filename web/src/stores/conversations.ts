import { defineStore } from 'pinia'
import { api, csrfHeaders, type Conversation, type ConversationDetail } from '../api'

export const useConversations = defineStore('conversations', {
  state: () => ({
    channel: 'web' as 'web' | 'forum', conversations: [] as Conversation[],
    selected: null as ConversationDetail | null, loading: false, running: false,
    error: '', liveEvents: [] as any[], search: '', hasMore: false, lastFailedMessage: '',
  }),
  actions: {
    async load(more = false) {
      this.loading = true; this.error = ''
      try {
        const offset = more ? this.conversations.length : 0
        const page = await api<Conversation[]>(`/api/conversations?channel=${this.channel}&search=${encodeURIComponent(this.search)}&limit=50&offset=${offset}`)
        this.conversations = more ? [...this.conversations, ...page] : page
        this.hasMore = page.length === 50
      }
      catch (error) { this.error = String(error) } finally { this.loading = false }
    },
    async select(id: string) { this.selected = await api(`/api/conversations/${id}`); this.liveEvents = [] },
    async create() {
      const item = await api<Conversation>('/api/conversations', { method: 'POST', body: '{}' })
      await this.load(); await this.select(item.id)
    },
    async rename(title: string) {
      if (!this.selected) return
      await api(`/api/conversations/${this.selected.conversation.id}`, { method: 'PATCH', body: JSON.stringify({ title }) })
      await this.select(this.selected.conversation.id); await this.load()
    },
    async clear() {
      if (!this.selected) return
      await api(`/api/conversations/${this.selected.conversation.id}/clear`, { method: 'POST' })
      await this.select(this.selected.conversation.id)
    },
    async remove() {
      if (!this.selected) return
      await api(`/api/conversations/${this.selected.conversation.id}`, { method: 'DELETE' })
      this.selected = null; await this.load()
    },
    async send(message: string) {
      if (!this.selected || this.selected.conversation.channel !== 'web') return
      this.running = true; this.error = ''; this.liveEvents = []
      try {
        const response = await fetch(`/api/conversations/${this.selected.conversation.id}/messages/stream`, {
          method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', ...csrfHeaders() },
          body: JSON.stringify({ message }),
        })
        if (!response.ok || !response.body) throw new Error(await response.text())
        const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ''
        while (true) {
          const { value, done } = await reader.read(); if (done) break
          buffer += decoder.decode(value, { stream: true })
          const blocks = buffer.split('\n\n'); buffer = blocks.pop() || ''
          for (const block of blocks) {
            const event = block.match(/^event: (.+)$/m)?.[1] || 'message'
            const raw = block.match(/^data: (.+)$/m)?.[1]
            if (raw) {
              const payload = JSON.parse(raw)
              this.liveEvents.push({ type: event, payload })
              if (event === 'stream.error') throw new Error(payload.error || '请求失败')
            }
          }
        }
        this.lastFailedMessage = ''
        await this.select(this.selected.conversation.id); await this.load()
      } catch (error) { this.error = String(error); this.lastFailedMessage = message } finally { this.running = false }
    },
  },
})
