import { defineStore } from 'pinia'
import { api, type Attachment, type Conversation, type ConversationDetail, type Message, type RunEvent } from '../api'

function revokeLocalPreviews(messages: Message[] | undefined) {
  for (const message of messages || []) {
    for (const attachment of message.attachments) {
      if (attachment.artifact_id.startsWith('local:') && attachment.url.startsWith('blob:')) {
        URL.revokeObjectURL(attachment.url)
      }
    }
  }
}

export const useConversations = defineStore('conversations', {
  state: () => ({
    channel: 'web' as 'web' | 'forum', conversations: [] as Conversation[],
    selected: null as ConversationDetail | null, loading: false, running: false,
    error: '', liveEvents: [] as RunEvent[], search: '', hasMore: false, lastFailedMessage: '',
    lastFailedFiles: [] as File[],
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
    async select(id: string) {
      const detail = await api<ConversationDetail>(`/api/conversations/${id}`)
      revokeLocalPreviews(this.selected?.messages)
      this.selected = detail; this.liveEvents = []
    },
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
    async send(message: string, files: File[] = []) {
      if (!this.selected || this.selected.conversation.channel !== 'web') return
      if (!message.trim() && !files.length) return
      const conversationId = this.selected.conversation.id
      const previousFailed = this.selected.messages.findIndex(item =>
        item.id.startsWith('local:') && item.status === 'failed' && item.content === message,
      )
      if (previousFailed >= 0) {
        const [failed] = this.selected.messages.splice(previousFailed, 1)
        revokeLocalPreviews(failed ? [failed] : [])
      }
      const pendingUrls: string[] = []
      const pendingAttachments: Attachment[] = files.map((file, index) => {
        const url = URL.createObjectURL(file); pendingUrls.push(url)
        return {
          artifact_id: `local:${Date.now()}:${index}`, url, mime_type: file.type,
          filename: file.name, source_kind: 'user_upload',
        }
      })
      const pendingMessage: Message = {
        id: `local:${Date.now()}`,
        role: 'user',
        content: message,
        status: 'sending',
        attachments: pendingAttachments,
        created_at: new Date().toISOString(),
        epoch: this.selected.conversation.context_epoch || 0,
      }
      this.selected.messages.push(pendingMessage)
      this.running = true; this.error = ''; this.liveEvents = []
      try {
        let body: BodyInit
        let headers: HeadersInit | undefined
        if (files.length) {
          const form = new FormData(); form.set('message', message)
          for (const file of files) form.append('images', file, file.name)
          body = form
        } else {
          body = JSON.stringify({ message }); headers = { 'Content-Type': 'application/json' }
        }
        const response = await fetch(`/api/conversations/${conversationId}/messages/stream`, {
          method: 'POST', credentials: 'same-origin', headers, body,
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
              const { event_id, run_id, created_at, ...eventPayload } = payload
              this.liveEvents.push({
                id: Number(event_id || this.liveEvents.length + 1),
                run_id: String(run_id || ''),
                type: event,
                payload: eventPayload,
                created_at: String(created_at || new Date().toISOString()),
              })
              if (event === 'stream.error') throw new Error(payload.error || '请求失败')
            }
          }
        }
        this.lastFailedMessage = ''
        this.lastFailedFiles = []
        pendingUrls.forEach(url => URL.revokeObjectURL(url))
        if (this.selected?.conversation.id === conversationId) await this.select(conversationId)
        await this.load()
      } catch (error) {
        pendingMessage.status = 'failed'
        this.error = String(error)
        this.lastFailedMessage = message
        this.lastFailedFiles = files
      } finally { this.running = false }
    },
  },
})
