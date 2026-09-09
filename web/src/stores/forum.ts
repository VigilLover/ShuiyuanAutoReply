import { defineStore } from 'pinia'
import { api, type Conversation, type RunEvent } from '../api'
import { activeRun, applyForumEvent, mergeRun, type ForumEvent, type ForumRun } from '../forum'
import { useConversations } from './conversations'

interface Snapshot { cursor: number; runs: ForumRun[]; conversations: Conversation[] }
let source: EventSource | undefined
let reconnectTimer: ReturnType<typeof setTimeout> | undefined
let refreshTimer: ReturnType<typeof setTimeout> | undefined
let generation = 0
let refreshList = false
let refreshDetail = false
let refreshing = false

// Only these events change persisted history; everything else is patched live
// from the event payload, so the conversation detail is not refetched per step.
const REFRESH_LIST_EVENTS = new Set(['run.accepted'])
const REFRESH_DETAIL_EVENTS = new Set(['run.completed', 'run.failed', 'run.interrupted', 'run.needs_review'])

export const useForumMonitor = defineStore('forum', {
  state: () => ({ runs: {} as Record<string, ForumRun>, events: {} as Record<string, RunEvent[]>, connection: '' }),
  actions: {
    ingest(runs: ForumRun[]) { for (const run of runs) mergeRun(this.runs, run) },
    async start() {
      this.stop()
      const version = generation
      const connect = async () => {
        if (version !== generation) return
        this.connection = '正在连接实时监控…'
        try {
          const snapshot = await api<Snapshot>('/api/forum/monitor')
          if (version !== generation) return
          const activeIds = new Set(snapshot.runs.map(run => run.id))
          for (const run of Object.values(this.runs)) {
            if (activeRun(run) && !activeIds.has(run.id)) delete this.runs[run.id]
          }
          this.ingest(snapshot.runs)
          const store = useConversations()
          for (const item of snapshot.conversations) {
            const old = store.conversations.find(c => c.id === item.id)
            if (old) Object.assign(old, item)
          }
          source = new EventSource(`/api/forum/events/stream?after=${snapshot.cursor}`)
          source.onopen = () => { if (version === generation) this.connection = '' }
          source.addEventListener('forum.event', event => {
            if (version !== generation) return
            const value: ForumEvent = JSON.parse((event as MessageEvent).data)
            applyForumEvent(this.runs, this.events, value)
            if (REFRESH_LIST_EVENTS.has(value.type)) this.scheduleRefresh('list')
            else if (REFRESH_DETAIL_EVENTS.has(value.type)) this.scheduleRefresh()
          })
          source.onerror = () => {
            if (version !== generation) return
            source?.close()
            this.connection = '实时连接已断开，正在重连…'
            reconnectTimer = setTimeout(connect, 1500)
          }
          // Refresh history after opening the stream; live events survive older snapshots.
          this.scheduleRefresh('list')
        } catch {
          if (version !== generation) return
          this.connection = '实时连接已断开，正在重连…'
          reconnectTimer = setTimeout(connect, 1500)
        }
      }
      await connect()
    },
    scheduleRefresh(kind: 'list' | 'detail' = 'detail') {
      if (kind === 'list') refreshList = true
      refreshDetail = true
      if (refreshTimer || refreshing) return
      const version = generation
      refreshTimer = setTimeout(async () => {
        refreshTimer = undefined
        if (version !== generation) return
        const wantsList = refreshList
        refreshing = true
        refreshList = false
        refreshDetail = false
        const store = useConversations()
        try {
          if (wantsList) await store.refresh()
          const id = store.selected?.conversation.id
          if (id && store.channel === 'forum') {
            await store.select(id, true)
            if (version === generation) this.ingest(store.selected?.runs || [])
          }
          for (const run of Object.values(this.runs)) {
            if (!activeRun(run) && run.conversation_id !== store.selected?.conversation.id) {
              delete this.events[run.id]
              delete this.runs[run.id]
            }
          }
        } catch {
          if (version === generation) {
            if (wantsList) refreshList = true
            refreshDetail = true
            refreshTimer = setTimeout(() => {
              refreshTimer = undefined
              this.scheduleRefresh(wantsList ? 'list' : 'detail')
            }, 1500)
          }
        }
        finally {
          refreshing = false
          if ((refreshList || refreshDetail) && useConversations().channel === 'forum') {
            this.scheduleRefresh(refreshList ? 'list' : 'detail')
          }
        }
      }, 100)
    },
    stop() {
      generation++
      source?.close(); source = undefined
      clearTimeout(reconnectTimer); clearTimeout(refreshTimer)
      refreshTimer = undefined; refreshList = false; refreshDetail = false
      this.connection = ''
    },
  },
})
