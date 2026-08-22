<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import {
  PhArrowUp,
  PhBroom,
  PhCaretDown,
  PhGearSix,
  PhImages,
  PhMagnifyingGlass,
  PhPencilSimple,
  PhPlus,
  PhPlusCircle,
  PhRobot,
  PhShieldCheck,
  PhSidebarSimple,
  PhTrash,
  PhUserCircle,
  PhX,
} from '@phosphor-icons/vue'
import MarkdownContent from '../components/MarkdownContent.vue'
import PromptEvent from '../components/PromptEvent.vue'
import RunProgress from '../components/RunProgress.vue'
import { useConversations } from '../stores/conversations'

const store = useConversations()
const input = ref('')
const editing = ref(false)
const title = ref('')
const activeTab = ref<'chat' | 'trace'>('chat')
const sessionMenuOpen = ref(false)
const sidebarCollapsed = ref(false)
const messagesElement = ref<HTMLElement | null>(null)
const composerTextarea = ref<HTMLTextAreaElement | null>(null)
const composerDock = ref<HTMLElement | null>(null)
const composerSpace = ref(190)
const fileInput = ref<HTMLInputElement | null>(null)
const selectedImages = ref<Array<{ file: File; url: string }>>([])
const uploadError = ref('')
const dragActive = ref(false)
const dragDepth = ref(0)
const lightboxUrl = ref('')
let searchTimer: number | undefined
let composerResizeObserver: ResizeObserver | undefined

const channelLabel = computed(() => store.channel === 'web' ? '网页对话' : '论坛记录')
const selectedEvents = computed(() => store.selected?.events || [])
const shellStyle = computed(() => ({ '--composer-space': `${composerSpace.value}px` }))
const personaId = computed(() => store.selected?.conversation.persona_id || store.conversations[0]?.persona_id || 'persona')
const personaLabel = computed(() => personaId.value.toUpperCase())

onMounted(async () => {
  await store.load()
  if (!store.selected && store.conversations.length) await store.select(store.conversations[0].id)
  await nextTick()
  observeComposer()
  resizeComposer()
  window.addEventListener('resize', resizeComposer)
})

onBeforeUnmount(() => {
  window.clearTimeout(searchTimer)
  composerResizeObserver?.disconnect()
  window.removeEventListener('resize', resizeComposer)
  selectedImages.value.forEach(image => URL.revokeObjectURL(image.url))
})

watch(() => store.selected?.conversation.title, value => {
  title.value = value || ''
  activeTab.value = 'chat'
  sessionMenuOpen.value = false
  nextTick(scrollToBottom)
})
watch(() => store.selected?.messages.length, () => nextTick(scrollToBottom))
watch(() => store.selected?.conversation.id, () => nextTick(() => {
  observeComposer()
  resizeComposer()
  scrollToBottom()
}))
watch(() => store.selected?.conversation.channel, () => nextTick(() => {
  observeComposer()
  resizeComposer()
}))
watch(input, () => nextTick(resizeComposer))

async function switchChannel(channel: 'web' | 'forum') {
  store.channel = channel
  store.selected = null
  await store.load()
  if (store.conversations.length) await store.select(store.conversations[0].id)
}

async function createConversation() {
  if (store.channel !== 'web') {
    store.channel = 'web'
    store.selected = null
  }
  await store.create()
  activeTab.value = 'chat'
}

async function selectConversation(conversationId: string) {
  activeTab.value = 'chat'
  await store.select(conversationId)
  await nextTick()
  scrollToBottom()
}

async function send() {
  const value = input.value.trim()
  if ((!value && !selectedImages.value.length) || store.running) return
  const images = selectedImages.value
  const files = images.map(image => image.file)
  input.value = ''
  selectedImages.value = []
  uploadError.value = ''
  try {
    await store.send(value, files)
  } finally {
    images.forEach(image => URL.revokeObjectURL(image.url))
  }
}

const acceptedImageTypes = new Set(['image/jpeg', 'image/png', 'image/gif', 'image/webp'])
const maxImageBytes = 20 * 1024 * 1024

function addFiles(files: File[]) {
  uploadError.value = ''
  const remaining = Math.max(0, 20 - selectedImages.value.length)
  if (files.length > remaining) uploadError.value = '每条消息最多添加 20 张图片'
  for (const file of files.slice(0, remaining)) {
    if (!acceptedImageTypes.has(file.type)) {
      uploadError.value = `${file.name} 不是支持的图片格式`
      continue
    }
    if (file.size > maxImageBytes) {
      uploadError.value = `${file.name} 超过 20MB`
      continue
    }
    const duplicate = selectedImages.value.some(item =>
      item.file.name === file.name && item.file.size === file.size && item.file.lastModified === file.lastModified,
    )
    if (!duplicate) selectedImages.value.push({ file, url: URL.createObjectURL(file) })
  }
  nextTick(() => { observeComposer(); resizeComposer() })
}

function chooseFiles(event: Event) {
  const target = event.target as HTMLInputElement
  addFiles(Array.from(target.files || [])); target.value = ''
}

function removeImage(index: number) {
  const [removed] = selectedImages.value.splice(index, 1)
  if (removed) URL.revokeObjectURL(removed.url)
}

function hasDraggedFiles(event: DragEvent) {
  return Array.from(event.dataTransfer?.types || []).includes('Files')
}

function dragEnter(event: DragEvent) {
  if (store.selected?.conversation.channel !== 'web' || !hasDraggedFiles(event)) return
  dragDepth.value += 1; dragActive.value = true
}

function dragLeave() {
  dragDepth.value = Math.max(0, dragDepth.value - 1)
  if (!dragDepth.value) dragActive.value = false
}

function dropFiles(event: DragEvent) {
  dragDepth.value = 0; dragActive.value = false
  if (store.selected?.conversation.channel !== 'web') return
  addFiles(Array.from(event.dataTransfer?.files || []))
}

function pasteImages(event: ClipboardEvent) {
  const files = Array.from(event.clipboardData?.files || []).filter(file => file.type.startsWith('image/'))
  if (files.length) { event.preventDefault(); addFiles(files) }
}

async function rename() {
  if (!title.value.trim()) return
  await store.rename(title.value.trim())
  editing.value = false
}

async function removeConversation() {
  if (window.confirm('永久删除本地对话记录和关联图片？此操作不可撤销。')) await store.remove()
}

function searchConversations() {
  window.clearTimeout(searchTimer)
  searchTimer = window.setTimeout(() => store.load(), 250)
}

function eventsForMessage(runId?: string) {
  if (!runId) return []
  return (store.selected?.events || []).filter(event => event.run_id === runId)
}

function eventSummary(payload: Record<string, unknown>) {
  const text = JSON.stringify(payload)
  return text.length > 260 ? `${text.slice(0, 260)}…` : text
}

function observeComposer() {
  composerResizeObserver?.disconnect()
  const dock = composerDock.value
  if (!dock) return

  const updateSpace = () => {
    const nextSpace = Math.ceil(dock.getBoundingClientRect().height) + 14
    const messages = messagesElement.value
    const pinnedToBottom = !messages || messages.scrollHeight - messages.scrollTop - messages.clientHeight < 72
    composerSpace.value = nextSpace
    if (pinnedToBottom) nextTick(scrollToBottom)
  }

  composerResizeObserver = new ResizeObserver(updateSpace)
  composerResizeObserver.observe(dock)
  updateSpace()
}

function resizeComposer() {
  const textarea = composerTextarea.value
  if (!textarea) return

  const messages = messagesElement.value
  const pinnedToBottom = !messages || messages.scrollHeight - messages.scrollTop - messages.clientHeight < 72
  const maxHeight = Math.min(180, Math.max(120, Math.floor(window.innerHeight * 0.3)))
  textarea.style.height = 'auto'
  textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`
  textarea.style.overflowY = textarea.scrollHeight > maxHeight ? 'auto' : 'hidden'
  if (pinnedToBottom) requestAnimationFrame(scrollToBottom)
}

function scrollToBottom() {
  if (messagesElement.value) messagesElement.value.scrollTop = messagesElement.value.scrollHeight
}
</script>

<template>
  <main
    class="harness-shell" :class="{ 'sidebar-collapsed': sidebarCollapsed }" :style="shellStyle"
    @dragenter.prevent="dragEnter" @dragover.prevent @dragleave.prevent="dragLeave" @drop.prevent="dropFiles"
  >
    <aside class="harness-sidebar">
      <div class="harness-brand">
        <div class="brand-identity" aria-label="Shuiyuan AutoReply">
          <img class="brand-lockup" src="/assets/brand-lockup.svg" alt="Shuiyuan AutoReply" />
          <span class="persona-badge" :title="`当前 persona：${personaId}`">{{ personaLabel }}</span>
        </div>
        <button class="icon-button sidebar-toggle" :aria-label="sidebarCollapsed ? '展开侧栏' : '收起侧栏'" @click="sidebarCollapsed = !sidebarCollapsed">
          <PhSidebarSimple :size="20" />
        </button>
      </div>

      <button class="new-session" aria-label="新对话" @click="createConversation">
        <PhPlusCircle :size="21" /> <span>新对话</span>
      </button>

      <div class="channel-switch">
        <button :class="{ active: store.channel === 'web' }" @click="switchChannel('web')">网页对话</button>
        <button :class="{ active: store.channel === 'forum' }" @click="switchChannel('forum')">论坛记录</button>
      </div>
      <label class="sidebar-search">
        <PhMagnifyingGlass :size="16" />
        <input v-model="store.search" placeholder="搜索会话" @input="searchConversations" />
      </label>

      <div class="session-list">
        <button
          v-for="item in store.conversations"
          :key="item.id"
          class="session-item"
          :class="{ selected: store.selected?.conversation.id === item.id }"
          :aria-label="item.title"
          :title="item.title"
          @click="selectConversation(item.id)"
        >
          <span>{{ item.title }}</span>
          <small>{{ new Date(item.updated_at).toLocaleDateString() }}</small>
        </button>
        <p v-if="!store.loading && !store.conversations.length" class="empty-small">暂无{{ channelLabel }}</p>
        <button v-if="store.hasMore" class="load-more" @click="store.load(true)">加载更多</button>
      </div>

      <RouterLink class="settings-link" to="/settings" aria-label="设置"><PhGearSix :size="20" /><span>设置</span></RouterLink>
    </aside>

    <section v-if="store.selected" class="harness-workspace">
      <header class="workspace-topbar">
        <div class="workspace-title">
          <div class="title-line">
            <input
              v-if="editing"
              v-model="title"
              class="title-editor"
              autofocus
              @keyup.enter="rename"
              @keyup.esc="editing = false"
              @blur="rename"
            />
            <h1 v-else @dblclick="store.selected.conversation.channel === 'web' && (editing = true)">
              {{ store.selected.conversation.title }}
            </h1>
            <span class="runtime-pill">{{ store.selected.conversation.channel === 'web' ? 'WEB' : 'FORUM · READ ONLY' }}</span>
          </div>
          <div class="view-tabs">
            <button :class="{ active: activeTab === 'chat' }" @click="activeTab = 'chat'">对话</button>
            <button :class="{ active: activeTab === 'trace' }" @click="activeTab = 'trace'">轨迹 <span>{{ selectedEvents.length }}</span></button>
          </div>
        </div>
        <div class="workspace-actions">
          <button class="session-log-button" @click="sessionMenuOpen = !sessionMenuOpen">会话管理 <PhCaretDown :size="14" /></button>
          <div v-if="sessionMenuOpen" class="session-menu">
            <button v-if="store.selected.conversation.channel === 'web'" @click="editing = true; sessionMenuOpen = false"><PhPencilSimple :size="16" />重命名</button>
            <button @click="store.clear(); sessionMenuOpen = false"><PhBroom :size="16" />清除模型上下文</button>
            <button class="danger" @click="sessionMenuOpen = false; removeConversation()"><PhTrash :size="16" />永久删除本地记录</button>
          </div>
        </div>
      </header>

      <div v-if="activeTab === 'chat'" ref="messagesElement" class="harness-messages">
        <div class="message-stream">
          <template v-for="message in store.selected.messages" :key="message.id">
            <div v-if="message.role === 'system'" class="system-divider" :class="{ failed: message.status === 'failed' }">
              <span></span><p>{{ message.content }}</p><span></span>
            </div>
            <article v-else class="message-entry" :class="message.role">
              <div class="message-meta">
                <span class="message-avatar"><PhUserCircle v-if="message.role === 'user'" :size="19" /><PhRobot v-else :size="19" /></span>
                <div>
                  <strong>{{ message.role === 'user' ? '你' : 'Shuiyuan Bot' }}</strong>
                  <small>{{ new Date(message.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}</small>
                </div>
              </div>
              <div class="message-content" :class="{ 'user-surface': message.role === 'user' }">
                <MarkdownContent
                  :content="message.content"
                  :attachments="message.attachments"
                  @preview="lightboxUrl = $event"
                />
                <RunProgress
                  v-if="message.role === 'assistant' && eventsForMessage(message.run_id).length"
                  :events="eventsForMessage(message.run_id)"
                />
              </div>
            </article>
          </template>

          <RunProgress v-if="store.running" :events="store.liveEvents" running />
          <div v-if="store.error" class="request-error">
            <div><strong>请求失败</strong><p>{{ store.error }}</p></div>
            <button v-if="store.lastFailedMessage || store.lastFailedFiles.length" @click="store.send(store.lastFailedMessage, store.lastFailedFiles)">重试</button>
          </div>
        </div>
      </div>

      <div v-else class="trace-view">
        <div class="trace-toolbar">
          <span>执行事件</span><small>{{ selectedEvents.length }} records</small>
        </div>
        <div v-if="selectedEvents.length" class="trace-table">
          <div v-for="event in selectedEvents" :key="event.id" class="trace-table-row">
            <time>{{ new Date(event.created_at).toLocaleTimeString() }}</time>
            <span class="event-kind">{{ event.type }}</span>
            <PromptEvent v-if="event.type === 'model.prompt_prepared'" :payload="event.payload" />
            <code v-else>{{ eventSummary(event.payload) }}</code>
          </div>
        </div>
        <div v-else class="trace-empty">当前会话还没有执行轨迹。</div>
      </div>

      <div ref="composerDock" class="composer-dock">
        <form v-if="store.selected.conversation.channel === 'web'" class="harness-composer" @submit.prevent="send">
          <div v-if="selectedImages.length" class="composer-attachments">
            <div v-for="(image, index) in selectedImages" :key="image.url" class="composer-thumbnail" :title="image.file.name">
              <img :src="image.url" :alt="image.file.name" />
              <button type="button" aria-label="移除图片" @click="removeImage(index)"><PhX :size="12" weight="bold" /></button>
            </div>
          </div>
          <p v-if="uploadError" class="upload-error">{{ uploadError }}</p>
          <textarea ref="composerTextarea" v-model="input" rows="1" placeholder="给智能体发送消息" @paste="pasteImages" @input="resizeComposer" @keydown.enter.exact.prevent="send"></textarea>
          <div class="composer-bottom">
            <input ref="fileInput" class="visually-hidden" type="file" accept="image/jpeg,image/png,image/gif,image/webp" multiple @change="chooseFiles" />
            <button class="composer-add" type="button" aria-label="添加图片" :disabled="store.running || selectedImages.length >= 20" @click="fileInput?.click()"><PhPlus :size="18" weight="bold" /></button>
            <span class="permission-chip"><PhShieldCheck :size="16" /> 小狼bot运行中ψ(｀∇´)ψ</span>
            <span class="model-chip">DeepSeek V4 Flash Vision · {{ store.running ? '运行中' : '就绪' }}</span>
            <button class="composer-send" :disabled="store.running || (!input.trim() && !selectedImages.length)" aria-label="发送消息"><PhArrowUp :size="19" weight="bold" /></button>
          </div>
        </form>
        <div v-else class="readonly-dock">论坛自动回复记录为只读，无法从网页创建回复。</div>
      </div>
    </section>

    <section v-else class="harness-workspace empty-workspace">
      <div class="empty-content">
        <PhRobot class="empty-logo" :size="32" />
        <h1>Shuiyuan Auto Reply</h1>
        <p>选择一段会话，或新建网页对话。</p>
        <button @click="createConversation"><PhPlusCircle :size="18" /> 新建对话</button>
      </div>
    </section>

    <div v-if="dragActive" class="image-drop-overlay" aria-hidden="true">
      <div class="image-drop-content">
        <span class="image-drop-icon"><PhImages :size="54" weight="duotone" /></span>
        <strong>图片拖动到此处即可添加</strong>
        <small>最多 20 张，每张 20MB</small>
      </div>
    </div>

    <div v-if="lightboxUrl" class="image-lightbox" role="dialog" aria-modal="true" @click.self="lightboxUrl = ''">
      <button type="button" aria-label="关闭预览" @click="lightboxUrl = ''"><PhX :size="22" /></button>
      <img :src="lightboxUrl" alt="图片大图预览" />
    </div>
  </main>
</template>
