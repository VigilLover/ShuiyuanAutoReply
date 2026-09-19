<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import {
  PhArrowsClockwise,
  PhChatCircleText,
  PhCheck,
  PhCopy,
  PhCpu,
  PhFloppyDisk,
  PhGlobe,
  PhPlugsConnected,
  PhPlus,
  PhRocketLaunch,
  PhShieldCheck,
  PhTextT,
  PhWarningCircle,
  PhWarningDiamond,
  PhX,
} from '@phosphor-icons/vue'
import { api } from '../api'
import {
  activeEntry,
  configPayload,
  draftError,
  draftFromEntry,
  endpointHost,
  keyLabel,
  KIND_LABELS,
  newDraft,
  sortModels,
  usedBy,
  type ModelConfigDraft,
  type ModelConfigEntry,
} from '../model-configs'
import {
  draftCard,
  effectiveCard,
  hasArchivedPrompt,
  lastUsedCard,
  promptModeState,
  shortHash,
  verificationNotice,
} from '../prompt'

const profiles = ref<any[]>([])
const promptPreview = ref('')
const promptPreviewMeta = ref<any>({})
const promptView = ref<'edit' | 'preview' | 'archive'>('edit')
const promptBusy = ref(false)
const draftSnapshots = ref<Record<string, string>>({})
const scope = ref<'web' | 'forum'>('web')
const status = ref('')
const statusError = ref(false)
const tools = ref<any[]>([])
const mcp = ref<any>({ url: null, configured: false, connected: false, error: null, tools: [] })
const mcpLoading = ref(false)
const activeSection = ref<'model' | 'prompt' | 'tools'>('model')
const modelConfigs = ref<any>({ chat: [], image: [], active: {} })
const runtimeInfo = ref<any>(null)
const configEditor = ref<ModelConfigDraft | null>(null)
const configProbe = ref<{ ok: boolean; models: string[]; message: string } | null>(null)
const configBusy = ref(false)
const configError = ref('')
const modelKinds = ['chat', 'image'] as const

const current = () => profiles.value.find(item => item.scope === scope.value)
const chatActive = computed(() => activeEntry('chat', scope.value, modelConfigs.value))
const imageActive = computed(() => activeEntry('image', 'image', modelConfigs.value))
const probeModels = computed(() =>
  sortModels(configProbe.value?.models || [], configEditor.value?.model),
)
const defaultKeyVisible = computed(() => modelConfigs.value.active?.[scope.value] === 'default')
const mode = computed(() => promptModeState(current()))
const draftDirty = computed(() => {
  const item = current()
  if (!item) return false
  return JSON.stringify(item.draft) !== (draftSnapshots.value[item.scope] ?? '')
})
const promptCards = computed(() => [
  effectiveCard(current()),
  draftCard(current(), draftDirty.value),
  lastUsedCard(current()),
])
const promptNotice = computed(() => verificationNotice(current()))
const previewHint = computed(() => {
  const meta = promptPreviewMeta.value || {}
  return [
    meta.rules_version ? `规则 v${meta.rules_version}` : '',
    meta.prompt_hash ? `指纹 ${shortHash(meta.prompt_hash)}` : '',
    draftDirty.value ? '含未保存编辑' : '',
  ].filter(Boolean).join(' · ')
})
const promptTabs = computed(() => {
  const tabs: { id: 'edit' | 'preview' | 'archive'; label: string }[] = [
    { id: 'edit', label: '编辑' },
    { id: 'preview', label: '预览' },
  ]
  if (mode.value.managed && hasArchivedPrompt(current())) {
    tabs.push({ id: 'archive', label: '归档' })
  }
  return tabs
})

function snapshotDrafts() {
  draftSnapshots.value = Object.fromEntries(
    profiles.value.map(item => [item.scope, JSON.stringify(item.draft)]),
  )
}

function countText(value: unknown) {
  return `${String(value ?? '').length} 字`
}

async function load() {
  profiles.value = await api('/api/settings/profiles')
  snapshotDrafts()
  api<any>('/api/bootstrap').then(value => { runtimeInfo.value = value }).catch(() => { runtimeInfo.value = null })
  await loadModelConfigs()
  await loadScopeSettings()
}

async function loadScopeSettings() {
  status.value = ''
  await Promise.all([loadTools(), loadMcp()])
}

async function changeScope(value: 'web' | 'forum') {
  scope.value = value
  promptPreview.value = ''
  promptView.value = 'edit'
  configEditor.value = null
  configProbe.value = null
  configError.value = ''
  await loadScopeSettings()
}

async function loadTools() {
  tools.value = await api(`/api/settings/tools/${scope.value}`)
}

async function loadMcp() {
  mcpLoading.value = true
  try {
    mcp.value = await api(`/api/settings/mcp/${scope.value}`)
  } catch (error) {
    mcp.value = { url: null, configured: false, connected: false, error: String(error), tools: [] }
  } finally {
    mcpLoading.value = false
  }
}

async function loadModelConfigs() {
  modelConfigs.value = await api('/api/settings/model-configs')
}

function startCreate(kind: 'chat' | 'image') {
  configEditor.value = newDraft(kind)
  configProbe.value = null
  configError.value = ''
}

function startEdit(entry: ModelConfigEntry) {
  configEditor.value = draftFromEntry(entry)
  configProbe.value = null
  configError.value = ''
}

function cancelEdit() {
  configEditor.value = null
  configProbe.value = null
  configError.value = ''
}

async function saveConfig() {
  const draft = configEditor.value
  if (!draft) return
  const problem = draftError(draft)
  if (problem) {
    configError.value = problem
    return
  }
  configBusy.value = true
  configError.value = ''
  try {
    const body = JSON.stringify(configPayload(draft))
    if (draft.id) {
      await api(`/api/settings/model-configs/${draft.id}`, { method: 'PUT', body })
    } else {
      await api('/api/settings/model-configs', { method: 'POST', body })
    }
    await loadModelConfigs()
    setStatus('配置已保存')
    cancelEdit()
  } catch (error) {
    configError.value = String(error)
  } finally {
    configBusy.value = false
  }
}

async function probeConfig() {
  const draft = configEditor.value
  if (!draft) return
  configBusy.value = true
  configError.value = ''
  configProbe.value = null
  try {
    const payload: Record<string, string> = {
      kind: draft.kind,
      base_url: draft.base_url,
      model: draft.model,
    }
    if (draft.api_key.trim()) payload.api_key = draft.api_key.trim()
    if (draft.id) payload.config_id = draft.id
    configProbe.value = await api('/api/settings/model-configs/probe', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  } catch (error) {
    configError.value = String(error)
  } finally {
    configBusy.value = false
  }
}

async function deleteConfig(entry: ModelConfigEntry) {
  if (!window.confirm(`删除配置《${entry.name}》？已启用的应用会回到默认配置。`)) return
  try {
    await api(`/api/settings/model-configs/${entry.id}`, { method: 'DELETE' })
    if (configEditor.value?.id === entry.id) cancelEdit()
    await load()
    setStatus('配置已删除')
  } catch (error) {
    setStatus(String(error), true)
  }
}

async function activateConfig(entry: ModelConfigEntry, target: 'web' | 'forum' | 'image') {
  configBusy.value = true
  try {
    const result: any = await api(
      `/api/settings/model-configs/${entry.id}/activate`,
      { method: 'POST', body: JSON.stringify({ scope: target }) },
    )
    await load()
    const where = target === 'image' ? '生图模型' : target === 'web' ? '网页对话' : '论坛自动回复'
    setStatus(
      target === 'image'
        ? `${where}已切换到《${entry.name}》，下次生图生效`
        : `${where}已切换到《${entry.name}》 · revision ${result.active_revision}`,
    )
  } catch (error) {
    setStatus(String(error), true)
  } finally {
    configBusy.value = false
  }
}

async function save(showStatus = true) {
  const item = current()
  item.draft.enabled_tools = tools.value.filter(tool => tool.enabled).map(tool => tool.name)
  if (mcp.value.connected) {
    item.draft.disabled_mcp_tools = mcp.value.tools.filter((tool: any) => !tool.enabled).map((tool: any) => tool.name)
  }
  await api(`/api/settings/profiles/${scope.value}/draft`, { method: 'PUT', body: JSON.stringify(item.draft) })
  snapshotDrafts()
  if (promptView.value === 'preview') await previewPrompt()
  if (showStatus) setStatus('草稿已保存')
}

async function apply() {
  try {
    await save(false)
    const result: any = await api(`/api/settings/profiles/${scope.value}/apply`, { method: 'POST' })
    setStatus(`已应用 revision ${result.active_revision}`)
    await load()
  } catch (error) {
    setStatus(String(error), true)
  }
}

async function testProvider() {
  try {
    await save(false)
    const result: any = await api(`/api/settings/profiles/${scope.value}/provider-test`, { method: 'POST' })
    setStatus(result.message, !result.ok)
  } catch (error) {
    setStatus(String(error), true)
  }
}

async function previewPrompt() {
  promptBusy.value = true
  try {
    const result: any = await api(`/api/settings/profiles/${scope.value}/prompt-preview`, { method: 'POST', body: JSON.stringify(current().draft) })
    promptPreview.value = result.template
    promptPreviewMeta.value = result
  } catch (error) {
    promptPreview.value = ''
    setStatus(String(error), true)
  } finally {
    promptBusy.value = false
  }
}

async function migratePrompt() {
  await save(false)
  await api(`/api/settings/profiles/${scope.value}/prompt-migrate`, { method: 'POST' })
  await load()
  promptView.value = 'edit'
  setStatus('已切换到托管规则草稿；旧完整提示词已归档，请审阅后应用')
}

async function copyArchived() {
  try {
    await navigator.clipboard.writeText(current().draft.system_prompt ?? '')
    setStatus('已复制归档提示词')
  } catch {
    setStatus('复制失败，请手动选择文本', true)
  }
}

function restorePersona() {
  current().draft.persona_text = null
  setStatus('已恢复内置人设，保存草稿后生效')
}

async function restoreDefault() {
  if (!window.confirm('将重置当前应用的模型、API 格式、提示词及工具开关草稿。已保存的密钥不变。继续？')) return
  await api(`/api/settings/profiles/${scope.value}/restore-default`, { method: 'POST' })
  setStatus('已恢复默认草稿，应用后生效')
  await load()
}

function setStatus(message: string, error = false) {
  status.value = message
  statusError.value = error
}

watch(promptView, view => {
  if (view === 'preview') previewPrompt()
})

onMounted(load)
</script>

<template>
  <main class="settings-shell">
    <div class="settings-window">
      <header class="settings-header">
        <div><h1>设置</h1><p>管理不同应用的模型、提示词与工具</p></div>
        <div class="settings-header-actions">
          <span v-if="status" class="settings-status" :class="{ failed: statusError }">{{ status }}</span>
          <RouterLink to="/" class="close-settings" aria-label="关闭设置"><PhX :size="21" /></RouterLink>
        </div>
      </header>

      <div class="settings-body">
        <aside class="settings-nav">
          <p>应用</p>
          <button :class="{ 'scope-selected': scope === 'web' }" @click="changeScope('web')"><PhChatCircleText :size="20" /><div>网页对话<small>独立会话 Runtime</small></div></button>
          <button :class="{ 'scope-selected': scope === 'forum' }" @click="changeScope('forum')"><PhGlobe :size="20" /><div>论坛自动回复<small>论坛 Worker Runtime</small></div></button>
          <p>配置</p>
          <button :class="{ active: activeSection === 'model' }" @click="activeSection = 'model'"><PhCpu :size="20" /><div>模型<small>DeepSeek 端点与密钥</small></div></button>
          <button :class="{ active: activeSection === 'prompt' }" @click="activeSection = 'prompt'"><PhTextT :size="20" /><div>提示词<small>System Prompt</small></div></button>
          <button :class="{ active: activeSection === 'tools' }" @click="activeSection = 'tools'"><PhPlugsConnected :size="20" /><div>工具与 MCP<small>能力开关</small></div></button>
        </aside>

        <section v-if="current()" class="settings-content">
          <div class="settings-content-head">
            <div><span class="eyebrow">{{ scope === 'web' ? 'WEB RUNTIME' : 'FORUM RUNTIME' }}</span><h2>{{ activeSection === 'model' ? '模型配置' : activeSection === 'prompt' ? 'System Prompt' : '工具与 MCP' }}</h2></div>
            <span class="revision-badge">ACTIVE · r{{ current().active_revision }}</span>
          </div>

          <div v-if="activeSection === 'model'" class="settings-section model-section">
            <p class="section-intro">文字模型按应用启用（网页 / 论坛各自选用），生图模型全机器人共用。默认配置来自部署文件、只读；新增的配置存在本机，可随时一键切换。</p>

            <div class="config-active-grid">
              <div class="config-active-card">
                <span class="eyebrow">{{ scope === 'web' ? '本应用启用 · 网页对话' : '本应用启用 · 论坛自动回复' }}</span>
                <strong class="config-active-name">{{ chatActive?.name || '—' }}</strong>
                <small>{{ chatActive?.model }} · {{ endpointHost(chatActive?.base_url) }}</small>
                <small class="config-key" :class="{ ok: chatActive?.secret?.configured }">{{ keyLabel(chatActive?.secret, { fallback: chatActive?.source === 'custom' }) }}</small>
              </div>
              <div class="config-active-card">
                <span class="eyebrow">全机器人启用 · 生图</span>
                <strong>{{ imageActive?.name }}</strong>
                <small>{{ imageActive?.model }} · {{ endpointHost(imageActive?.base_url) }}</small>
                <small class="config-key" :class="{ ok: imageActive?.secret?.configured }">{{ keyLabel(imageActive?.secret, { fallback: imageActive?.source === 'custom' }) }}</small>
              </div>
            </div>

            <div v-for="kind in modelKinds" :key="kind" class="config-group">
              <div class="config-group-head">
                <div>
                  <h3>{{ KIND_LABELS[kind] }}</h3>
                  <p>{{ kind === 'chat' ? '每应用各自选用，切换走热切换通道' : '全机器人共用，切换后下一次生图即生效' }}</p>
                </div>
                <button class="outline-action compact" :disabled="configBusy" @click="startCreate(kind)"><PhPlus :size="14" />新建配置</button>
              </div>
              <div class="config-list">
                <div
                  v-for="entry in modelConfigs[kind]"
                  :key="entry.id"
                  class="config-row"
                  :class="{ selected: usedBy(entry, modelConfigs).length }"
                >
                  <div class="config-row-main">
                    <strong>{{ entry.name }}</strong>
                    <small>{{ entry.model }} · {{ endpointHost(entry.base_url) }}</small>
                  </div>
                  <span v-if="entry.source === 'default'" class="config-badge">部署配置</span>
                  <span v-else-if="usedBy(entry, modelConfigs).length" class="config-badge used">{{ usedBy(entry, modelConfigs).join(' / ') }}</span>
                  <span class="config-key" :class="{ ok: entry.secret?.configured }">{{ keyLabel(entry.secret) }}</span>
                  <div class="config-row-actions">
                    <button
                      v-if="kind === 'chat'"
                      class="text-action"
                      :class="{ active: modelConfigs.active[scope] === entry.id }"
                      :disabled="configBusy || modelConfigs.active[scope] === entry.id"
                      @click="activateConfig(entry, scope)"
                    >{{ modelConfigs.active[scope] === entry.id ? '本应用使用中' : '本应用启用' }}</button>
                    <button
                      v-else
                      class="text-action"
                      :class="{ active: modelConfigs.active.image === entry.id }"
                      :disabled="configBusy || modelConfigs.active.image === entry.id"
                      @click="activateConfig(entry, 'image')"
                    >{{ modelConfigs.active.image === entry.id ? '使用中' : '启用' }}</button>
                    <button v-if="entry.source !== 'default'" class="text-action" :disabled="configBusy" @click="startEdit(entry)">编辑</button>
                    <button v-if="entry.source !== 'default'" class="text-action danger" :disabled="configBusy" @click="deleteConfig(entry)">删除</button>
                  </div>
                </div>
              </div>
            </div>

            <div v-if="runtimeInfo?.runtime" class="runtime-limits">
              <div class="tool-group-title"><div><h3>运行参数</h3><p>来自部署配置，只读；修改 deployment.toml / .env 后重启生效</p></div></div>
              <dl class="runtime-limits-grid">
                <div><dt>推理强度</dt><dd>调查 {{ runtimeInfo.reasoning?.investigate }} · 收尾 {{ runtimeInfo.reasoning?.final }}</dd></div>
                <div><dt>单次模型请求</dt><dd>{{ runtimeInfo.runtime.model_call_timeout }}s（收尾保留 {{ runtimeInfo.runtime.final_reserve_seconds }}s）</dd></div>
                <div><dt>整轮预算</dt><dd>{{ runtimeInfo.runtime.timeout }}s · {{ runtimeInfo.runtime.model_limit }} 轮 · {{ runtimeInfo.runtime.query_limit }} 次查询</dd></div>
                <div><dt>并发</dt><dd>回复 {{ runtimeInfo.runtime.concurrency }} · 生图 {{ runtimeInfo.runtime.image_concurrency }}</dd></div>
                <div><dt>上下文预算</dt><dd>{{ runtimeInfo.runtime.context_token_budget }} tokens · 连续 {{ runtimeInfo.runtime.no_progress_batches }} 批无新证据即收尾</dd></div>
              </dl>
            </div>
            <label v-if="defaultKeyVisible" class="full-field config-default-key">
              <span>默认端点密钥（仅默认文字模型使用；配置自带的密钥优先）</span>
              <input v-model="current().draft.api_key" type="password" :placeholder="current().secret?.configured ? `已配置 ····${current().secret.last_four}` : '输入新密钥'" />
            </label>

            <div v-if="configEditor" class="config-editor">
              <div class="config-editor-head">
                <div>
                  <span class="eyebrow">{{ configEditor.id ? '编辑配置' : '新建配置' }}</span>
                  <strong>{{ KIND_LABELS[configEditor.kind] }}</strong>
                </div>
                <button class="text-action" @click="cancelEdit"><PhX :size="15" />收起</button>
              </div>

              <div class="form-grid">
                <label class="full-field"><span>名称</span><input v-model="configEditor.name" placeholder="例如：聚合站 A" /></label>
                <label class="full-field"><span>Base URL</span><input v-model="configEditor.base_url" placeholder="https://provider.example/v1" spellcheck="false" /></label>
                <label class="full-field">
                  <span>API Key</span>
                  <input
                    v-model="configEditor.api_key"
                    type="password"
                    :placeholder="configEditor.id ? '留空表示不修改已保存的密钥' : 'sk-...（留空则用环境密钥）'"
                  />
                </label>
                <div class="full-field model-picker-field">
                  <span>模型</span>
                  <div class="model-picker">
                    <input v-model="configEditor.model" placeholder="选择或手动输入模型名" spellcheck="false" />
                    <button class="outline-action compact" :disabled="configBusy" @click="probeConfig">
                      <PhPlugsConnected :size="14" />{{ configBusy ? '检测中…' : '获取模型列表 / 测试连通' }}
                    </button>
                  </div>
                  <div v-if="probeModels.length" class="model-options">
                    <button
                      v-for="model in probeModels"
                      :key="model"
                      type="button"
                      class="model-chip"
                      :class="{ selected: configEditor.model === model }"
                      @click="configEditor.model = model"
                    >{{ model }}</button>
                  </div>
                  <small v-if="configProbe" class="config-probe" :class="{ ok: configProbe.ok }">{{ configProbe.message }}</small>
                </div>
                <div v-if="configEditor.kind === 'chat'" class="full-field api-format-field">
                  <span>API 格式</span>
                  <div class="api-format-grid" role="radiogroup" aria-label="API 格式">
                    <button
                      type="button"
                      class="provider-card api-format-card"
                      :class="{ selected: configEditor.api_format !== 'responses' }"
                      @click="configEditor.api_format = 'chat_completions'"
                    >
                      <strong>Chat Completions</strong>
                      <span v-if="configEditor.api_format !== 'responses'"><PhCheck :size="15" weight="bold" /> 已选择</span>
                    </button>
                    <button
                      type="button"
                      class="provider-card api-format-card"
                      :class="{ selected: configEditor.api_format === 'responses' }"
                      @click="configEditor.api_format = 'responses'"
                    >
                      <strong>Responses</strong>
                      <span v-if="configEditor.api_format === 'responses'"><PhCheck :size="15" weight="bold" /> 已选择</span>
                    </button>
                  </div>
                </div>
              </div>

              <p v-if="configEditor.kind === 'chat'" class="config-hint">
                <PhWarningCircle :size="14" weight="fill" />论坛 Agent 依赖视觉能力；自定义端点若不支持 DeepSeek 的 /files 上传链路，图片理解可能失败（只影响看图，不影响文字回答）。
              </p>
              <p v-if="configError" class="config-error">{{ configError }}</p>
              <div class="config-editor-actions">
                <button class="outline-action" :disabled="configBusy" @click="saveConfig"><PhFloppyDisk :size="16" />保存配置</button>
                <button class="text-action" @click="cancelEdit">取消</button>
              </div>
            </div>

            <button class="outline-action" :disabled="configBusy" @click="testProvider"><PhPlugsConnected :size="16" />测试当前启用的文字模型（真实请求一次）</button>
          </div>

          <div v-else-if="activeSection === 'prompt'" class="settings-section prompt-section">
            <p class="section-intro">执行规则随版本升级；人设与补充要求单独保存。修改后先保存草稿，再“应用并热切换”。</p>

            <div class="prompt-status">
              <div v-for="card in promptCards" :key="card.label" class="prompt-status-card" :class="card.tone">
                <span class="eyebrow">{{ card.label }}</span>
                <strong class="prompt-status-value">{{ card.value }}</strong>
                <small>{{ card.hint }}</small>
              </div>
            </div>

            <p v-if="promptNotice" class="prompt-notice">
              <PhWarningCircle :size="15" weight="fill" /><span>{{ promptNotice }}</span>
            </p>

            <div class="prompt-toolbar">
              <div class="prompt-mode" :class="mode.tone">
                <PhShieldCheck v-if="mode.managed" :size="18" weight="fill" />
                <PhWarningDiamond v-else :size="18" weight="fill" />
                <div><strong>{{ mode.label }}</strong><small>{{ mode.hint }}</small></div>
              </div>
              <div class="prompt-view-tabs" role="tablist" aria-label="提示词视图">
                <button
                  v-for="tab in promptTabs"
                  :key="tab.id"
                  type="button"
                  role="tab"
                  :aria-selected="promptView === tab.id"
                  :class="{ active: promptView === tab.id }"
                  @click="promptView = tab.id"
                >{{ tab.label }}</button>
              </div>
            </div>

            <div v-if="promptView === 'edit'" class="prompt-editor">
              <template v-if="mode.managed">
                <div class="prompt-field">
                  <div class="prompt-field-head">
                    <label for="prompt-persona">人设</label>
                    <span>留空文本即为空人设；恢复按钮使用内置人设</span>
                  </div>
                  <textarea id="prompt-persona" v-model="current().draft.persona_text" placeholder="使用内置默认人设" spellcheck="false"></textarea>
                  <div class="prompt-field-foot">
                    <small>{{ countText(current().draft.persona_text) }}</small>
                    <button class="text-action" @click="restorePersona"><PhArrowsClockwise :size="14" />恢复内置人设</button>
                  </div>
                </div>

                <div class="prompt-field compact">
                  <div class="prompt-field-head">
                    <label for="prompt-instructions">补充要求</label>
                    <span>追加在托管规则之后，用于本应用的额外约束</span>
                  </div>
                  <textarea id="prompt-instructions" v-model="current().draft.additional_instructions" placeholder="例如：回答尽量简短，不要使用颜文字" spellcheck="false"></textarea>
                  <div class="prompt-field-foot">
                    <small>{{ countText(current().draft.additional_instructions) }}</small>
                  </div>
                </div>
              </template>

              <template v-else>
                <div class="prompt-field mono tall">
                  <div class="prompt-field-head">
                    <label for="prompt-legacy">完整 System Prompt</label>
                    <span>旧文本会直接作为系统提示词使用，请确认仍然适用</span>
                  </div>
                  <textarea id="prompt-legacy" v-model="current().draft.system_prompt" spellcheck="false"></textarea>
                  <div class="prompt-field-foot">
                    <small>{{ countText(current().draft.system_prompt) }}</small>
                    <button class="outline-action compact" @click="migratePrompt"><PhArrowsClockwise :size="14" />迁移到托管规则</button>
                  </div>
                </div>
              </template>
            </div>

            <div v-else-if="promptView === 'preview'" class="prompt-view">
              <div class="prompt-panel-head">
                <div><strong>最终提示词模板</strong><small>{{ previewHint || '按当前草稿渲染' }}</small></div>
                <button class="outline-action compact" :disabled="promptBusy" @click="previewPrompt"><PhArrowsClockwise :size="14" />{{ promptBusy ? '生成中…' : '重新生成' }}</button>
              </div>
              <pre class="prompt-pre" :class="{ empty: !promptPreview }">{{ promptPreview || (promptBusy ? '正在生成预览…' : '点击“重新生成”查看最终模板。') }}</pre>
            </div>

            <div v-else class="prompt-view">
              <div class="prompt-panel-head">
                <div><strong>归档的旧完整提示词</strong><small>迁移前使用的原文，只读保留</small></div>
                <button class="outline-action compact" @click="copyArchived"><PhCopy :size="14" />复制</button>
              </div>
              <pre class="prompt-pre">{{ current().draft.system_prompt }}</pre>
            </div>
          </div>

          <div v-else class="settings-section tool-settings">
            <p v-if="current().suggested_tools?.length">新增能力尚未启用：{{ current().suggested_tools.join('、') }}</p>
            <p class="section-intro">内置工具使用启用列表；MCP 使用独立禁用列表，新发现的 MCP 工具默认启用。</p>
            <div class="tool-group">
              <div class="tool-group-title"><div><h3>内置工具</h3><p>论坛只读查询、图片生成与长期记忆</p></div><span>{{ tools.filter(tool => tool.enabled).length }}/{{ tools.length }} enabled</span></div>
              <div class="tool-card-grid">
                <label v-for="tool in tools" :key="tool.name" class="switch-card">
                  <div><strong>{{ tool.name }}</strong><small>{{ tool.loaded === false ? '加载失败' : tool.source }}</small></div>
                  <input v-model="tool.enabled" type="checkbox" /><span class="switch"></span>
                </label>
              </div>
            </div>

            <div class="tool-group mcp-group">
              <div class="tool-group-title">
                <div><h3>MCP Server</h3><p class="mcp-address">{{ mcp.url || '未配置 MCP_SERVER_URL' }}</p></div>
                <div class="connection-state" :class="{ connected: mcp.connected }"><span></span>{{ mcp.connected ? '已连接' : '未连接' }}</div>
              </div>
              <div class="mcp-toolbar"><span>{{ mcp.connected ? `发现 ${mcp.tools.length} 个工具` : (mcp.error || '等待连接') }}</span><button class="outline-action compact" :disabled="mcpLoading" @click="loadMcp"><PhArrowsClockwise :size="15" />{{ mcpLoading ? '检测中…' : '重新检测' }}</button></div>
              <div v-if="mcp.tools.length" class="tool-card-grid">
                <label v-for="tool in mcp.tools" :key="tool.name" class="switch-card" :title="tool.description">
                  <div><strong>{{ tool.name }}</strong><small>MCP TOOL</small></div>
                  <input v-model="tool.enabled" type="checkbox" /><span class="switch"></span>
                </label>
              </div>
              <div v-else class="mcp-empty">没有可显示的 MCP 工具。</div>
            </div>
          </div>
        </section>
      </div>

      <footer class="settings-footer">
        <button class="text-action" @click="restoreDefault"><PhArrowsClockwise :size="16" />重置整个配置</button>
        <div><button class="outline-action" @click="save()"><PhFloppyDisk :size="16" />保存草稿</button><button class="primary-action" @click="apply"><PhRocketLaunch :size="16" />应用并热切换</button></div>
      </footer>
    </div>
  </main>
</template>
