<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import {
  PhArrowsClockwise,
  PhChatCircleText,
  PhCheck,
  PhCpu,
  PhFloppyDisk,
  PhGlobe,
  PhPlugsConnected,
  PhRocketLaunch,
  PhTextT,
  PhX,
} from '@phosphor-icons/vue'
import { api } from '../api'

const profiles = ref<any[]>([])
const scope = ref<'web' | 'forum'>('web')
const status = ref('')
const statusError = ref(false)
const tools = ref<any[]>([])
const mcp = ref<any>({ url: null, configured: false, connected: false, error: null, tools: [] })
const mcpLoading = ref(false)
const activeSection = ref<'model' | 'prompt' | 'tools'>('model')

const current = () => profiles.value.find(item => item.scope === scope.value)

async function load() {
  profiles.value = await api('/api/settings/profiles')
  await loadScopeSettings()
}

async function loadScopeSettings() {
  status.value = ''
  await Promise.all([loadTools(), loadMcp()])
}

async function changeScope(value: 'web' | 'forum') {
  scope.value = value
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

async function save(showStatus = true) {
  const item = current()
  item.draft.enabled_tools = tools.value.filter(tool => tool.enabled).map(tool => tool.name)
  if (mcp.value.connected) {
    item.draft.disabled_mcp_tools = mcp.value.tools.filter((tool: any) => !tool.enabled).map((tool: any) => tool.name)
  }
  await api(`/api/settings/profiles/${scope.value}/draft`, { method: 'PUT', body: JSON.stringify(item.draft) })
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

async function restoreDefault() {
  await api(`/api/settings/profiles/${scope.value}/restore-default`, { method: 'POST' })
  setStatus('已恢复默认草稿，应用后生效')
  await load()
}

function setStatus(message: string, error = false) {
  status.value = message
  statusError.value = error
}

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
          <button :class="{ active: activeSection === 'model' }" @click="activeSection = 'model'"><PhCpu :size="20" /><div>模型<small>Provider 与密钥</small></div></button>
          <button :class="{ active: activeSection === 'prompt' }" @click="activeSection = 'prompt'"><PhTextT :size="20" /><div>提示词<small>System Prompt</small></div></button>
          <button :class="{ active: activeSection === 'tools' }" @click="activeSection = 'tools'"><PhPlugsConnected :size="20" /><div>工具与 MCP<small>能力开关</small></div></button>
        </aside>

        <section v-if="current()" class="settings-content">
          <div class="settings-content-head">
            <div><span class="eyebrow">{{ scope === 'web' ? 'WEB RUNTIME' : 'FORUM RUNTIME' }}</span><h2>{{ activeSection === 'model' ? '模型配置' : activeSection === 'prompt' ? 'System Prompt' : '工具与 MCP' }}</h2></div>
            <span class="revision-badge">ACTIVE · r{{ current().active_revision }}</span>
          </div>

          <div v-if="activeSection === 'model'" class="settings-section">
            <p class="section-intro">网页与论坛 Agent 已统一接入 DeepSeek 原生视觉模型。API Key 保存后只显示配置状态和末四位。</p>
            <div class="provider-grid">
              <div class="provider-card selected">
                <strong>DeepSeek</strong>
                <span><PhCheck :size="15" weight="bold" /> Vision 固定模型</span>
              </div>
            </div>
            <div class="form-grid">
              <label class="full-field"><span>模型名称</span><input value="deepseek-v4-flash-vision-exp" readonly /></label>
              <label class="full-field"><span>API Key</span><input v-model="current().draft.api_key" type="password" :placeholder="current().secret?.configured ? `已配置 ····${current().secret.last_four}` : '输入新密钥'" /></label>
            </div>
            <button class="outline-action" @click="testProvider"><PhPlugsConnected :size="16" />测试 Provider 连接</button>
          </div>

          <div v-else-if="activeSection === 'prompt'" class="settings-section prompt-section">
            <p class="section-intro">网页和论坛使用相互独立的完整 System Prompt。修改后需要应用 Runtime。</p>
            <textarea v-model="current().draft.system_prompt" spellcheck="false"></textarea>
          </div>

          <div v-else class="settings-section tool-settings">
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
        <button class="text-action" @click="restoreDefault"><PhArrowsClockwise :size="16" />恢复默认</button>
        <div><button class="outline-action" @click="save()"><PhFloppyDisk :size="16" />保存草稿</button><button class="primary-action" @click="apply"><PhRocketLaunch :size="16" />应用并热切换</button></div>
      </footer>
    </div>
  </main>
</template>
