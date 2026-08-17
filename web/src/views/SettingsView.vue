<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { api } from '../api'

const profiles = ref<any[]>([]); const scope = ref<'web'|'forum'>('web'); const status = ref(''); const tools = ref<any[]>([])
const mcp = ref<any>({ url: null, configured: false, connected: false, error: null, tools: [] })
const mcpLoading = ref(false)
const current = () => profiles.value.find(item => item.scope === scope.value)
async function load() { profiles.value = await api('/api/settings/profiles'); await loadScopeSettings() }
async function loadScopeSettings() { await Promise.all([loadTools(), loadMcp()]) }
async function loadTools() { tools.value = await api(`/api/settings/tools/${scope.value}`) }
async function loadMcp() {
  mcpLoading.value = true
  try { mcp.value = await api(`/api/settings/mcp/${scope.value}`) }
  catch (error) { mcp.value = { url: null, configured: false, connected: false, error: String(error), tools: [] } }
  finally { mcpLoading.value = false }
}
async function save() {
  const item=current()
  item.draft.enabled_tools = tools.value.filter(tool => tool.enabled).map(tool => tool.name)
  if (mcp.value.connected) item.draft.disabled_mcp_tools = mcp.value.tools.filter((tool:any) => !tool.enabled).map((tool:any) => tool.name)
  await api(`/api/settings/profiles/${scope.value}/draft`, {method:'PUT', body:JSON.stringify(item.draft)})
  status.value='草稿已保存'
}
async function apply() { try { await save(); const result:any=await api(`/api/settings/profiles/${scope.value}/apply`, {method:'POST'}); status.value=`已应用 revision ${result.active_revision}`; await load() } catch (error) { status.value=String(error) } }
async function testProvider() { try { await save(); const result:any=await api(`/api/settings/profiles/${scope.value}/provider-test`, {method:'POST'}); status.value=result.message } catch (error) { status.value=String(error) } }
async function restoreDefault() { await api(`/api/settings/profiles/${scope.value}/restore-default`, {method:'POST'}); status.value='已恢复默认草稿（尚未应用）'; await load() }
onMounted(load)
</script>

<template><main class="settings-page"><div class="settings-title"><div><h1>运行设置</h1><p>论坛自动回复和网页对话使用独立的 Provider、Prompt 与工具开关。</p></div><span v-if="status" class="status">{{ status }}</span></div>
  <div class="scope-tabs"><button :class="{active:scope==='web'}" @click="scope='web';loadScopeSettings()">网页对话</button><button :class="{active:scope==='forum'}" @click="scope='forum';loadScopeSettings()">论坛自动回复</button></div>
  <section v-if="current()" class="settings-grid">
    <div class="panel"><h3>模型供应商</h3><p>Active revision: {{ current().active_revision }}</p><label>Provider<select v-model="current().draft.provider"><option>openrouter</option><option>deepseek</option><option>tongyi</option><option>mimo</option></select></label><label>模型<input v-model="current().draft.model" /></label><label>Fallback<input v-model="current().draft.fallback_model" placeholder="可选" /></label><label>API Key<input v-model="current().draft.api_key" type="password" :placeholder="current().secret?.configured ? `已配置 ····${current().secret.last_four}` : '输入新密钥'" /></label><button class="secondary" @click="testProvider">测试连接</button></div>
    <div class="panel prompt-panel"><h3>System Prompt</h3><textarea v-model="current().draft.system_prompt" rows="18"></textarea></div>
    <div class="panel tools-panel"><h3>内置工具</h3><p>论坛读取、图片与记忆工具；关闭后在下一次应用 Runtime 时生效。</p><label v-for="tool in tools" :key="tool.name" class="tool-row"><input type="checkbox" v-model="tool.enabled"/><span>{{ tool.name }}</span><small>{{ tool.loaded === false ? '加载失败' : tool.source }}</small></label></div>
    <div class="panel tools-panel mcp-panel">
      <div class="panel-heading"><h3>MCP</h3><button class="secondary" :disabled="mcpLoading" @click="loadMcp">{{ mcpLoading ? '检测中…' : '重新检测' }}</button></div>
      <p class="mcp-url"><strong>服务器：</strong>{{ mcp.url || '未配置 MCP_SERVER_URL' }}</p>
      <p><span class="connection-dot" :class="mcp.connected ? 'connected' : 'disconnected'"></span>{{ mcp.connected ? `已连接 · ${mcp.tools.length} 个工具` : (mcp.error || '未连接') }}</p>
      <p>新发现的 MCP 工具默认启用；取消勾选后点击“应用并热切换”生效。</p>
      <label v-for="tool in mcp.tools" :key="tool.name" class="tool-row" :title="tool.description"><input type="checkbox" v-model="tool.enabled"/><span>{{ tool.name }}</span><small>MCP</small></label>
    </div>
  </section>
  <div class="settings-actions"><button class="secondary" @click="restoreDefault">恢复默认草稿</button><button class="secondary" @click="save">保存草稿</button><button @click="apply">应用并热切换</button></div>
</main></template>
