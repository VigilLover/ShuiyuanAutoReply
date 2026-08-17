<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { api, setCsrfToken } from './api'

const authenticated = ref(false)
const token = ref('')
const error = ref('')
onMounted(async () => { try { const result:any = await api('/api/bootstrap'); setCsrfToken(result.csrf_token); authenticated.value = true } catch {} })
async function login() {
  error.value = ''
  try { const result:any = await api('/api/admin/login', { method: 'POST', body: JSON.stringify({ token: token.value }) }); setCsrfToken(result.csrf_token); authenticated.value = true }
  catch (reason) { error.value = String(reason) }
}
</script>

<template>
  <div v-if="!authenticated" class="login-shell">
    <form class="login-card" @submit.prevent="login">
      <div class="brand-mark">水</div><h1>Shuiyuan Auto Reply</h1>
      <p>输入启动终端中显示的本地管理令牌。</p>
      <input v-model="token" type="password" autofocus placeholder="管理令牌" />
      <button>进入管理站</button><p v-if="error" class="error">{{ error }}</p>
    </form>
  </div>
  <div v-else class="app-shell">
    <header><div class="brand"><span class="brand-mark small">水</span><strong>Shuiyuan</strong></div>
      <nav><RouterLink to="/">对话</RouterLink><RouterLink to="/settings">设置</RouterLink></nav>
    </header>
    <RouterView />
  </div>
</template>
