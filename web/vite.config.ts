import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: resolve(__dirname, '../src/shuiyuan_auto_reply/interfaces/api/static'),
    emptyOutDir: true,
  },
  server: { proxy: { '/api': 'http://127.0.0.1:11451' } },
})
