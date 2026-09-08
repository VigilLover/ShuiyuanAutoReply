import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import { resolve } from 'node:path'

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  build: {
    outDir: resolve(__dirname, '../src/shuiyuan_auto_reply/interfaces/api/static'),
    emptyOutDir: true,
  },
  server: { proxy: { '/api': 'http://127.0.0.1:11451' } },
})
