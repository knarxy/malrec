import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // dev server talks to the API directly; the container build uses nginx
    proxy: { '/api': { target: 'http://localhost:8000', rewrite: p => p.replace(/^\/api/, '') } },
  },
  build: { outDir: 'dist', sourcemap: false },
})
