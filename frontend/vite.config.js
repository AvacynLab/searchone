import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'process.env.SEARCHONE_SENTRY_DSN': JSON.stringify(process.env.SEARCHONE_SENTRY_DSN || ''),
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:2001',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
    port: 2000,
  },
})
