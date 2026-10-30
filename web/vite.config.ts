import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves the site from /<repo>/; local dev and tests from /.
export default defineConfig({
  base: process.env.GLASSBOX_BASE ?? '/',
  plugins: [react()],
  worker: { format: 'es' },
  // Benchmark results live at the repository root, outside web/.
  server: { fs: { allow: ['..'] } },
  build: { target: 'es2022' },
})
