import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  build: {
    rolldownOptions: {
      output: {
        // Library code changes far less often than the editor: keeping the
        // notation, audio and React runtimes in their own chunks lets a
        // returning browser reuse them across deployments and fetch the
        // first load in parallel instead of as one 1.8 MB script.
        advancedChunks: {
          groups: [
            { name: 'vexflow', test: /node_modules[\\/]vexflow[\\/]/ },
            { name: 'audio', test: /node_modules[\\/](tone|smplr|standardized-audio-context|automation-events)[\\/]/ },
            { name: 'react', test: /node_modules[\\/](react|react-dom|scheduler|zustand)[\\/]/ },
          ],
        },
      },
    },
  },
})
