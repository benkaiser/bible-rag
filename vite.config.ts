import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { statSync } from 'fs'

// https://vite.dev/config/
export default defineConfig({
  base: '/bible-rag/',
  plugins: [
    react(),
    {
      name: 'content-length',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          // Add content-length for static files in public/
          if (req.url && (req.url.endsWith('.bin') || req.url.endsWith('.json') || req.url.endsWith('.onnx') || req.url.endsWith('.onnx_data'))) {
            try {
              const filePath = `public${req.url}`
              const stat = statSync(filePath)
              res.setHeader('Content-Length', stat.size)
            } catch {
              // file not found, let vite handle it
            }
          }
          next()
        })
      },
    },
  ],
})
