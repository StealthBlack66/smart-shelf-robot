import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 개발 서버: /api 와 /ws 를 백엔드(uvicorn :8080)로 프록시.
// 빌드 결과(dist)는 백엔드가 직접 서빙(StaticFiles)하므로 운영 시 프록시 불필요.
// ※ 'localhost' 는 환경에 따라 IPv6(::1)로 풀려 백엔드(IPv4 127.0.0.1) 프록시가
//    실패(→ WS 끊김/연결 중 멈춤)할 수 있어 127.0.0.1 로 고정한다.
const BACKEND = process.env.DASH_BACKEND || 'http://127.0.0.1:8080'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/ws': { target: BACKEND, ws: true, changeOrigin: true },
    },
  },
  build: { outDir: 'dist', sourcemap: false },
})
