import { useEffect, useRef, useState } from 'react'
import type { TelemetryPayload } from '../types'

// WebSocket 텔레메트리 구독 훅. 끊기면 자동 재연결(1s).
export function useTelemetry() {
  const [data, setData] = useState<TelemetryPayload | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    let stop = false
    let retry: ReturnType<typeof setTimeout>

    const connect = () => {
      if (stop) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onmessage = (ev) => {
        try {
          setData(JSON.parse(ev.data))
        } catch {
          /* ignore malformed frame */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!stop) retry = setTimeout(connect, 1000)
      }
      ws.onerror = () => ws.close()
    }

    connect()
    return () => {
      stop = true
      clearTimeout(retry)
      wsRef.current?.close()
    }
  }, [])

  return { data, connected }
}
