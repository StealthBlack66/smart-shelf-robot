import { useEffect, useRef } from 'react'
import uPlot from 'uplot'
import type { CurrentSeries } from '../types'

// 그리퍼 전류 시계열 그래프 (최근 ~10s). uPlot 으로 고주파 갱신에도 가볍게.
const WINDOW_SEC = 10

export function CurrentChart({ data }: { data: CurrentSeries | undefined }) {
  const ref = useRef<HTMLDivElement>(null)
  const plotRef = useRef<uPlot | null>(null)

  useEffect(() => {
    if (!ref.current) return
    const opts: uPlot.Options = {
      width: ref.current.clientWidth,
      height: 220,
      legend: { show: true },
      cursor: { show: true },
      scales: { x: { time: false }, y: { auto: true } },
      axes: [
        { stroke: '#8b949e', grid: { stroke: '#2a313c' }, ticks: { stroke: '#2a313c' },
          values: (_u, vals) => vals.map((v) => `${v.toFixed(0)}s`) },
        { stroke: '#8b949e', grid: { stroke: '#2a313c' }, ticks: { stroke: '#2a313c' } },
      ],
      series: [
        {},
        { label: '전류(mA)', stroke: '#a371f7', width: 2, fill: 'rgba(163,113,247,.12)' },
        { label: '제한(mA)', stroke: '#d29922', width: 1, dash: [6, 4] },
      ],
    }
    const u = new uPlot(opts, [[], [], []], ref.current)
    plotRef.current = u
    const onResize = () => u.setSize({ width: ref.current!.clientWidth, height: 220 })
    window.addEventListener('resize', onResize)
    return () => {
      window.removeEventListener('resize', onResize)
      u.destroy()
    }
  }, [])

  useEffect(() => {
    const u = plotRef.current
    if (!u || !data) return
    const pts = data.series
    if (pts.length === 0) return
    const tEnd = pts[pts.length - 1][0]
    const xs: number[] = []
    const ys: number[] = []
    const lim: number[] = []
    for (const [t, v] of pts) {
      const rel = t - tEnd // 우측 0초 기준 과거 음수
      if (rel < -WINDOW_SEC) continue
      xs.push(rel)
      ys.push(v)
      lim.push(data.limit ?? NaN)
    }
    u.setData([xs, ys, lim])
  }, [data])

  return <div ref={ref} />
}
