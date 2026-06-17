import { useEffect, useState } from 'react'
import { Panel } from './common'
import { api } from '../api'

interface LogEntry { t: number; level: number; name: string; msg: string }
interface EventEntry { id: number; ts: number; kind: string; success: number | null; detail: string | null }

function fmt(ts: number) {
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString('ko-KR', { hour12: false }) +
    '.' + String(d.getMilliseconds()).padStart(3, '0').slice(0, 2)
}

// 에러 로그(rosout) + 이벤트 로그(파지/배치/에러) 폴링 표시.
export function LogsPanel() {
  const [errors, setErrors] = useState<LogEntry[]>([])
  const [events, setEvents] = useState<EventEntry[]>([])

  useEffect(() => {
    const tick = () =>
      api.getLogs(60).then((d) => {
        setErrors(d.rosout_errors ?? [])
        setEvents(d.events ?? [])
      }).catch(() => {})
    tick()
    const id = setInterval(tick, 2000)
    return () => clearInterval(id)
  }, [])

  return (
    <Panel title="로그 (Logs)" className="col-8">
      <div className="row2">
        <div>
          <div className="muted" style={{ marginBottom: 6 }}>이벤트 / 상태 전이</div>
          <div className="log">
            {events.length === 0 && <div className="muted" style={{ padding: 8 }}>기록 없음</div>}
            {events.map((e) => (
              <div className="line" key={e.id}>
                <span className="ts">{fmt(e.ts)}</span>
                <span className={`tag`}>{e.kind}</span>
                <span className={e.success === 0 ? 'lvl-err' : e.success === 1 ? '' : 'muted'}>
                  {e.success === 1 ? 'OK' : e.success === 0 ? 'FAIL' : ''} {e.detail ?? ''}
                </span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="muted" style={{ marginBottom: 6 }}>ERROR / FATAL (rosout)</div>
          <div className="log">
            {errors.length === 0 && <div className="muted" style={{ padding: 8 }}>에러 없음</div>}
            {errors.map((e, i) => (
              <div className="line" key={i}>
                <span className="ts">{fmt(e.t)}</span>
                <span className="tag">{e.name}</span>
                <span className={e.level >= 50 ? 'lvl-err' : 'lvl-warn'}>{e.msg}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Panel>
  )
}
