import { useEffect, useState } from 'react'
import { api } from '../api'
import { Panel } from './common'
import type { GripperState } from '../types'

// 전 기능 제어 패널: 로봇 명명 이동/조그, 그리퍼 위치/파지/토크/모션프로파일, 파이프라인.
export function ControlPanel({ gripper }: { gripper: GripperState }) {
  const [limits, setLimits] = useState({ position_max: 1150, current_max: 820 })
  const [pos, setPos] = useState(500)
  const [graspPos, setGraspPos] = useState(740)
  const [maxCurrent, setMaxCurrent] = useState(300)
  const [deltaTh, setDeltaTh] = useState(20)
  const [vel, setVel] = useState(60)
  const [acc, setAcc] = useState(40)
  const [jog, setJog] = useState<number[]>([0, 0, 0, 0, 0, 0])
  const [msg, setMsg] = useState('')

  useEffect(() => {
    api.getLimits().then(setLimits).catch(() => {})
  }, [])

  const flash = (r: any) =>
    setMsg(r?.ok ? `✓ ${r.label ?? '전송됨'}` : `✗ ${r?.error ?? '실패'}`)

  return (
    <Panel title="제어 (Control)" className="col-4">
      {/* 로봇 명명 이동 */}
      <div className="muted" style={{ marginBottom: 6 }}>로봇 자세</div>
      <div className="btn-row">
        <button onClick={() => api.moveNamed('home').then(flash)}>HOME</button>
        <button onClick={() => api.moveNamed('shelf_view').then(flash)}>매대뷰</button>
        <button onClick={() => api.moveNamed('product_view').then(flash)}>상품뷰</button>
        <button onClick={() => api.moveNamed('place').then(flash)}>배치</button>
      </div>

      {/* 조인트 조그 (상대 이동) */}
      <label className="field">조인트 조그 (상대, °) — vel {vel} / acc {acc}</label>
      <div className="row3">
        {jog.map((v, i) => (
          <input key={i} type="number" value={v}
            onChange={(e) => {
              const n = [...jog]; n[i] = parseFloat(e.target.value) || 0; setJog(n)
            }} />
        ))}
      </div>
      <div className="row2" style={{ marginTop: 8 }}>
        <input type="number" value={vel} onChange={(e) => setVel(+e.target.value)} placeholder="vel" />
        <input type="number" value={acc} onChange={(e) => setAcc(+e.target.value)} placeholder="acc" />
      </div>
      <div className="btn-row" style={{ marginTop: 8 }}>
        <button className="primary"
          onClick={() => api.moveJoint(jog, vel, acc, true).then(flash)}>
          조그 실행
        </button>
        <button className="ghost" onClick={() => setJog([0, 0, 0, 0, 0, 0])}>초기화</button>
      </div>

      <hr style={{ borderColor: 'var(--border)', margin: '14px 0' }} />

      {/* 그리퍼 위치 */}
      <label className="field">
        그리퍼 위치: {pos} (현재 {gripper.present_position ?? '-'} / {limits.position_max})
      </label>
      <input type="range" min={0} max={limits.position_max} value={pos}
        onChange={(e) => setPos(+e.target.value)} />
      <div className="btn-row" style={{ marginTop: 6 }}>
        <button onClick={() => api.gripperSetPosition(pos).then(flash)}>이동</button>
        <button onClick={() => api.gripperSetPosition(0).then(flash)}>열기(0)</button>
        <button onClick={() => api.gripperSetPosition(limits.position_max).then(flash)}>닫기</button>
      </div>

      {/* 안전 파지 (힘 제어) */}
      <label className="field">파지 목표 위치: {graspPos}</label>
      <input type="range" min={0} max={limits.position_max} value={graspPos}
        onChange={(e) => setGraspPos(+e.target.value)} />
      <label className="field">최대 전류(파지힘, mA): {maxCurrent}</label>
      <input type="range" min={0} max={limits.current_max} value={maxCurrent}
        onChange={(e) => setMaxCurrent(+e.target.value)} />
      <label className="field">전류 delta 임계: {deltaTh}</label>
      <input type="range" min={0} max={200} value={deltaTh}
        onChange={(e) => setDeltaTh(+e.target.value)} />
      <button className="primary" style={{ marginTop: 8, width: '100%' }}
        onClick={() => api.gripperSafeGrasp(graspPos, maxCurrent, deltaTh).then(flash)}>
        SAFE GRASP 실행
      </button>

      {/* 토크 */}
      <div className="btn-row" style={{ marginTop: 10 }}>
        <button onClick={() => api.gripperTorque(true).then(flash)}>토크 ON</button>
        <button onClick={() => api.gripperTorque(false).then(flash)}>토크 OFF</button>
        <button onClick={() => api.gripperMotionProfile(vel, acc).then(flash)}>모션프로파일</button>
      </div>

      {msg && <div className="tag" style={{ marginTop: 12, display: 'block' }}>{msg}</div>}
    </Panel>
  )
}
