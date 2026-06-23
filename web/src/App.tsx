import { useTelemetry } from './hooks/useTelemetry'
import { Panel, Pill, KV, Stat, Gauge } from './components/common'
import { CurrentChart } from './components/CurrentChart'
import { CameraFeed } from './components/CameraFeed'
import { OperatorPanel } from './components/OperatorPanel'
import { api } from './api'
import type { Snapshot } from './types'

const JOINT_LIMIT_DEG = 360 // 바 정규화용(시각화 한도)

function ConnectionPanel({ s }: { s: Snapshot }) {
  const c = s.connections
  return (
    <Panel title="시스템 상태" className="col-4"
      right={<span className={`state-pill state-${s.state}`}>{s.state}</span>}>
      <div className="muted" style={{ marginBottom: 8 }}>노드 연결</div>
      <div className="btn-row" style={{ marginBottom: 12 }}>
        <Pill ok={c.nodes.vision} label="vision" />
        <Pill ok={c.nodes.motion} label="motion" />
        <Pill ok={c.nodes.gripper} label="gripper" />
        <Pill ok={c.nodes.integration} warn={!c.nodes.integration} label="integration" />
      </div>
      <KV k="로봇 (E0509)" v={<Pill ok={c.robot} label={c.robot ? '연결됨' : '끊김'} />} />
      <KV k="그리퍼 (Modbus/TCP)" v={<Pill ok={c.gripper} label={c.gripper ? '연결됨' : '끊김'} />} />
      <KV k="카메라 (RealSense)" v={<Pill ok={c.camera} label={c.camera ? '스트리밍' : '대기'} />} />
    </Panel>
  )
}

function RobotPanel({ s }: { s: Snapshot }) {
  const r = s.robot
  return (
    <Panel title="실시간 로봇 상태" className="col-4"
      right={<Pill ok={!r.moving} warn={r.moving} label={r.moving ? '이동 중' : '정지'} />}>
      {r.joints_deg.map((d, i) => (
        <div className="jointrow" key={i}>
          <span className="name">J{i + 1}</span>
          <div className="bar">
            <span style={{
              left: '50%',
              width: `${Math.min(50, Math.abs(d) / JOINT_LIMIT_DEG * 100)}%`,
              transform: d < 0 ? 'translateX(-100%)' : 'none',
            }} />
          </div>
          <span className="val">{d.toFixed(1)}°</span>
        </div>
      ))}
      <hr style={{ borderColor: 'var(--border)', margin: '12px 0' }} />
      <div className="muted" style={{ marginBottom: 6 }}>TCP 위치 (base, mm/°)</div>
      {r.tcp ? (
        <div className="row3">
          {['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz'].map((ax, i) => (
            <KV key={ax} k={ax} v={r.tcp![i]?.toFixed(1)} />
          ))}
        </div>
      ) : (
        <div className="muted">posx 서비스 대기 중…</div>
      )}
    </Panel>
  )
}

function GripperPanel({ s }: { s: Snapshot }) {
  const g = s.gripper
  const cur = g.present_current ?? 0
  return (
    <Panel title="실시간 그리퍼 상태" className="col-4"
      right={<Pill ok={!!g.grasp_detected} warn={!g.grasp_detected}
        label={g.grasp_detected ? 'GRASP 감지' : '미감지'} />}>
      <Gauge label="위치" value={g.present_position ?? 0} max={1150} unit="" color="var(--accent)" />
      <Gauge label="전류" value={cur} max={Math.max(g.current_limit ?? 820, cur, 100)}
        unit="mA" color="var(--grasp)" />
      <div className="row2" style={{ marginTop: 10 }}>
        <KV k="현재 전류" v={`${cur} mA`} />
        <KV k="목표/제한" v={`${g.current_limit ?? '-'} mA`} />
        <KV k="목표 위치" v={g.goal_position ?? '-'} />
        <KV k="온도" v={`${g.present_temperature ?? '-'}°`} />
        <KV k="토크" v={g.torque_enabled ? 'ON' : 'OFF'} />
        <KV k="object_lost" v={g.object_lost ? 'YES' : 'no'} />
      </div>
      {g.status_text && <div className="tag" style={{ marginTop: 8, display: 'block' }}>{g.status_text}</div>}
    </Panel>
  )
}

function RvizPanel() {
  // RViz 3D 화면(로봇모델 + cuRobo 충돌구체) MJPEG 스트림
  return (
    <Panel title="RViz — 충돌구체 3D 뷰" className="col-4"
      right={<span className="tag">collision spheres</span>}>
      <div className="camera">
        <img src="/api/rviz/stream" alt="rviz collision spheres view"
          style={{ width: '100%', display: 'block', borderRadius: 8 }} />
      </div>
    </Panel>
  )
}

function ShelfInventoryPanel({ s }: { s: Snapshot }) {
  // 매대 물품 품목: 캔/바틀/스낵 (처음 매대확인 시 있으면 1, 없으면 0; 진열하면 1)
  const inv = s.vision.shelf_inventory ?? { can: 0, bottle: 0, snack: 0 }
  const items: [string, number][] = [
    ['캔 (can)', inv.can],
    ['바틀 (bottle)', inv.bottle],
    ['스낵 (snack)', inv.snack],
  ]
  return (
    <Panel title="매대 물품 품목" className="col-4">
      <div className="stat-grid">
        {items.map(([label, n]) => (
          <Stat key={label} label={label} value={`${n}개`} tone={n > 0 ? 'ok' : 'warn'} />
        ))}
      </div>
    </Panel>
  )
}

function ManualControlPanel() {
  // 웹캠 키 = 명칭 버튼 (대시보드에서 클릭으로 사용). /dashboard/operator_cmd 발행
  const send = (cmd: string) => { api.operatorCmd(cmd) }
  const btns: [string, string][] = [
    ['home', '🏠 홈위치 (H)'],
    ['shelf', '🛒 매대위치 (V)'],
    ['grasp', '✋ 파지생성 (G)'],
    ['pick', '⬇ 전진+집기 (P)'],
    ['open', '✊ 그리퍼 열기 (O)'],
    ['unlock', '✖ 취소/언락 (R)'],
  ]
  const nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
  return (
    <Panel title="수동 제어 (웹캠 키)" className="col-4">
      <div className="muted" style={{ marginBottom: 6 }}>물체 선택 (화면 번호로 lock)</div>
      <div className="btn-row" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
        {nums.map((n) => (
          <button key={n} onClick={() => send(`lock:${n}`)}
            style={{ flex: '1 1 9%', minWidth: 34 }}>{n}</button>
        ))}
      </div>
      <div className="muted" style={{ marginBottom: 6 }}>동작</div>
      <div className="btn-row" style={{ flexWrap: 'wrap', gap: 8 }}>
        {btns.map(([cmd, label]) => (
          <button key={cmd} onClick={() => send(cmd)}
            style={{ flex: '1 1 46%' }}>{label}</button>
        ))}
      </div>
    </Panel>
  )
}

export default function App() {
  const { data, connected } = useTelemetry()

  return (
    <>
      <div className="topbar">
        <h1>🤖 Smart Shelf Robot — Operations Dashboard</h1>
        <span className="spacer" />
        {data && <span className={`state-pill state-${data.snapshot.state}`}>{data.snapshot.state}</span>}
        <Pill ok={connected} label={connected ? 'WS 연결됨' : 'WS 끊김'} />
        <button className="estop" onClick={() => api.estop(0)}>⏹ E-STOP</button>
      </div>

      {!data ? (
        <div style={{ padding: 40, textAlign: 'center', color: 'var(--muted)' }}>
          백엔드(dashboard_node)에 연결 중…
        </div>
      ) : (
        <div className="grid">
          <ConnectionPanel s={data.snapshot} />
          <RobotPanel s={data.snapshot} />
          <GripperPanel s={data.snapshot} />

          <Panel title="카메라 라이브 피드" className="col-5">
            <CameraFeed available={data.camera_available} />
          </Panel>
          <RvizPanel />
          <Panel title="그리퍼 전류 (최근 10s)" className="col-3">
            <CurrentChart data={data.current_series} />
          </Panel>

          <ShelfInventoryPanel s={data.snapshot} />
          <OperatorPanel state={data.snapshot.state} />
          <ManualControlPanel />
        </div>
      )}
    </>
  )
}
