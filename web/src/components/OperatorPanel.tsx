import { useState } from 'react'
import { api } from '../api'
import { Panel } from './common'

// 통합 컨트롤러(main_controller) 수동 진행 패널.
// 빈 슬롯(보충 대상)과 "감지 완료"를 비전 토픽 대신 여기 버튼으로 제어한다.
const CLASSES = ['can', 'bottle', 'snack_bag']

export function OperatorPanel({ state }: { state: string }) {
  const [msg, setMsg] = useState('')
  const send = (cmd: string) =>
    api.operatorCmd(cmd).then((r) =>
      setMsg(r?.ok ? `✓ ${cmd}` : `✗ ${r?.error ?? '실패'}`))

  const is = (s: string) => state === s

  return (
    <Panel title="통합 컨트롤러 (Operator)" className="col-4"
      right={<span className={`state-pill state-${state}`}>{state}</span>}>
      <div className="muted" style={{ marginBottom: 6 }}>사이클 시작</div>
      <div className="btn-row">
        <button className={is('IDLE') ? 'primary' : ''}
          onClick={() => send('start')}>▶ START</button>
      </div>

      <label className="field">보충 대상 선택 (SHELF_DETECTING 통과)</label>
      <div className="btn-row">
        {CLASSES.map((c) => (
          <button key={c} className={is('SHELF_DETECTING') ? 'primary' : ''}
            onClick={() => send(`restock:${c}`)}>{c}</button>
        ))}
      </div>

      <label className="field">상품 감지 완료 (PRODUCT_DETECTING 통과 → 픽)</label>
      <div className="btn-row">
        <button className={is('PRODUCT_DETECTING') ? 'primary' : ''}
          onClick={() => send('confirm')}>✓ 감지 완료 / 픽 진행</button>
      </div>

      <hr style={{ borderColor: 'var(--border)', margin: '14px 0' }} />
      <div className="btn-row">
        <button className="danger" onClick={() => send('abort')}>⏹ ABORT</button>
        <button onClick={() => send('reset')}>RESET (ERROR/HALT 해제)</button>
      </div>

      {msg && <div className="tag" style={{ marginTop: 12, display: 'block' }}>{msg}</div>}
    </Panel>
  )
}
