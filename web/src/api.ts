// REST 제어 호출 래퍼. 모든 제어는 백엔드 /api/control/* 로.

async function post(path: string, body?: unknown): Promise<any> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  return res.json().catch(() => ({ ok: res.ok }))
}

async function get(path: string): Promise<any> {
  const res = await fetch(path)
  return res.json()
}

export const api = {
  // 로봇
  moveNamed: (name: string) => post('/api/control/move_named', { name }),
  moveJoint: (pos_deg: number[], vel = 30, acc = 30, relative = false) =>
    post('/api/control/move_joint', { pos_deg, vel, acc, relative }),
  estop: (mode = 0) => post('/api/control/estop', { mode }),

  // 그리퍼
  gripperSetPosition: (position: number, timeout = 5) =>
    post('/api/control/gripper/set_position', { position, timeout }),
  gripperSafeGrasp: (
    target_position: number,
    max_current: number,
    current_delta_threshold = 20,
    timeout = 5,
  ) =>
    post('/api/control/gripper/safe_grasp', {
      target_position,
      max_current,
      current_delta_threshold,
      timeout,
    }),
  gripperTorque: (enable: boolean) => post('/api/control/gripper/torque', { enable }),
  gripperMotionProfile: (velocity: number, acceleration: number) =>
    post('/api/control/gripper/motion_profile', { velocity, acceleration }),

  // 지표
  markPlace: (success: boolean, detail = '') =>
    post('/api/metrics/place', { success, detail }),
  cycleStart: () => post('/api/metrics/cycle/start'),
  cycleEnd: (success = true) => post(`/api/metrics/cycle/end?success=${success}`),
  resetMetrics: () => post('/api/metrics/reset'),

  // 통합 컨트롤러 operator 버튼 (main_controller 의 /dashboard/operator_cmd)
  operatorCmd: (cmd: string) => post('/api/operator/cmd', { cmd }),

  // 조회
  getLogs: (n = 50) => get(`/api/logs?n=${n}`),
  getMetrics: () => get('/api/metrics'),
  getLimits: () => get('/api/limits'),
}
