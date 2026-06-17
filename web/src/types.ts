// 백엔드 server.py 의 WS payload 와 1:1 대응

export interface Connections {
  robot: boolean
  gripper: boolean
  camera: boolean
  nodes: { vision: boolean; motion: boolean; gripper: boolean; integration: boolean }
}

export interface RobotState {
  joints_rad: number[]
  joints_deg: number[]
  joint_vel: number[]
  tcp: number[] | null // [x,y,z,rx,ry,rz]
  tcp_solution_space: number | null
  moving: boolean
}

export interface GripperState {
  ready?: boolean
  torque_enabled?: boolean
  moving?: boolean
  in_position?: boolean
  grasp_detected?: boolean
  object_lost?: boolean
  status?: number
  moving_status?: number
  present_position?: number // 0~1150
  goal_position?: number
  present_current?: number // mA
  current_limit?: number
  present_velocity?: number
  present_temperature?: number
  status_text?: string
}

export interface Pose {
  frame?: string
  position: { x: number; y: number; z: number }
  orientation: { x: number; y: number; z: number; w: number }
}

export interface VisionState {
  grasp_class: string | null
  pick_pose: Pose | null
  target_pose: Pose | null
  grasp_candidates: Pose[]
  obstacles: string | null
  shelf_slots: string[]
}

export interface Snapshot {
  connections: Connections
  state: string
  robot: RobotState
  gripper: GripperState
  vision: VisionState
}

export interface CurrentSeries {
  series: [number, number][] // [t, mA]
  goal: number | null
  limit: number | null
}

export interface Metrics {
  attempts: number
  grasp_success_rate_current: number | null
  grasp_success_rate_pose: number | null
  place_success_rate: number | null
  avg_tact_time: number | null
  error_count: number
}

export interface TelemetryPayload {
  snapshot: Snapshot
  current_series: CurrentSeries
  metrics: Metrics
  camera_available: boolean
}
