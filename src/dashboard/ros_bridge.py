"""ROS2 상태 집계기 — 모든 토픽 구독 + 서비스 폴링 + 스레드 안전 스냅샷.

설계 원칙
  · ROS 콜백은 전부 여기서 받아 StateStore(락 보호)에 기록만 한다.
  · 웹서버 스레드는 store.snapshot() 으로 최신값을 한 번에 읽는다 (콜백과 분리).
  · 의존 인터페이스(dsr_msgs2 / dsr_gripper_tcp_interfaces)는 없을 수도 있으므로
    import 를 방어적으로 처리하고, 없으면 해당 기능만 비활성화한다.

확인된 실제 토픽/서비스 (코드 기준):
  /dsr01/joint_states                         sensor_msgs/JointState   (로봇 6축)
  /gripper_service/state                      GripperState             (그리퍼, 20Hz)
  /dsr01/curobo/grasp_class                   std_msgs/String          (감지 클래스)
  /dsr01/curobo/pick_pose, /target_pose       geometry_msgs/PoseStamped(3D 위치)
  /dsr01/curobo/grasp_candidates              geometry_msgs/PoseArray
  /dsr01/curobo/obstacles                     std_msgs/String
  /dsr01/aux_control/get_current_posx         GetCurrentPosx (서비스)   (TCP 위치)
  /rosout                                     rcl_interfaces/Log        (에러 로그)
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, PoseArray
from std_msgs.msg import String

# ── 선택 의존: 로그(rosout) ──────────────────────────────────────────
try:
    from rcl_interfaces.msg import Log as RosoutLog
    _ROSOUT_AVAIL = True
except Exception:                       # pragma: no cover
    RosoutLog = None
    _ROSOUT_AVAIL = False

# ── 선택 의존: 그리퍼 상태 메시지 ───────────────────────────────────
try:
    from dsr_gripper_tcp_interfaces.msg import GripperState
    _GRIPPER_STATE_AVAIL = True
except Exception:                       # pragma: no cover
    GripperState = None
    _GRIPPER_STATE_AVAIL = False

# ── 선택 의존: 두산 posx 서비스 ─────────────────────────────────────
try:
    from dsr_msgs2.srv import GetCurrentPosx
    _POSX_AVAIL = True
except Exception:                       # pragma: no cover
    GetCurrentPosx = None
    _POSX_AVAIL = False


# 소스를 "연결됨"으로 볼 최대 무수신 시간(초). 이 시간 넘으면 stale=끊김.
STALE_SEC = {
    'robot': 2.0,        # joint_states (보통 100Hz)
    'gripper': 2.0,      # gripper state (20Hz)
    'vision': 10.0,      # curobo/* (이벤트성, 느림)
}
# 그리퍼 전류 시계열: 20Hz × 30s 여유분
CURRENT_HISTORY_LEN = 600
# E0509 조인트 이름(순서 보장용). joint_states 가 다른 순서로 와도 이 이름으로 정렬.
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']


def _now() -> float:
    return time.monotonic()


class StateStore:
    """스레드 안전 최신 상태 저장소. 콜백이 쓰고 웹서버가 읽는다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_seen: dict[str, float] = {}     # 소스별 마지막 수신 monotonic time
        self._t0 = _now()

        # 로봇
        self.joint_pos: list[float] = [0.0] * 6     # rad
        self.joint_vel: list[float] = [0.0] * 6
        self.tcp: Optional[list[float]] = None      # [x,y,z,rx,ry,rz] mm/deg
        self.tcp_sol_space: Optional[int] = None

        # 그리퍼 (GripperState 필드 그대로)
        self.gripper: dict[str, Any] = {}
        self.current_hist: deque[tuple[float, int]] = deque(maxlen=CURRENT_HISTORY_LEN)

        # 비전
        self.grasp_class: Optional[str] = None
        self.pick_pose: Optional[dict] = None
        self.target_pose: Optional[dict] = None
        self.grasp_candidates: list[dict] = []
        self.obstacles: Optional[str] = None
        self.shelf_slots: list[str] = []            # 슬롯 점유(미배선 시 빈 배열)
        # 매대 품목 재고 (캔/바틀/스낵, 각 0/1). 매대확인 결과 + 진열(place)로 갱신
        self.shelf_inventory: dict = {'can': 0, 'bottle': 0, 'snack': 0}

        # 의사 상태머신 / 에러 로그 링버퍼
        self.error_log: deque[dict] = deque(maxlen=200)

    # ── 쓰기(콜백 측) ────────────────────────────────────────────
    def _touch(self, source: str) -> None:
        self._last_seen[source] = _now()

    def set_joints(self, names: list[str], pos: list[float], vel: list[float]) -> None:
        with self._lock:
            # 이름 기반 정렬 (순서 불일치 방어)
            idx = {n: i for i, n in enumerate(names)}
            for j, name in enumerate(JOINT_NAMES):
                i = idx.get(name)
                if i is not None and i < len(pos):
                    self.joint_pos[j] = float(pos[i])
                    self.joint_vel[j] = float(vel[i]) if i < len(vel) else 0.0
            self._touch('robot')

    def set_tcp(self, vals: list[float], sol: Optional[int]) -> None:
        with self._lock:
            self.tcp = [float(v) for v in vals[:6]]
            self.tcp_sol_space = sol

    def set_gripper(self, data: dict[str, Any], cur: int) -> None:
        with self._lock:
            self.gripper = data
            self.current_hist.append((_now() - self._t0, int(cur)))
            self._touch('gripper')

    def set_grasp_class(self, v: str) -> None:
        with self._lock:
            self.grasp_class = v
            self._touch('vision')

    def set_pose(self, which: str, d: dict) -> None:
        with self._lock:
            setattr(self, which, d)
            self._touch('vision')

    def set_candidates(self, cands: list[dict]) -> None:
        with self._lock:
            self.grasp_candidates = cands
            self._touch('vision')

    def set_obstacles(self, v: str) -> None:
        with self._lock:
            self.obstacles = v
            self._touch('vision')

    def set_shelf_inventory(self, inv: dict) -> None:
        with self._lock:
            for k in ('can', 'bottle', 'snack'):
                if k in inv:
                    try:
                        self.shelf_inventory[k] = int(inv[k])
                    except Exception:
                        pass
            self._touch('vision')

    def add_error(self, entry: dict) -> None:
        with self._lock:
            self.error_log.appendleft(entry)

    # ── 읽기(웹서버 측) ──────────────────────────────────────────
    def _conn(self, source: str) -> bool:
        ts = self._last_seen.get(source)
        if ts is None:
            return False
        return (_now() - ts) < STALE_SEC.get(source, 2.0)

    def pseudo_state(self) -> str:
        """컨트롤러(미구현) 대신 관측 토픽으로 추정한 의사 상태머신 단계.

        실제 /system/state 가 발행되기 시작하면 server 에서 그 값으로 덮어쓴다.
        """
        g = self.gripper
        if not self._conn('robot') and not self._conn('gripper'):
            return 'DISCONNECTED'
        if g.get('grasp_detected'):
            return 'GRASPED'
        if g.get('moving') or any(abs(v) > 0.01 for v in self.joint_vel):
            return 'MOVING'
        if self.grasp_class:
            return 'OBJECT_DETECTED'
        return 'IDLE'

    def snapshot(self) -> dict[str, Any]:
        """브라우저로 보낼 단일 JSON 스냅샷."""
        with self._lock:
            conn = {
                'robot': self._conn('robot'),
                'gripper': self._conn('gripper'),
                'vision': self._conn('vision'),
            }
            # 노드 연결: motion/integration 은 직접 신호가 없어 robot/그래프로 근사
            joints_deg = [math.degrees(p) for p in self.joint_pos]
            moving = (any(abs(v) > 0.01 for v in self.joint_vel)
                      or bool(self.gripper.get('moving')))
            return {
                'connections': {
                    'robot': conn['robot'],
                    'gripper': conn['gripper'],
                    'camera': conn['vision'],   # vision 노드 = 카메라 스트림 출처
                    'nodes': {
                        'vision': conn['vision'],
                        'motion': conn['robot'],     # motion 노드가 joint 구독/구동
                        'gripper': conn['gripper'],
                        'integration': False,        # 미구현
                    },
                },
                'state': self.pseudo_state(),
                'robot': {
                    'joints_rad': list(self.joint_pos),
                    'joints_deg': joints_deg,
                    'joint_vel': list(self.joint_vel),
                    'tcp': self.tcp,
                    'tcp_solution_space': self.tcp_sol_space,
                    'moving': moving,
                },
                'gripper': dict(self.gripper),
                'vision': {
                    'grasp_class': self.grasp_class,
                    'pick_pose': self.pick_pose,
                    'target_pose': self.target_pose,
                    'grasp_candidates': self.grasp_candidates,
                    'obstacles': self.obstacles,
                    'shelf_slots': self.shelf_slots,
                    'shelf_inventory': dict(self.shelf_inventory),
                },
            }

    def current_series(self) -> dict[str, Any]:
        """전류 시계열 (그래프용). [[t, mA], ...]"""
        with self._lock:
            return {
                'series': list(self.current_hist),
                'goal': self.gripper.get('goal_current'),
                'limit': self.gripper.get('current_limit'),
            }

    def errors(self, n: int = 50) -> list[dict]:
        with self._lock:
            return list(self.error_log)[:n]


def _pose_to_dict(msg: PoseStamped) -> dict:
    p, o = msg.pose.position, msg.pose.orientation
    return {
        'frame': msg.header.frame_id,
        'position': {'x': p.x, 'y': p.y, 'z': p.z},
        'orientation': {'x': o.x, 'y': o.y, 'z': o.z, 'w': o.w},
    }


class RosBridge(Node):
    """모든 구독 + posx 폴링을 담당하는 ROS2 노드."""

    def __init__(self, store: StateStore) -> None:
        super().__init__('dashboard_bridge')
        self.store = store
        self.cb_group = ReentrantCallbackGroup()

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
        reliable_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                  history=HistoryPolicy.KEEP_LAST, depth=10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        # 로봇 조인트
        self.create_subscription(JointState, '/dsr01/joint_states',
                                 self._on_joints, sensor_qos)
        # 비전 파이프라인 B 토픽
        self.create_subscription(String, '/dsr01/curobo/grasp_class',
                                 self._on_grasp_class, reliable_qos)
        self.create_subscription(PoseStamped, '/dsr01/curobo/pick_pose',
                                 lambda m: self._on_pose('pick_pose', m), reliable_qos)
        self.create_subscription(PoseStamped, '/dsr01/curobo/target_pose',
                                 lambda m: self._on_pose('target_pose', m), reliable_qos)
        self.create_subscription(PoseArray, '/dsr01/curobo/grasp_candidates',
                                 self._on_candidates, latched)
        self.create_subscription(String, '/dsr01/curobo/obstacles',
                                 self._on_obstacles, reliable_qos)
        # 매대 품목 재고 (webcam 발행 JSON: {"can":1,"bottle":0,"snack":1})
        self.create_subscription(String, '/dashboard/shelf_inventory',
                                 self._on_shelf_inv, reliable_qos)

        # 그리퍼 상태
        if _GRIPPER_STATE_AVAIL:
            self.create_subscription(GripperState, '/gripper_service/state',
                                     self._on_gripper, reliable_qos)
        else:
            self.get_logger().warn('GripperState 미가용 — 그리퍼 패널 비활성')

        # 에러 로그(rosout)
        if _ROSOUT_AVAIL:
            self.create_subscription(RosoutLog, '/rosout', self._on_rosout, sensor_qos)

        # TCP posx 주기 폴링 (서비스 → 토픽 없음)
        self.cli_posx = None
        if _POSX_AVAIL:
            self.cli_posx = self.create_client(
                GetCurrentPosx, '/dsr01/aux_control/get_current_posx',
                callback_group=self.cb_group)
            self.create_timer(0.5, self._poll_posx, callback_group=self.cb_group)
        else:
            self.get_logger().warn('GetCurrentPosx 미가용 — TCP 위치 비활성')

        self._posx_inflight = False
        self.get_logger().info('dashboard_bridge 준비 완료')

    # ── 콜백 ─────────────────────────────────────────────────────
    def _on_joints(self, msg: JointState) -> None:
        self.store.set_joints(list(msg.name), list(msg.position), list(msg.velocity))

    def _on_grasp_class(self, msg: String) -> None:
        self.store.set_grasp_class(msg.data)

    def _on_shelf_inv(self, msg: String) -> None:
        try:
            import json
            self.store.set_shelf_inventory(json.loads(msg.data))
        except Exception:
            pass

    def _on_pose(self, which: str, msg: PoseStamped) -> None:
        self.store.set_pose(which, _pose_to_dict(msg))

    def _on_candidates(self, msg: PoseArray) -> None:
        cands = [{'position': {'x': p.position.x, 'y': p.position.y, 'z': p.position.z},
                  'orientation': {'x': p.orientation.x, 'y': p.orientation.y,
                                  'z': p.orientation.z, 'w': p.orientation.w}}
                 for p in msg.poses]
        self.store.set_candidates(cands)

    def _on_obstacles(self, msg: String) -> None:
        self.store.set_obstacles(msg.data)

    def _on_gripper(self, msg) -> None:
        data = {
            'ready': bool(msg.ready),
            'torque_enabled': bool(msg.torque_enabled),
            'moving': bool(msg.moving),
            'in_position': bool(msg.in_position),
            'grasp_detected': bool(msg.grasp_detected),
            'object_lost': bool(msg.object_lost),
            'status': int(msg.status),
            'moving_status': int(msg.moving_status),
            'present_position': int(msg.present_position),   # 0~1150
            'goal_position': int(msg.goal_position),
            'present_current': int(msg.present_current),     # mA
            'current_limit': int(msg.current_limit),         # 목표/제한 전류
            'present_velocity': int(msg.present_velocity),
            'present_temperature': int(msg.present_temperature),
            'status_text': str(msg.status_text),
        }
        self.store.set_gripper(data, msg.present_current)

    def _on_rosout(self, msg) -> None:
        # level 40=ERROR, 50=FATAL 만 수집
        if msg.level >= 40:
            self.store.add_error({
                't': time.time(),
                'level': int(msg.level),
                'name': msg.name,
                'msg': msg.msg,
            })

    # ── posx 폴링 ────────────────────────────────────────────────
    def _poll_posx(self) -> None:
        if self.cli_posx is None or self._posx_inflight:
            return
        if not self.cli_posx.service_is_ready():
            return
        self._posx_inflight = True
        req = GetCurrentPosx.Request()
        req.ref = 0   # DR_BASE
        fut = self.cli_posx.call_async(req)
        fut.add_done_callback(self._on_posx)

    def _on_posx(self, fut) -> None:
        self._posx_inflight = False
        try:
            res = fut.result()
            if res and res.success and res.task_pos_info:
                arr = list(res.task_pos_info[0].data)
                sol = int(arr[6]) if len(arr) > 6 else None
                self.store.set_tcp(arr[:6], sol)
        except Exception as e:
            self.get_logger().warn(f'posx 폴링 실패: {e}', throttle_duration_sec=5.0)
