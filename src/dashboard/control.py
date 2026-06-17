"""제어 표면 — GUI 버튼/슬라이더가 호출하는 ROS2 서비스·액션 래퍼.

코드에서 확인된 제어 인터페이스:
  그리퍼:
    /gripper_service/set_position        SetPosition  (0~1150, 0=열기)
    /gripper_service/safe_grasp          SafeGrasp(action) target_position·max_current(파지힘)·…
    /gripper_service/set_torque          SetTorque
    /gripper_service/set_motion_profile  SetMotionProfile
  로봇:
    /move_to_home|shelf_view|product_view|place   std_srvs/Trigger  (motion 노드 제공)
    /dsr01/motion/move_joint             MoveJoint  (조그/절대 이동)
    /dsr01/motion/move_stop              MoveStop   (E-STOP: 0=Quick stop)
  파이프라인 트리거(토픽 발행):
    /dsr01/curobo/pick_pose, /target_pose, /grasp_class, /obstacles

서비스 호출은 응답을 기다려 실제 성공/실패를 반환한다 (executor 는 별도 스레드 spin).
액션(safe_grasp)은 결과 콜백으로 metrics 집계.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup

from std_srvs.srv import Trigger
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# 방어적 import (인터페이스 미빌드 환경 대비)
try:
    from dsr_msgs2.srv import MoveJoint, MoveStop
    _DSR_AVAIL = True
except Exception:                       # pragma: no cover
    MoveJoint = MoveStop = None
    _DSR_AVAIL = False

try:
    from dsr_gripper_tcp_interfaces.srv import (SetPosition, SetTorque,
                                                SetMotionProfile)
    from dsr_gripper_tcp_interfaces.action import SafeGrasp
    _GRIP_AVAIL = True
except Exception:                       # pragma: no cover
    SetPosition = SetTorque = SetMotionProfile = SafeGrasp = None
    _GRIP_AVAIL = False


# 명명 자세 트리거 서비스 (motion 노드)
NAMED_MOVES = {
    'home': '/move_to_home',
    'shelf_view': '/move_to_shelf_view',
    'product_view': '/move_to_product_view',
    'place': '/move_to_place',
}
# 그리퍼 안전 범위
POSITION_MAX = 1150
CURRENT_MAX = 820       # RH-P12-RN 전류 상한(근사). UI 슬라이더 클램프용.


class ControlClients:
    """제어 클라이언트 묶음. dashboard_node 의 자식 노드에 attach 한다."""

    def __init__(self, node: Node, on_grasp_result: Optional[Callable] = None) -> None:
        self.node = node
        self.log = node.get_logger()
        self.cb = ReentrantCallbackGroup()
        self._on_grasp_result = on_grasp_result
        self._lock = threading.Lock()

        # 로봇 명명 이동
        self.named = {k: node.create_client(Trigger, v, callback_group=self.cb)
                      for k, v in NAMED_MOVES.items()}

        # 두산 직접 제어
        self.cli_movej = self.cli_stop = None
        if _DSR_AVAIL:
            self.cli_movej = node.create_client(
                MoveJoint, '/dsr01/motion/move_joint', callback_group=self.cb)
            self.cli_stop = node.create_client(
                MoveStop, '/dsr01/motion/move_stop', callback_group=self.cb)

        # 그리퍼
        self.cli_setpos = self.cli_torque = self.cli_profile = None
        self.act_grasp = None
        if _GRIP_AVAIL:
            self.cli_setpos = node.create_client(
                SetPosition, '/gripper_service/set_position', callback_group=self.cb)
            self.cli_torque = node.create_client(
                SetTorque, '/gripper_service/set_torque', callback_group=self.cb)
            self.cli_profile = node.create_client(
                SetMotionProfile, '/gripper_service/set_motion_profile',
                callback_group=self.cb)
            self.act_grasp = ActionClient(
                node, SafeGrasp, '/gripper_service/safe_grasp', callback_group=self.cb)

        # 파이프라인 트리거 퍼블리셔
        self.pub_pick = node.create_publisher(PoseStamped, '/dsr01/curobo/pick_pose', 10)
        self.pub_target = node.create_publisher(PoseStamped, '/dsr01/curobo/target_pose', 10)
        self.pub_grasp_class = node.create_publisher(String, '/dsr01/curobo/grasp_class', 10)
        self.pub_obstacles = node.create_publisher(String, '/dsr01/curobo/obstacles', 10)
        # main_controller 로 보내는 operator 버튼 명령
        self.pub_operator_cmd = node.create_publisher(String, '/dashboard/operator_cmd', 10)

    # ── 헬퍼 ─────────────────────────────────────────────────────
    @staticmethod
    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _call(self, cli, req, label: str, wait: float = 4.0) -> dict:
        if cli is None:
            return {'ok': False, 'error': f'{label}: 인터페이스 미가용'}
        if not cli.service_is_ready() and not cli.wait_for_service(timeout_sec=0.5):
            return {'ok': False, 'error': f'{label}: 서비스 대기 시간 초과'}
        fut = cli.call_async(req)
        self.log.info(f'[control] {label} 호출')
        # 응답을 기다려 실제 성공/실패를 반환 (executor 는 별도 스레드에서 spin 중).
        ev = threading.Event()
        fut.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout=wait):
            return {'ok': False, 'error': f'{label}: 응답 시간 초과(서비스 무응답)'}
        try:
            res = fut.result()
        except Exception as e:
            return {'ok': False, 'error': f'{label}: {e}'}
        success = bool(getattr(res, 'success', True))
        msg = getattr(res, 'message', '') or ''
        out = {'ok': success, 'label': label, 'message': msg}
        if hasattr(res, 'present_position'):
            out['present_position'] = int(res.present_position)
        if hasattr(res, 'goal_position'):
            out['goal_position'] = int(res.goal_position)
        return out

    # ── 통합 컨트롤러 operator 명령 (토픽 발행, fire-and-forget) ──
    def operator_cmd(self, cmd: str) -> dict:
        """main_controller 의 /dashboard/operator_cmd 로 명령 발행.
        지원: start | restock:<class> | confirm | abort | reset"""
        m = String()
        m.data = str(cmd).strip()
        self.pub_operator_cmd.publish(m)
        self.log.info(f'[control] operator_cmd: {m.data}')
        return {'ok': True, 'label': f'operator:{m.data}'}

    # ── 로봇 제어 ────────────────────────────────────────────────
    def move_named(self, name: str) -> dict:
        cli = self.named.get(name)
        if cli is None:
            return {'ok': False, 'error': f'알 수 없는 자세: {name}'}
        return self._call(cli, Trigger.Request(), f'move:{name}')

    def move_joint(self, pos_deg: list, vel: float = 30.0, acc: float = 30.0,
                   relative: bool = False) -> dict:
        if not _DSR_AVAIL:
            return {'ok': False, 'error': 'MoveJoint 미가용'}
        if len(pos_deg) != 6:
            return {'ok': False, 'error': 'pos 는 6개여야 함'}
        req = MoveJoint.Request()
        req.pos = [float(v) for v in pos_deg]
        req.vel = float(vel)
        req.acc = float(acc)
        req.mode = 1 if relative else 0    # RELATIVE / ABSOLUTE
        return self._call(self.cli_movej, req, 'move_joint')

    def estop(self, mode: int = 0) -> dict:
        """비상정지. mode 0=Quick stop(STO 없음)."""
        if not _DSR_AVAIL:
            return {'ok': False, 'error': 'MoveStop 미가용'}
        req = MoveStop.Request()
        req.stop_mode = int(mode)
        return self._call(self.cli_stop, req, 'E-STOP')

    # ── 그리퍼 제어 ──────────────────────────────────────────────
    def set_position(self, position: int, timeout: float = 5.0) -> dict:
        if not _GRIP_AVAIL:
            return {'ok': False, 'error': 'SetPosition 미가용'}
        req = SetPosition.Request()
        req.position = int(self._clamp(position, 0, POSITION_MAX))
        req.timeout_sec = float(timeout)
        return self._call(self.cli_setpos, req, f'set_position({req.position})')

    def set_torque(self, enable: bool) -> dict:
        if not _GRIP_AVAIL:
            return {'ok': False, 'error': 'SetTorque 미가용'}
        req = SetTorque.Request()
        # SetTorque 필드명은 환경에 따라 다를 수 있어 방어적으로 설정
        for field in ('enable', 'torque_enable', 'on'):
            if hasattr(req, field):
                setattr(req, field, bool(enable))
                break
        return self._call(self.cli_torque, req, f'set_torque({enable})')

    def set_motion_profile(self, velocity: int, acceleration: int) -> dict:
        if not _GRIP_AVAIL:
            return {'ok': False, 'error': 'SetMotionProfile 미가용'}
        req = SetMotionProfile.Request()
        if hasattr(req, 'velocity'):
            req.velocity = int(velocity)
        if hasattr(req, 'acceleration'):
            req.acceleration = int(acceleration)
        return self._call(self.cli_profile, req, 'set_motion_profile')

    def safe_grasp(self, target_position: int, max_current: int,
                   current_delta_threshold: int = 20, timeout: float = 5.0) -> dict:
        """파지 액션. max_current 가 파지힘(mA). 결과는 metrics 콜백으로 전달."""
        if not _GRIP_AVAIL or self.act_grasp is None:
            return {'ok': False, 'error': 'SafeGrasp 미가용'}
        if not self.act_grasp.server_is_ready() and \
           not self.act_grasp.wait_for_server(timeout_sec=0.5):
            return {'ok': False, 'error': 'SafeGrasp 액션 서버 대기 초과'}
        goal = SafeGrasp.Goal()
        goal.target_position = int(self._clamp(target_position, 0, POSITION_MAX))
        goal.max_current = int(self._clamp(max_current, 0, CURRENT_MAX))
        goal.current_delta_threshold = int(current_delta_threshold)
        goal.timeout_sec = float(timeout)
        send_fut = self.act_grasp.send_goal_async(goal)
        send_fut.add_done_callback(self._on_grasp_accepted)
        self.log.info(f'[control] safe_grasp(pos={goal.target_position}, '
                      f'maxI={goal.max_current})')
        return {'ok': True, 'label': 'safe_grasp'}

    def _on_grasp_accepted(self, fut) -> None:
        try:
            handle = fut.result()
            if not handle.accepted:
                self.log.warn('safe_grasp 거부됨')
                return
            handle.get_result_async().add_done_callback(self._on_grasp_done)
        except Exception as e:
            self.log.warn(f'safe_grasp accept 처리 실패: {e}')

    def _on_grasp_done(self, fut) -> None:
        try:
            result = fut.result().result
            payload = {
                'success': bool(result.success),
                'grasp_detected': bool(result.grasp_detected),
                'object_lost': bool(result.object_lost),
                'final_position': int(result.final_position),
                'final_current': int(result.final_current),
                'message': str(result.message),
            }
            self.log.info(f'[control] safe_grasp 결과: {payload}')
            if self._on_grasp_result:
                self._on_grasp_result(payload)
        except Exception as e:
            self.log.warn(f'safe_grasp 결과 처리 실패: {e}')
