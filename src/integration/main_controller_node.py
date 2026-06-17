#!/usr/bin/env python3
"""smart-shelf-robot 통합 컨트롤러 (상태머신 + 예외처리 중심).

실제 노드 인터페이스 기준으로 작성 (curobo_planner_node / webcam_seg_node / gripper):

  [모션 — curobo_planner_node]  Trigger 서비스 4개 (※ /move_to_pick 는 없음):
    /move_to_shelf_view  /move_to_product_view  /move_to_place  /move_to_home
    픽은 서비스가 아니라 /dsr01/curobo/pick_pose 토픽 발행 → curobo 내부에서
    approach → move_stop → safe_grasp → lift 까지 수행. (grasp_class 로 파지힘 선택)

  [비전 — webcam_seg_node]  발행:
    /dsr01/curobo/grasp_class (String)   감지 클래스
    /dsr01/curobo/pick_pose   (PoseStamped) 파지 3D 위치
    /dsr01/curobo/target_pose (PoseStamped)
    /dsr01/curobo/obstacles   (String)

  [그리퍼]  /gripper_service/state (GripperState) 상태 모니터(20Hz),
    /gripper_service/set_position (SetPosition) 열기/닫기.
    (safe_grasp 액션은 curobo 가 pick 중에 호출하므로 컨트롤러는 중복 호출하지 않음.)

  [로봇]  /dsr01/joint_states (JointState) 생존 확인,
    /dsr01/motion/move_stop (MoveStop) 비상정지.

예외처리 설계 (이 파일의 핵심)
  1) 모든 상태 핸들러를 try/except 로 감싸 타이머가 절대 죽지 않게 한다.
  2) 모든 대기에 데드라인(타임아웃) — 무한 대기 금지. 초과 시 ERROR.
  3) 연결 워치독 — 현재 단계에 필요한 노드(robot/gripper/vision)가 stale 이면
     즉시 비상정지 + ERROR.
  4) 파지 검증 — pick 후 gripper_state.grasp_detected 확인. object_lost/미감지면
     제한 횟수까지 재시도, 초과하면 ERROR.
  5) 복구(RECOVER) — 그리퍼 열기 → 홈 복귀 → IDLE. 복구도 실패가 누적되면 HALT
     (안전정지 후 수동 개입 요구).
  6) 서비스/토픽 미연결 환경에서도 노드가 죽지 않도록 방어적 import + 가드.
"""
import os
import time
import traceback

import yaml

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState

# ── 선택 의존 (없으면 해당 기능만 비활성, 노드는 계속 동작) ──────────────
try:
    from dsr_gripper_tcp_interfaces.msg import GripperState
    from dsr_gripper_tcp_interfaces.srv import SetPosition
    _GRIPPER_AVAIL = True
except Exception:                       # pragma: no cover
    GripperState = SetPosition = None
    _GRIPPER_AVAIL = False

try:
    from dsr_msgs2.srv import MoveStop
    _MOVESTOP_AVAIL = True
except Exception:                       # pragma: no cover
    MoveStop = None
    _MOVESTOP_AVAIL = False


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _envi(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


# ── 타임아웃/재시도 파라미터 (env 로 조정 가능) ─────────────────────────
SVC_WAIT_TIMEOUT   = _envf('SSR_SVC_WAIT', 2.0)     # 서비스 가용 대기(초)
MOVE_TIMEOUT       = _envf('SSR_MOVE_TIMEOUT', 40.0)  # 이동 서비스 완료 데드라인
DETECT_TIMEOUT     = _envf('SSR_DETECT_TIMEOUT', 15.0)  # 비전 감지 대기 데드라인
GRASP_TIMEOUT      = _envf('SSR_GRASP_TIMEOUT', 25.0)   # pick→파지확정 데드라인
PLACE_OPEN_TIMEOUT = _envf('SSR_PLACE_TIMEOUT', 8.0)    # 그리퍼 열기 데드라인
STALE_ROBOT        = _envf('SSR_STALE_ROBOT', 2.0)   # joint_states 무수신 한계
STALE_GRIPPER      = _envf('SSR_STALE_GRIPPER', 2.0)  # gripper state 무수신 한계
STALE_VISION       = _envf('SSR_STALE_VISION', 12.0)  # 비전 토픽 무수신 한계
MAX_GRASP_RETRY    = _envi('SSR_MAX_GRASP_RETRY', 2)  # 파지 실패 재시도 횟수
MAX_RECOVER        = _envi('SSR_MAX_RECOVER', 3)      # 연속 복구 시도 한계
VALID_CLASSES      = ('bottle', 'can', 'snack_bag')


class TaskState:
    IDLE                 = 'IDLE'
    MOVE_TO_SHELF_VIEW   = 'MOVE_TO_SHELF_VIEW'
    SHELF_DETECTING      = 'SHELF_DETECTING'
    MOVE_TO_PRODUCT_VIEW = 'MOVE_TO_PRODUCT_VIEW'
    PRODUCT_DETECTING    = 'PRODUCT_DETECTING'
    MOVING_PICK          = 'MOVING_PICK'
    GRASPING             = 'GRASPING'
    MOVING_PLACE         = 'MOVING_PLACE'
    PLACING              = 'PLACING'
    DONE                 = 'DONE'
    ERROR                = 'ERROR'      # 예외 발생 → 복구 진입
    RECOVER              = 'RECOVER'    # 그리퍼 열기 → 홈
    HALT                 = 'HALT'       # 복구 실패 누적 → 안전정지(수동 개입)


def _now() -> float:
    return time.monotonic()


class MainControllerNode(Node):
    def __init__(self):
        super().__init__('main_controller_node')
        self.cb = ReentrantCallbackGroup()

        # config
        config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', '..', 'config')
        self._grasp_force = (self._load_yaml(
            os.path.join(config_dir, 'grasp_force_params.yaml')) or {}).get('grasp_force', {})
        self._place_targets = (self._load_yaml(
            os.path.join(config_dir, 'place_targets.yaml')) or {}).get('place_targets', {})

        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST, depth=10)
        sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            history=HistoryPolicy.KEEP_LAST, depth=10)

        # ── 구독 (비전/로봇/그리퍼 모니터) ──────────────────────────
        self.create_subscription(String, '/dsr01/curobo/grasp_class',
                                 self._on_grasp_class, reliable)
        self.create_subscription(PoseStamped, '/dsr01/curobo/pick_pose',
                                 self._on_pick_pose, reliable)
        self.create_subscription(JointState, '/dsr01/joint_states',
                                 self._on_joint_states, sensor)
        if _GRIPPER_AVAIL:
            self.create_subscription(GripperState, '/gripper_service/state',
                                     self._on_gripper_state, reliable)

        # ── 대시보드 operator 버튼 명령 ─────────────────────────────
        # 빈 슬롯(보충 대상) 선택과 "감지 완료" 확정을 비전 토픽 대신 버튼으로 받는다.
        #   start           : IDLE 에서 사이클 시작
        #   restock:<class> : 보충 대상 클래스(bottle/can/snack_bag) — SHELF_DETECTING 통과
        #   confirm         : 상품 감지 완료 확정 — PRODUCT_DETECTING 통과(픽 진입)
        #   abort           : 즉시 비상정지 + 복구
        #   reset           : ERROR/HALT 에서 IDLE 로 수동 리셋
        self.create_subscription(String, '/dashboard/operator_cmd',
                                 self._on_op_cmd, reliable)

        # ── 발행 (픽 트리거 — curobo 가 pick_pose 받아 내부 파지 수행) ──
        self.pub_pick_pose   = self.create_publisher(PoseStamped, '/dsr01/curobo/pick_pose', 10)
        self.pub_grasp_class = self.create_publisher(String, '/dsr01/curobo/grasp_class', 10)

        # ── 서비스/그리퍼 클라이언트 ────────────────────────────────
        self.cli_shelf_view   = self.create_client(Trigger, '/move_to_shelf_view', callback_group=self.cb)
        self.cli_product_view = self.create_client(Trigger, '/move_to_product_view', callback_group=self.cb)
        self.cli_place        = self.create_client(Trigger, '/move_to_place', callback_group=self.cb)
        self.cli_home         = self.create_client(Trigger, '/move_to_home', callback_group=self.cb)
        self.cli_gripper_open = (self.create_client(
            SetPosition, '/gripper_service/set_position', callback_group=self.cb)
            if _GRIPPER_AVAIL else None)
        self.cli_estop = (self.create_client(
            MoveStop, '/dsr01/motion/move_stop', callback_group=self.cb)
            if _MOVESTOP_AVAIL else None)

        # ── 상태 ────────────────────────────────────────────────────
        self.state = TaskState.IDLE
        self.object_class = None
        self.object_pose = None           # PoseStamped (비전 최신 pick_pose)
        self.gripper = {}                 # 최신 GripperState dict
        self._last_seen = {}              # 소스별 마지막 수신 monotonic
        self._fut = None                  # 진행 중 서비스 future
        self._deadline = 0.0              # 현재 대기/상태 데드라인
        self._grasp_retry = 0
        self._recover_attempt = 0
        self._tick = 0
        self._estopped = False
        self._op = {}                     # operator 버튼 one-shot 플래그

        self.timer = self.create_timer(0.2, self._safe_tick)  # 5Hz
        self.get_logger().info('main_controller_node 시작 (예외처리 상태머신)')

    # ════════════════════════════════════════════════════════════════
    # 콜백 (수신 기록만 — 상태 전이는 상태머신에서)
    # ════════════════════════════════════════════════════════════════
    def _on_grasp_class(self, msg):
        cls = (msg.data or '').strip().lower()
        if cls in VALID_CLASSES:
            self.object_class = cls
        self._last_seen['vision'] = _now()

    def _on_pick_pose(self, msg):
        self.object_pose = msg
        self._last_seen['vision'] = _now()

    def _on_joint_states(self, _msg):
        self._last_seen['robot'] = _now()

    def _on_gripper_state(self, msg):
        self.gripper = {
            'ready': bool(msg.ready),
            'moving': bool(msg.moving),
            'grasp_detected': bool(msg.grasp_detected),
            'object_lost': bool(msg.object_lost),
            'present_position': int(msg.present_position),
            'present_current': int(msg.present_current),
        }
        self._last_seen['gripper'] = _now()

    def _on_op_cmd(self, msg):
        cmd = (msg.data or '').strip().lower()
        if not cmd:
            return
        self.get_logger().info(f'[operator] 버튼 명령: {cmd}')
        if cmd == 'start':
            self._op['start'] = True
        elif cmd.startswith('restock:'):
            cls = cmd.split(':', 1)[1].strip()
            if cls in VALID_CLASSES:
                self._op['restock'] = cls
            else:
                self.get_logger().warn(f'[operator] 알 수 없는 클래스: {cls}')
        elif cmd == 'confirm':
            self._op['confirm'] = True
        elif cmd == 'abort':
            self._op['abort'] = True
        elif cmd == 'reset':
            self._op['reset'] = True
        else:
            self.get_logger().warn(f'[operator] 미지원 명령: {cmd}')

    def _op_take(self, key):
        """one-shot 플래그 소비 (있으면 꺼내고 True/값 반환, 없으면 None)."""
        return self._op.pop(key, None)

    # ════════════════════════════════════════════════════════════════
    # 상태머신 (예외처리의 핵심 — 절대 죽지 않는 틱)
    # ════════════════════════════════════════════════════════════════
    def _safe_tick(self):
        """타이머 콜백. 어떤 예외가 나도 타이머는 살아남고 ERROR 로 보낸다."""
        try:
            # 0) operator 전역 명령 — abort(즉시 정지) / reset(HALT·ERROR 해제)
            if self._op_take('abort'):
                self.get_logger().warn('[operator] ABORT 요청')
                raise RuntimeError('operator abort')
            if self._op_take('reset'):
                if self.state in (TaskState.HALT, TaskState.ERROR):
                    self.get_logger().warn('[operator] RESET → IDLE')
                    self._halt_logged = False
                    self._reset_and_idle()
                    return

            # 1) 연결 워치독 — 활성 작업 중 필요한 노드가 죽으면 즉시 안전정지
            dead = self._dead_sources()
            if dead and self._is_active_state():
                raise RuntimeError(f'필수 노드 끊김: {dead}')

            handler = self._handlers().get(self.state)
            if handler is None:
                raise RuntimeError(f'알 수 없는 상태: {self.state}')
            handler()
        except Exception as e:
            tb = traceback.format_exc()
            self.get_logger().error(f'[{self.state}] 예외 발생: {e}\n{tb}')
            self._enter_error(reason=str(e), estop=True)

    def _handlers(self):
        return {
            TaskState.IDLE:                 self._on_idle,
            TaskState.MOVE_TO_SHELF_VIEW:   self._on_move_shelf_view,
            TaskState.SHELF_DETECTING:      self._on_shelf_detecting,
            TaskState.MOVE_TO_PRODUCT_VIEW: self._on_move_product_view,
            TaskState.PRODUCT_DETECTING:    self._on_product_detecting,
            TaskState.MOVING_PICK:          self._on_moving_pick,
            TaskState.GRASPING:             self._on_grasping,
            TaskState.MOVING_PLACE:         self._on_moving_place,
            TaskState.PLACING:              self._on_placing,
            TaskState.DONE:                 self._on_done,
            TaskState.ERROR:                self._on_error,
            TaskState.RECOVER:              self._on_recover,
            TaskState.HALT:                 self._on_halt,
        }

    def _is_active_state(self) -> bool:
        """로봇을 실제로 움직이는(연결이 필수인) 상태인지."""
        return self.state in (
            TaskState.MOVE_TO_SHELF_VIEW, TaskState.MOVE_TO_PRODUCT_VIEW,
            TaskState.MOVING_PICK, TaskState.GRASPING, TaskState.MOVING_PLACE,
            TaskState.PLACING, TaskState.DONE)

    # ── 개별 상태 ────────────────────────────────────────────────────
    def _on_idle(self):
        # 시작은 대시보드 'start' 버튼으로 (자동 시작 안 함). 로봇 연결은 사전 확인.
        if not self._op_take('start'):
            return
        if 'robot' not in self._last_seen:
            self.get_logger().warn('[IDLE] start 눌렀지만 로봇(joint_states) 미연결 — 대기')
            return
        self.get_logger().info('[IDLE] 작업 시작 (operator start)')
        self._transition(TaskState.MOVE_TO_SHELF_VIEW)

    def _on_move_shelf_view(self):
        self._await_move(self.cli_shelf_view, '매대뷰 이동',
                         nxt=TaskState.SHELF_DETECTING)

    def _on_shelf_detecting(self):
        # 빈 슬롯(보충 대상)은 대시보드 'restock:<class>' 버튼으로 받는다 (operator-paced
        # → 타임아웃 없음). 버튼이 클래스를 주면 그걸 보충 대상으로 진행.
        cls = self._op_take('restock')
        if cls in VALID_CLASSES:
            self.object_class = cls
            self.get_logger().info(f'[SHELF_DETECTING] 보충 대상(버튼): {cls}')
            self._transition(TaskState.MOVE_TO_PRODUCT_VIEW)

    def _on_move_product_view(self):
        self._await_move(self.cli_product_view, '상품뷰 이동',
                         nxt=TaskState.PRODUCT_DETECTING)

    def _on_product_detecting(self):
        # "감지 완료"는 대시보드 'confirm' 버튼으로 확정 (operator-paced → 타임아웃 없음).
        # 단, 실제 픽에는 비전의 pick_pose(3D 위치)가 반드시 있어야 한다.
        if not self._op_take('confirm'):
            return
        if self.object_pose is None:
            self.get_logger().warn(
                '[PRODUCT_DETECTING] confirm 눌렀지만 pick_pose 없음 — 비전 감지 후 다시 확정')
            return
        self.get_logger().info(
            f'[PRODUCT_DETECTING] 감지 완료 확정(버튼) {self.object_class} → 픽 진입')
        self._transition(TaskState.MOVING_PICK)

    def _on_moving_pick(self):
        # 픽 트리거: grasp_class + pick_pose 재발행 → curobo 가 approach+grasp+lift.
        if self._fut is None:  # 1회만 발행
            if self.object_pose is None:
                raise RuntimeError('pick_pose 없음 — 픽 트리거 불가')
            gc = String(); gc.data = self.object_class
            self.pub_grasp_class.publish(gc)
            self.object_pose.header.stamp = self.get_clock().now().to_msg()
            self.pub_pick_pose.publish(self.object_pose)
            self._fut = 'published'   # 발행 표시(서비스 future 아님)
            self.get_logger().info(f'[MOVING_PICK] pick 트리거 발행 ({self.object_class})')
            self._transition(TaskState.GRASPING)

    def _on_grasping(self):
        # curobo 가 내부에서 파지 수행 → 그리퍼 상태로 결과 검증.
        if not self.gripper:
            if self._deadline_expired(GRASP_TIMEOUT):
                raise TimeoutError('그리퍼 상태 수신 없음')
            return
        if self.gripper.get('grasp_detected'):
            self.get_logger().info('[GRASPING] 파지 성공 (grasp_detected)')
            self._grasp_retry = 0
            self._transition(TaskState.MOVING_PLACE)
            return
        if self.gripper.get('object_lost'):
            self._handle_grasp_failure('object_lost')
            return
        if self._deadline_expired(GRASP_TIMEOUT):
            self._handle_grasp_failure('파지 확정 타임아웃')

    def _on_moving_place(self):
        # curobo 가 place_targets.yaml 기반으로 배치 위치 이동 (grasp_class 사용).
        self._await_move(self.cli_place, '배치 위치 이동',
                         nxt=TaskState.PLACING)

    def _on_placing(self):
        # 그리퍼 열기(release). SetPosition(0).
        self._await_gripper_open('배치 release', nxt=TaskState.DONE)

    def _on_done(self):
        self._await_move(self.cli_home, '홈 복귀', nxt=None,
                         on_success=self._cycle_complete)

    # ── 예외/복구 상태 ───────────────────────────────────────────────
    def _on_error(self):
        # ERROR 진입 직후 1틱: 비상정지 보장 후 복구로.
        self.get_logger().warn(f'[ERROR] 복구 진입 (시도 {self._recover_attempt+1}/{MAX_RECOVER})')
        if self._recover_attempt >= MAX_RECOVER:
            self._transition(TaskState.HALT)
            return
        self._recover_attempt += 1
        self._transition(TaskState.RECOVER)

    def _on_recover(self):
        # 복구 순서: 그리퍼 열기 → 홈 복귀 → IDLE. 단계마다 실패해도 다음으로 진행.
        step = getattr(self, '_recover_step', 0)
        if step == 0:
            self.get_logger().warn('[RECOVER] 그리퍼 열기')
            self._recover_step = 1
            self._fut = None
            self._safe_call(self.cli_gripper_open, self._open_req())
        elif step == 1:
            if self._fut_settled(timeout=PLACE_OPEN_TIMEOUT):
                self.get_logger().warn('[RECOVER] 홈 복귀')
                self._recover_step = 2
                self._fut = None
                self._safe_call(self.cli_home, Trigger.Request())
        else:
            if self._fut_settled(timeout=MOVE_TIMEOUT):
                self.get_logger().warn('[RECOVER] 복구 완료 → IDLE')
                self._recover_step = 0
                self._reset_and_idle()

    def _on_halt(self):
        # 복구 누적 실패 → 안전정지 유지. 수동 개입(리셋) 전까지 대기.
        if not getattr(self, '_halt_logged', False):
            self._emergency_stop()
            self.get_logger().fatal(
                '[HALT] 자동 복구 실패 — 안전정지. 수동 점검 후 재시작 필요.')
            self._halt_logged = True

    # ════════════════════════════════════════════════════════════════
    # 예외처리 헬퍼
    # ════════════════════════════════════════════════════════════════
    def _dead_sources(self):
        """현재 stale(무수신) 인 필수 소스 목록."""
        dead = []
        now = _now()
        if now - self._last_seen.get('robot', 0.0) > STALE_ROBOT:
            dead.append('robot')
        if _GRIPPER_AVAIL and self.state in (TaskState.GRASPING, TaskState.PLACING):
            if now - self._last_seen.get('gripper', 0.0) > STALE_GRIPPER:
                dead.append('gripper')
        return dead

    def _deadline_expired(self, timeout: float) -> bool:
        if self._deadline <= 0.0:
            self._deadline = _now() + timeout
        return _now() > self._deadline

    def _await_move(self, client, tag, nxt, on_success=None):
        """이동 Trigger 서비스: 호출 → 데드라인 내 응답 대기 → 성공/실패 분기.
        실패·타임아웃·미연결은 모두 예외로 올려 ERROR 로 보낸다."""
        if self._fut is None:
            if not client.service_is_ready() and \
               not client.wait_for_service(timeout_sec=SVC_WAIT_TIMEOUT):
                raise RuntimeError(f'{tag}: 서비스 미연결')
            self._fut = client.call_async(Trigger.Request())
            self._deadline = _now() + MOVE_TIMEOUT
            self.get_logger().info(f'[{self.state}] {tag} 호출')
            return
        if _now() > self._deadline:
            raise TimeoutError(f'{tag}: 응답 타임아웃({MOVE_TIMEOUT}s)')
        if self._fut.done():
            res = self._fut.result()
            self._fut = None
            if not getattr(res, 'success', False):
                raise RuntimeError(f'{tag} 실패: {getattr(res, "message", "")}')
            self.get_logger().info(f'[{self.state}] {tag} 완료')
            (on_success or (lambda: self._transition(nxt)))()

    def _await_gripper_open(self, tag, nxt):
        if not _GRIPPER_AVAIL or self.cli_gripper_open is None:
            self.get_logger().warn(f'{tag}: 그리퍼 서비스 미가용 — 건너뜀')
            self._transition(nxt)
            return
        if self._fut is None:
            if not self.cli_gripper_open.service_is_ready() and \
               not self.cli_gripper_open.wait_for_service(timeout_sec=SVC_WAIT_TIMEOUT):
                raise RuntimeError(f'{tag}: set_position 미연결')
            self._fut = self.cli_gripper_open.call_async(self._open_req())
            self._deadline = _now() + PLACE_OPEN_TIMEOUT
            self.get_logger().info(f'[{self.state}] {tag} (그리퍼 열기)')
            return
        if _now() > self._deadline:
            raise TimeoutError(f'{tag}: 그리퍼 응답 타임아웃')
        if self._fut.done():
            res = self._fut.result()
            self._fut = None
            if not getattr(res, 'success', True):
                raise RuntimeError(f'{tag} 실패: {getattr(res, "message", "")}')
            self._transition(nxt)

    def _open_req(self):
        req = SetPosition.Request()
        req.position = 0          # 0 = 완전 열기
        req.timeout_sec = PLACE_OPEN_TIMEOUT
        return req

    def _handle_grasp_failure(self, reason):
        self.get_logger().warn(f'[GRASPING] 파지 실패: {reason} '
                               f'(재시도 {self._grasp_retry}/{MAX_GRASP_RETRY})')
        if self._grasp_retry < MAX_GRASP_RETRY:
            self._grasp_retry += 1
            # 그리퍼 열고 다시 상품 감지부터 재시도
            self._fut = None
            self._safe_call(self.cli_gripper_open, self._open_req())
            self.object_pose = None
            self._transition(TaskState.PRODUCT_DETECTING)
        else:
            raise RuntimeError(f'파지 재시도 한계 초과: {reason}')

    def _safe_call(self, client, req):
        """결과를 기다리지 않는 fire-and-forget 서비스 호출 (복구용). 미연결이면 무시."""
        try:
            if client is not None and (client.service_is_ready()
                                       or client.wait_for_service(timeout_sec=0.5)):
                self._fut = client.call_async(req)
            else:
                self._fut = None
        except Exception as e:
            self.get_logger().warn(f'_safe_call 실패: {e}')
            self._fut = None

    def _fut_settled(self, timeout: float) -> bool:
        """복구 단계용: future 가 완료됐거나(또는 None), 타임아웃이면 True(다음 단계로)."""
        if self._fut is None:
            return True
        if self._deadline <= 0.0:
            self._deadline = _now() + timeout
        if self._fut.done() or _now() > self._deadline:
            self._fut = None
            self._deadline = 0.0
            return True
        return False

    def _emergency_stop(self):
        if self._estopped:
            return
        self._estopped = True
        try:
            if self.cli_estop is not None and (self.cli_estop.service_is_ready()
                                               or self.cli_estop.wait_for_service(timeout_sec=0.5)):
                req = MoveStop.Request()
                req.stop_mode = 0     # DR_QSTOP_STO — Quick stop
                self.cli_estop.call_async(req)
                self.get_logger().warn('[E-STOP] move_stop(Quick) 호출')
            else:
                self.get_logger().warn('[E-STOP] move_stop 미연결 — 물리 E-stop 필요')
        except Exception as e:
            self.get_logger().error(f'[E-STOP] 호출 실패: {e}')

    def _enter_error(self, reason='', estop=False):
        if estop:
            self._emergency_stop()
        self.get_logger().error(f'→ ERROR 진입: {reason}')
        self._transition(TaskState.ERROR)

    # ── 전이/리셋 ────────────────────────────────────────────────────
    def _transition(self, new_state):
        if new_state is None:
            return
        self.get_logger().info(f'State: {self.state} → {new_state}')
        self.state = new_state
        self._fut = None
        self._deadline = 0.0
        self._tick = 0

    def _cycle_complete(self):
        self.get_logger().info('=== 사이클 완료 ===')
        self._reset_and_idle()

    def _reset_and_idle(self):
        self.object_class = None
        self.object_pose = None
        self._grasp_retry = 0
        self._recover_attempt = 0
        self._recover_step = 0
        self._estopped = False
        self._transition(TaskState.IDLE)

    def _load_yaml(self, path):
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.get_logger().warn(f'yaml 로드 실패: {path} → {e}')
            return None


def main(args=None):
    rclpy.init(args=args)
    node = MainControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._emergency_stop()    # 종료 시 안전정지 시도
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
